"""
ecgfm_finetune.py
=================
Fine-tune the ECG-FM foundation model on PTB-XL for 5-class superclass prediction.

Architecture:
  ECGFMEncoder (pretrained, 90.4M params)
    → mean-pool → (B, 768)
  concat with 14-dim aux features → (B, 782)
  MLP head: Linear(782→256) → GELU → Dropout(0.3) → Linear(256→5)

Two-stage training:
  Stage 1 — Frozen encoder (linear probe on cached embeddings)
    Precomputes all 768-dim embeddings once (saves to models/ecgfm_embeddings.npz).
    Trains MLP head on (embedding, aux) → CPU training is seconds per epoch.
    lr_head = 1e-3, batch = 256, up to 40 epochs, patience = 10
    Saved to models/ecgfm_stage1.pt

  Stage 2 — Partial unfreeze: top 4 transformer layers + head (differential LR)
    Bottom 8 layers frozen. lr_top_layers = 1e-5, lr_head = 1e-4, batch = 8, up to 20 epochs, patience = 8
    Saved to models/ecgfm_stage2.pt

Usage:
    python ecgfm_finetune.py --precompute      # just precompute embeddings (~2h on CPU)
    python ecgfm_finetune.py --stage 1         # stage 1 (precomputes if needed, then trains head)
    python ecgfm_finetune.py --stage 2         # stage 2 (needs stage 1 checkpoint)
    python ecgfm_finetune.py                   # both stages
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

os.chdir(Path(__file__).parent)

# Use all logical CPU threads for faster transformer inference
import torch as _torch
_torch.set_num_threads(16)
del _torch

from cnn_classifier import (
    SUPERCLASS_LABELS, LABEL_TO_IDX,
    N_AUX, N_CLASSES, SIGNAL_LEN, N_LEADS,
    load_dataset, load_dataset_demographics,
    _load_raw_signal, extract_voltage_features, augment_signal,
    AsymmetricLoss,
)
from ecgfm_encoder import ECGFMEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_FM         = "models/ecgfm/mimic_iv_ecg_physionet_pretrained.pt"
CKPT_S1         = "models/ecgfm_stage1.pt"
CKPT_S2         = "models/ecgfm_stage2.pt"
EMBED_CACHE     = "models/ecgfm_embeddings.npz"   # precomputed 768-dim embeddings

HYP_IDX    = LABEL_TO_IDX["HYP"]
EMBED_DIM  = 768

RESAMPLE_RATIOS = {
    0: 0.5,   # NORM: downsample
    1: 1.5,   # MI
    2: 0.8,   # STTC
    3: 1.2,   # HYP
    4: 0.9,   # CD
}


# ---------------------------------------------------------------------------
# Raw-signal dataset (used for Stage 2 and embedding precomputation)
# ---------------------------------------------------------------------------

class ECGFMDataset(Dataset):
    """Returns (signal_T12, aux_14, label) — raw mV signal in (T, 12) format."""
    def __init__(self, paths, labels, raw_cache, aux_cache, demographics, augment=False):
        self.paths       = paths
        self.labels      = labels
        self.raw_cache   = raw_cache
        self.aux_cache   = aux_cache
        self.demographics = demographics
        self.augment     = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        sig12 = self.raw_cache[path].copy()   # (12, 5000) raw mV
        if self.augment:
            sig12 = augment_signal(sig12)
        sig_t12 = torch.from_numpy(sig12.T.astype(np.float32))   # (5000, 12)
        aux = self.aux_cache.get(path, np.zeros(N_AUX, dtype=np.float32)).copy()
        sf, an = self.demographics.get(path, (0.0, 0.625))
        aux[8], aux[9] = sf, an
        return sig_t12, torch.from_numpy(aux), torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Precomputed-embedding dataset (Stage 1 fast training)
# ---------------------------------------------------------------------------

class ECGFMEmbedDataset(Dataset):
    """
    Uses precomputed 768-dim embeddings.
    Returns (embedding_768, aux_14, label) — no encoder forward pass needed.
    Augmentation not applied here (embeddings are fixed for frozen encoder).
    """
    def __init__(self, paths, labels, embed_cache, aux_cache, demographics):
        self.paths       = paths
        self.labels      = labels
        self.embed_cache = embed_cache    # path -> (768,) float32
        self.aux_cache   = aux_cache
        self.demographics = demographics

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        emb  = torch.from_numpy(self.embed_cache[path].copy())   # (768,)
        aux  = self.aux_cache.get(path, np.zeros(N_AUX, dtype=np.float32)).copy()
        sf, an = self.demographics.get(path, (0.0, 0.625))
        aux[8], aux[9] = sf, an
        return emb, torch.from_numpy(aux), torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Preload raw signals
# ---------------------------------------------------------------------------

def preload_raw(all_paths):
    raw_cache = {}
    aux_cache = {}
    unique = list(set(all_paths))
    print(f"\n  Pre-loading {len(unique)} signals (raw mV + aux) ...")
    t0 = time.time()
    for i, path in enumerate(unique):
        if (i + 1) % 2000 == 0 or (i + 1) == len(unique):
            e = time.time() - t0
            print(f"    {i+1}/{len(unique)}  ({(i+1)/e:.0f} rec/s)", end="\r", flush=True)
        raw = _load_raw_signal(path)
        raw_cache[path] = raw
        aux_cache[path] = extract_voltage_features(raw)
    print(f"\n  Done in {time.time()-t0:.0f}s", flush=True)
    return raw_cache, aux_cache


# ---------------------------------------------------------------------------
# Embedding precomputation (one-time, cached to disk)
# ---------------------------------------------------------------------------

def precompute_embeddings(paths, raw_cache, aux_cache, enc, device, batch_size=16,
                          cache_path=EMBED_CACHE, save_every=512):
    """
    Run all unique signals through frozen ECGFMEncoder and return path → embedding dict.
    Saves a checkpoint every `save_every` new samples so the job resumes safely if killed.
    On restart, already-computed paths are skipped automatically.
    """
    unique_paths = list(set(paths))

    # Resume: load partial cache if present
    embed_cache = {}
    saved_aux   = {}
    if Path(cache_path).exists():
        try:
            ec, ac = load_embed_cache(cache_path)
            embed_cache.update(ec)
            saved_aux.update(ac)
            print(f"  Resuming — {len(embed_cache)}/{len(unique_paths)} embeddings already cached.",
                  flush=True)
        except Exception as e:
            print(f"  Could not load partial cache ({e}), starting fresh.", flush=True)

    todo_paths = [p for p in unique_paths if p not in embed_cache]
    n_todo  = len(todo_paths)
    n_total = len(unique_paths)

    if n_todo == 0:
        print(f"  All {n_total} embeddings already cached — nothing to compute.", flush=True)
        return embed_cache

    print(f"\n  Precomputing {n_todo}/{n_total} embeddings (batch={batch_size}) ...", flush=True)
    t0 = time.time()
    enc.eval()
    done_this_run = 0

    for start in range(0, n_todo, batch_size):
        batch_paths = todo_paths[start : start + batch_size]
        sigs = np.stack([raw_cache[p].T for p in batch_paths], axis=0)   # (B, 5000, 12)
        sig_t = torch.from_numpy(sigs).to(device)
        with torch.no_grad():
            embs = enc(sig_t).cpu().numpy()   # (B, 768)
        for path, emb in zip(batch_paths, embs):
            embed_cache[path] = emb.astype(np.float32)
        done_this_run += len(batch_paths)

        if done_this_run % (batch_size * 20) == 0 or done_this_run == n_todo:
            elapsed = time.time() - t0
            rate    = done_this_run / (elapsed + 1e-9)
            eta     = (n_todo - done_this_run) / (rate + 1e-9)
            print(f"    {len(embed_cache)}/{n_total}  {rate:.1f} samp/s  ETA {eta/60:.0f}m",
                  end="\r", flush=True)

        # Incremental save every `save_every` new samples
        if done_this_run % save_every == 0:
            _save_partial(embed_cache, aux_cache, saved_aux, cache_path)

    _save_partial(embed_cache, aux_cache, saved_aux, cache_path)
    print(f"\n  Precomputed {n_todo} new embeddings in {(time.time()-t0)/60:.1f}m "
          f"({n_total} total cached)", flush=True)
    return embed_cache


def _save_partial(embed_cache, aux_cache, aux_cache_partial, cache_path):
    """Save whatever embeddings we have so far (partial or complete)."""
    keys = sorted(embed_cache.keys())
    embs = np.stack([embed_cache[k] for k in keys], axis=0)
    # aux: prefer freshly computed aux_cache then pre-existing partial cache
    auxs = np.stack([
        aux_cache.get(k, aux_cache_partial.get(k, np.zeros(N_AUX, dtype=np.float32)))
        for k in keys
    ], axis=0)
    np.savez_compressed(cache_path, keys=np.array(keys), embs=embs, auxs=auxs)


def save_embed_cache(embed_cache, aux_cache, path=EMBED_CACHE):
    """Save complete embedding + aux caches to disk (numpy .npz)."""
    keys = sorted(embed_cache.keys())
    embs = np.stack([embed_cache[k] for k in keys], axis=0)   # (N, 768)
    auxs = np.stack([aux_cache[k]   for k in keys], axis=0)   # (N, 14)
    np.savez_compressed(path, keys=np.array(keys), embs=embs, auxs=auxs)
    print(f"  Saved embedding cache -> {path}  ({embs.shape[0]} records, "
          f"{Path(path).stat().st_size / 1e6:.0f} MB)", flush=True)


def load_embed_cache(path=EMBED_CACHE):
    """Load embedding + aux caches from disk."""
    d = np.load(path, allow_pickle=True)
    keys = list(d["keys"])
    embs = d["embs"]   # (N, 768)
    auxs = d["auxs"]   # (N, 14)
    embed_cache = {k: embs[i] for i, k in enumerate(keys)}
    aux_cache   = {k: auxs[i] for i, k in enumerate(keys)}
    print(f"  Loaded embedding cache from {path}  ({len(keys)} records)")
    return embed_cache, aux_cache


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ECGFMClassifier(nn.Module):
    """ECGFMEncoder + MLP head. head_in = EMBED_DIM + N_AUX = 782."""
    def __init__(self, encoder: ECGFMEncoder, n_classes=N_CLASSES):
        super().__init__()
        self.encoder = encoder
        head_in = EMBED_DIM + N_AUX
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward_from_embedding(self, emb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Head-only forward (Stage 1 fast path)."""
        return self.head(torch.cat([emb, aux], dim=1))

    def forward(self, sig_t12: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Full forward (Stage 2 end-to-end)."""
        return self.head(torch.cat([self.encoder(sig_t12), aux], dim=1))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def unfreeze_top_layers(self, n_top: int = 4):
        """Freeze all encoder params, then unfreeze the top n_top transformer layers + layer_norm."""
        self.freeze_encoder()
        # encoder.encoder is the _TransformerEncoder; .layers is the ModuleList of 12 layers
        for layer in self.encoder.encoder.layers[-n_top:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in self.encoder.encoder.layer_norm.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_head(model, loader, device, criterion=None):
    """Evaluate using precomputed embeddings (Stage 1 head)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = total_n = 0
    with torch.no_grad():
        for emb, aux, lbl in loader:
            emb, aux, lbl = emb.to(device), aux.to(device), lbl.to(device)
            logits = model.forward_from_embedding(emb, aux)
            if criterion:
                total_loss += criterion(logits, lbl).item() * emb.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            total_n += emb.size(0)
    return _metrics(all_preds, all_labels, total_loss, total_n, criterion is not None)


def evaluate_full(model, loader, device, criterion=None):
    """Evaluate using full signal forward pass (Stage 2)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = total_n = 0
    with torch.no_grad():
        for sig, aux, lbl in loader:
            sig, aux, lbl = sig.to(device), aux.to(device), lbl.to(device)
            logits = model(sig, aux)
            if criterion:
                total_loss += criterion(logits, lbl).item() * sig.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            total_n += sig.size(0)
    return _metrics(all_preds, all_labels, total_loss, total_n, criterion is not None)


def _metrics(preds, labels, total_loss, total_n, has_loss):
    preds  = np.array(preds)
    labels = np.array(labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    return {
        "loss":      total_loss / (total_n + 1e-9) if has_loss else 0.0,
        "acc":       float((preds == labels).mean()),
        "macro_f1":  float(f1.mean()),
        "hyp_f1":    float(f1[HYP_IDX]),
        "hyp_prec":  float(prec[HYP_IDX]),
        "hyp_rec":   float(rec[HYP_IDX]),
        "f1": f1, "prec": prec, "rec": rec,
    }


def print_epoch(epoch, m, elapsed, marker=""):
    print(f"  {epoch:>5d}  {m['loss']:>7.4f}  {m['acc']:>6.1%}  "
          f"{m['hyp_prec']:>7.1%}  {m['hyp_rec']:>6.1%}  "
          f"{m['hyp_f1']:>6.3f}  {m['macro_f1']:>7.3f}  [{elapsed:.0f}s]{marker}",
          flush=True)


def print_test(tm):
    print(f"\n  Test: Acc={tm['acc']:.1%}  HYPPrec={tm['hyp_prec']:.1%}  "
          f"HYPRec={tm['hyp_rec']:.1%}  HYPF1={tm['hyp_f1']:.3f}  MacroF1={tm['macro_f1']:.3f}")
    print("  Per-class:")
    for i, lbl in enumerate(SUPERCLASS_LABELS):
        print(f"    {lbl:>4}: P={tm['prec'][i]:.1%}  R={tm['rec'][i]:.1%}  F1={tm['f1'][i]:.3f}")


# ---------------------------------------------------------------------------
# Shared data preparation
# ---------------------------------------------------------------------------

def load_splits():
    """Load PTB-XL, return split arrays."""
    paths, labels, folds = load_dataset()
    demographics         = load_dataset_demographics()
    paths  = np.array(paths)
    labels = np.array(labels)
    folds  = np.array(folds)
    return paths, labels, folds, demographics


def resample_train(paths, labels, folds):
    """Stratified resampling of training folds."""
    train_mask   = (folds <= 8)
    train_paths  = paths[train_mask].tolist()
    train_labels = labels[train_mask].tolist()
    class_counts = np.bincount(train_labels, minlength=N_CLASSES)
    norm_n = int(class_counts[0])

    rp, rl = [], []
    for cls in range(N_CLASSES):
        idx_cls   = [i for i, l in enumerate(train_labels) if l == cls]
        paths_cls = [train_paths[i] for i in idx_cls]
        current   = len(idx_cls)
        target    = max(int(norm_n * RESAMPLE_RATIOS.get(cls, 1.0)), current)
        if target > current:
            extras = np.random.choice(current, target - current, replace=True)
            rp.extend([paths_cls[e] for e in extras])
            rl.extend([cls] * (target - current))
        rp.extend(paths_cls)
        rl.extend([cls] * current)
    return rp, rl, class_counts


# ---------------------------------------------------------------------------
# Stage 1: Frozen encoder — precompute embeddings, then train head
# ---------------------------------------------------------------------------

def train_stage1(batch_size=256, n_epochs=40, patience=10):
    print("\n" + "="*60)
    print("  ECG-FM Fine-tuning  STAGE 1  (linear probe on cached embeddings)")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    paths, labels, folds, demographics = load_splits()
    train_mask = (folds <= 8)
    val_mask   = (folds == 9)
    test_mask  = (folds == 10)
    train_paths, train_labels, class_counts = resample_train(paths, labels, folds)

    os_counts = np.bincount(train_labels, minlength=N_CLASSES)
    print("\n  Class distribution:")
    for i, lbl in enumerate(SUPERCLASS_LABELS):
        print(f"    {lbl}: {class_counts[i]} -> {os_counts[i]}")
    print(f"  Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    # ---- Step 1: get (or compute) embedding cache ----
    all_unique = list(set(train_paths + paths[val_mask].tolist() + paths[test_mask].tolist()))

    if Path(EMBED_CACHE).exists():
        print(f"\n  Found cached embeddings at {EMBED_CACHE}", flush=True)
        embed_cache, aux_cache = load_embed_cache(EMBED_CACHE)
        # Check coverage
        missing = [p for p in all_unique if p not in embed_cache]
        if missing:
            print(f"  {len(missing)} paths not in cache — running partial precompute ...")
            raw_cache, new_aux = preload_raw(missing)
            enc = ECGFMEncoder.from_pretrained(CKPT_FM, map_location=str(device)).to(device)
            new_embs = precompute_embeddings(missing, raw_cache, new_aux, enc, device,
                                             batch_size=16, cache_path=EMBED_CACHE)
            embed_cache.update(new_embs)
            aux_cache.update(new_aux)
            save_embed_cache(embed_cache, aux_cache, EMBED_CACHE)
            del enc, raw_cache
    else:
        print(f"\n  No full cache found — running precompute (resumes if partial exists) ...",
              flush=True)
        raw_cache, aux_cache = preload_raw(all_unique)
        enc = ECGFMEncoder.from_pretrained(CKPT_FM, map_location=str(device)).to(device)
        embed_cache = precompute_embeddings(all_unique, raw_cache, aux_cache, enc, device,
                                            batch_size=16, cache_path=EMBED_CACHE)
        save_embed_cache(embed_cache, aux_cache, EMBED_CACHE)
        del enc, raw_cache    # free ~5 GB of signal RAM

    # ---- Step 2: build datasets on precomputed embeddings ----
    train_ds = ECGFMEmbedDataset(train_paths, train_labels, embed_cache, aux_cache, demographics)
    val_ds   = ECGFMEmbedDataset(paths[val_mask].tolist(), labels[val_mask].tolist(),
                                 embed_cache, aux_cache, demographics)
    test_ds  = ECGFMEmbedDataset(paths[test_mask].tolist(), labels[test_mask].tolist(),
                                 embed_cache, aux_cache, demographics)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Step 3: build model (encoder not used in forward pass here) ----
    enc   = ECGFMEncoder.from_pretrained(CKPT_FM, map_location=str(device))
    model = ECGFMClassifier(enc).to(device)
    model.freeze_encoder()

    cw = 1.0 / (os_counts.astype(np.float32) + 1e-6)
    cw = torch.from_numpy(cw / cw.sum() * N_CLASSES).to(device)
    criterion = AsymmetricLoss(weight=cw, gamma_pos=0.0, gamma_neg=4.0,
                               margin=0.05, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs, pct_start=0.1,
    )

    best_hyp_f1 = best_macro = 0.0
    no_improve  = 0
    best_state  = None

    hdr = f"  {'Epoch':>5}  {'Loss':>7}  {'ValAcc':>6}  {'HYPPrec':>7}  {'HYPRec':>6}  {'HYPF1':>6}  {'MacroF1':>7}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, n_epochs + 1):
        model.train()
        model.encoder.eval()
        t0 = time.time()
        for emb, aux, lbl in train_loader:
            emb, aux, lbl = emb.to(device), aux.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model.forward_from_embedding(emb, aux)
            loss   = criterion(logits, lbl)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        m = evaluate_head(model, val_loader, device, criterion)
        improved = m["hyp_f1"] > best_hyp_f1 or (m["hyp_f1"] == best_hyp_f1 and m["macro_f1"] > best_macro)
        if improved:
            best_hyp_f1 = m["hyp_f1"]
            best_macro  = m["macro_f1"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
            marker = " <*>"
        else:
            no_improve += 1
            marker = ""
        print_epoch(epoch, m, time.time() - t0, marker)
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    tm = evaluate_head(model, test_loader, device)
    print_test(tm)

    torch.save({"model_state": best_state, "stage": 1,
                "best_hyp_f1": best_hyp_f1, "best_macro": best_macro,
                "test_metrics": tm}, CKPT_S1)
    print(f"\n  Saved -> {CKPT_S1}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Stage 2: Full fine-tune (differential LR, full forward pass)
# ---------------------------------------------------------------------------

def train_stage2(batch_size=8, n_epochs=20, patience=8):
    print("\n" + "="*60)
    print("  ECG-FM Fine-tuning  STAGE 2  (full fine-tune, differential LR)")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    if not Path(CKPT_S1).exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {CKPT_S1}")

    paths, labels, folds, demographics = load_splits()
    train_mask = (folds <= 8)
    val_mask   = (folds == 9)
    test_mask  = (folds == 10)
    train_paths, train_labels, class_counts = resample_train(paths, labels, folds)

    os_counts  = np.bincount(train_labels, minlength=N_CLASSES)
    test_paths = paths[test_mask].tolist()
    test_labels_list = labels[test_mask].tolist()

    # Load train+val only — keeps peak RAM ~4.5 GB instead of ~5.5 GB (important on free Colab)
    trainval_unique = list(set(train_paths + paths[val_mask].tolist()))
    raw_cache, aux_cache = preload_raw(trainval_unique)

    train_ds = ECGFMDataset(train_paths, train_labels, raw_cache, aux_cache, demographics, augment=True)
    val_ds   = ECGFMDataset(paths[val_mask].tolist(), labels[val_mask].tolist(),
                            raw_cache, aux_cache, demographics, augment=False)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_paths)} (loaded after training)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    enc   = ECGFMEncoder.from_pretrained(CKPT_FM, map_location=str(device))
    model = ECGFMClassifier(enc).to(device)
    s1    = torch.load(CKPT_S1, map_location=device, weights_only=False)
    model.load_state_dict(s1["model_state"])
    # Full unfreeze on GPU (fast enough); partial freeze on CPU (saves time)
    if device.type == "cuda":
        model.unfreeze_encoder()
        unfreeze_desc = "full encoder"
    else:
        model.unfreeze_top_layers(n_top=4)
        unfreeze_desc = "top 4 transformer layers + head"
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Loaded stage-1 (HYP F1={s1['best_hyp_f1']:.3f}  MacroF1={s1['best_macro']:.3f})")
    print(f"  Trainable params: {n_trainable/1e6:.1f}M  ({unfreeze_desc})")

    cw = 1.0 / (os_counts.astype(np.float32) + 1e-6)
    cw = torch.from_numpy(cw / cw.sum() * N_CLASSES).to(device)
    criterion = AsymmetricLoss(weight=cw, gamma_pos=0.0, gamma_neg=4.0,
                               margin=0.05, label_smoothing=0.05)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params,          "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1e-5, 1e-4],
        steps_per_epoch=len(train_loader),
        epochs=n_epochs, pct_start=0.05,
    )

    best_hyp_f1 = s1["best_hyp_f1"]
    best_macro  = s1["best_macro"]
    no_improve  = 0
    best_state  = None

    hdr = f"  {'Epoch':>5}  {'Loss':>7}  {'ValAcc':>6}  {'HYPPrec':>7}  {'HYPRec':>6}  {'HYPF1':>6}  {'MacroF1':>7}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        for sig, aux, lbl in train_loader:
            sig, aux, lbl = sig.to(device), aux.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(sig, aux)
            loss   = criterion(logits, lbl)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        m = evaluate_full(model, val_loader, device, criterion)
        improved = m["hyp_f1"] > best_hyp_f1 or (m["hyp_f1"] == best_hyp_f1 and m["macro_f1"] > best_macro)
        if improved:
            best_hyp_f1 = m["hyp_f1"]
            best_macro  = m["macro_f1"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
            marker = " <*>"
        else:
            no_improve += 1
            marker = ""
        print_epoch(epoch, m, time.time() - t0, marker)
        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Load test signals now (deferred to save RAM during training)
    print("  Loading test signals for final evaluation...")
    test_raw, test_aux = preload_raw(test_paths)
    test_ds     = ECGFMDataset(test_paths, test_labels_list, test_raw, test_aux, demographics, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    tm = evaluate_full(model, test_loader, device)
    print_test(tm)

    torch.save({"model_state": best_state, "stage": 2,
                "best_hyp_f1": best_hyp_f1, "best_macro": best_macro,
                "test_metrics": tm}, CKPT_S2)
    print(f"\n  Saved -> {CKPT_S2}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Standalone embedding precomputation
# ---------------------------------------------------------------------------

def run_precompute():
    """Precompute and cache embeddings for all PTB-XL records."""
    print("\n" + "="*60)
    print("  ECG-FM Embedding Precomputation")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    paths, labels, folds = load_dataset()
    paths = np.array(paths)
    raw_cache, aux_cache = preload_raw(paths.tolist())
    enc = ECGFMEncoder.from_pretrained(CKPT_FM, map_location=str(device)).to(device)
    embed_cache = precompute_embeddings(paths.tolist(), raw_cache, aux_cache, enc, device,
                                        batch_size=16, cache_path=EMBED_CACHE)
    save_embed_cache(embed_cache, aux_cache, EMBED_CACHE)
    print("  Precomputation complete.", flush=True)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def load_ecgfm_classifier(ckpt_path=None, map_location="cpu"):
    """Load fine-tuned ECGFMClassifier. Prefers stage-2, falls back to stage-1."""
    if ckpt_path is None:
        ckpt_path = CKPT_S2 if Path(CKPT_S2).exists() else CKPT_S1
    enc   = ECGFMEncoder.from_pretrained(CKPT_FM, map_location=map_location)
    model = ECGFMClassifier(enc)
    ckpt  = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_ecgfm(model, signal_12, fs=500, sex="M", age=50):
    """
    Run ECG-FM classifier on a single 12-lead ECG.
    Args:
        signal_12 : (N, 12) numpy array in mV
    Returns:
        dict with prediction, confidence, probabilities
    """
    from scipy.signal import resample as sci_resample
    sig = np.array(signal_12, dtype=np.float32)
    if fs != 500:
        sig = sci_resample(sig, int(sig.shape[0] * 500 / fs), axis=0)
    if sig.shape[0] < SIGNAL_LEN:
        sig = np.vstack([sig, np.zeros((SIGNAL_LEN - sig.shape[0], sig.shape[1]))])
    else:
        sig = sig[:SIGNAL_LEN]
    sig = np.clip(sig, -20.0, 20.0)
    aux = extract_voltage_features(sig.T, sex=sex, age=age)

    sig_t = torch.from_numpy(sig).unsqueeze(0)     # (1, 5000, 12)
    aux_t = torch.from_numpy(aux).unsqueeze(0)     # (1, 14)
    with torch.no_grad():
        logits = model(sig_t, aux_t)
        probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "prediction":    SUPERCLASS_LABELS[pred_idx],
        "confidence":    round(float(probs[pred_idx]), 3),
        "probabilities": {SUPERCLASS_LABELS[i]: round(float(p), 3) for i, p in enumerate(probs)},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0,
                        help="1=stage1 only, 2=stage2 only, 0=both")
    parser.add_argument("--precompute", action="store_true",
                        help="Only precompute and cache embeddings, then exit")
    parser.add_argument("--batch_s1", type=int, default=256,
                        help="Batch size for Stage 1 head training (default 256)")
    parser.add_argument("--batch_s2", type=int, default=8,
                        help="Batch size for Stage 2 full fine-tune (default 8)")
    args = parser.parse_args()

    if args.precompute:
        run_precompute()
        sys.exit(0)

    if args.stage in (0, 1):
        train_stage1(batch_size=args.batch_s1)
    if args.stage in (0, 2):
        train_stage2(batch_size=args.batch_s2)
