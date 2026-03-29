"""
ecgfm_multilabel.py
===================
Use ECG-FM Stage 2 encoder as a frozen backbone for 12-class multi-label
classification (same label set as multilabel_classifier.py).

Pipeline:
  1. Load ECGFMEncoder from models/ecgfm_stage2.pt (frozen)
  2. Precompute (B, 768) embeddings for all training signals → cache
  3. Concat with 14-dim aux features → (B, 782)
  4. Train lightweight MLP head with BCEWithLogitsLoss + per-class pos_weight
  5. Evaluate per-class F1, AUROC, macro metrics

Usage:
    python ecgfm_multilabel.py                  # precompute + train
    python ecgfm_multilabel.py --precompute     # only precompute embeddings
    python ecgfm_multilabel.py --train          # train head (needs cache)
    python ecgfm_multilabel.py --eval           # eval saved model
"""

import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score

# ── Reuse existing infrastructure ────────────────────────────────────────────
from ecgfm_encoder import ECGFMEncoder
from cnn_classifier import _load_raw_signal, extract_voltage_features, SIGNAL_LEN, N_AUX
from multilabel_classifier import (
    MULTILABEL_CODES, N_ML_CLASSES,
    load_multilabel_dataset, compute_pos_weights,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PRETRAINED = "models/ecgfm/mimic_iv_ecg_physionet_pretrained.pt"
CKPT_S2         = "models/ecgfm_stage2.pt"
EMBED_CACHE     = "models/ecgfm_ml_embeddings.npz"
CKPT_OUT   = "models/ecgfm_ml_head.pt"

EMBED_DIM  = 768
BATCH_SIZE = 256   # head training — fits easily in RAM


# ── MLP head ──────────────────────────────────────────────────────────────────
class MLHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM + N_AUX, n_classes: int = N_ML_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Precompute embeddings ─────────────────────────────────────────────────────
def precompute(paths, labels, aux, device):
    """Encode all signals with frozen ECG-FM Stage 2. Returns (N, 768) array."""
    # Build encoder architecture from original pretrained checkpoint (has cfg/model keys)
    print(f"\n  Building encoder from {CKPT_PRETRAINED} ...")
    enc = ECGFMEncoder.from_pretrained(CKPT_PRETRAINED, map_location=str(device)).to(device)

    # Load Stage 2 fine-tuned weights (saved as full model state: encoder.* + head.*)
    print(f"  Loading Stage 2 weights from {CKPT_S2} ...")
    s2 = torch.load(CKPT_S2, map_location=str(device), weights_only=False)
    enc_sd = {k[len("encoder."):]: v
              for k, v in s2["model_state"].items()
              if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (expected ~0)")

    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    n = len(paths)
    embeds = np.zeros((n, EMBED_DIM), dtype=np.float32)

    print(f"  Precomputing {n} embeddings (batch 16) ...")
    t0 = time.time()
    bs = 16
    for i in range(0, n, bs):
        batch_paths = paths[i:i+bs]
        sigs = []
        for p in batch_paths:
            raw = _load_raw_signal(p)
            if raw is None:
                sigs.append(np.zeros((12, SIGNAL_LEN), dtype=np.float32))
            else:
                sigs.append(raw)
        x = torch.tensor(np.stack(sigs), dtype=torch.float32)  # (bs, 12, T)
        x = x.permute(0, 2, 1).to(device)                     # (bs, T, 12) — encoder expects leads-last
        with torch.no_grad():
            emb = enc(x)  # (bs, 768)
        embeds[i:i+len(batch_paths)] = emb.cpu().numpy()
        if (i // bs + 1) % 25 == 0 or i + bs >= n:
            elapsed = time.time() - t0
            done = min(i + bs, n)
            eta = (n - done) / (done / elapsed) / 60 if done > 0 else 0
            print(f"    {done}/{n}  {done/elapsed:.0f} rec/s  ETA {eta:.0f} min", flush=True)

    print(f"\n  Done in {time.time()-t0:.0f}s")
    np.savez(EMBED_CACHE, embeds=embeds, labels=labels, aux=aux,
             paths=np.array(paths))
    print(f"  Saved cache -> {EMBED_CACHE}")
    return embeds


# ── Train head ────────────────────────────────────────────────────────────────
def train_head(n_epochs=60, patience=12, lr=1e-3):
    cache = np.load(EMBED_CACHE)
    embeds = cache["embeds"]   # (N, 768)
    labels = cache["labels"]   # (N, 12)
    aux    = cache["aux"]      # (N, 14)

    # Train/val split by fold (reuse same logic — fold 10 = val)
    _, _, folds = load_multilabel_dataset()
    folds      = np.array(folds)
    val_mask   = folds == 10
    train_mask = ~val_mask

    X_train = np.concatenate([embeds[train_mask], aux[train_mask]], axis=1)
    X_val   = np.concatenate([embeds[val_mask],   aux[val_mask]],   axis=1)
    y_train = labels[train_mask].astype(np.float32)
    y_val   = labels[val_mask].astype(np.float32)

    print(f"\n  Train: {train_mask.sum()} | Val: {val_mask.sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    Xt = torch.tensor(X_train).to(device)
    yt = torch.tensor(y_train).to(device)
    Xv = torch.tensor(X_val).to(device)
    yv = torch.tensor(y_val).to(device)

    train_ds = TensorDataset(Xt, yt)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MLHead().to(device)
    pos_weight = compute_pos_weights(y_train).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_macro_f1 = 0.0
    best_epoch    = 0
    no_improve    = 0

    print(f"\n{'Ep':>4}  {'Loss':>8}  {'MacroF1':>8}  {'HYP_F1':>8}")
    print("-" * 40)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        # Val
        model.eval()
        with torch.no_grad():
            logits_v = model(Xv)
        probs_v = torch.sigmoid(logits_v).cpu().numpy()
        preds_v = (probs_v >= 0.5).astype(int)
        y_np    = yv.cpu().numpy().astype(int)

        macro_f1 = f1_score(y_np, preds_v, average="macro", zero_division=0)

        # HYP = LVH index (index 2 in MULTILABEL_CODES)
        lvh_idx  = MULTILABEL_CODES.index("LVH")
        hyp_f1   = f1_score(y_np[:, lvh_idx], preds_v[:, lvh_idx], zero_division=0)

        avg_loss = total_loss / train_mask.sum()
        print(f"{epoch:>4}  {avg_loss:>8.4f}  {macro_f1:>8.3f}  {hyp_f1:>8.3f}", flush=True)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch    = epoch
            no_improve    = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "macro_f1": macro_f1, "hyp_f1": hyp_f1,
                        "label_codes": MULTILABEL_CODES},
                       CKPT_OUT)
            print(f"       ^ saved (macro_f1={macro_f1:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    print(f"\n  Best MacroF1: {best_macro_f1:.3f} at epoch {best_epoch}")
    return best_macro_f1


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate():
    cache = np.load(EMBED_CACHE)
    embeds = cache["embeds"]
    labels = cache["labels"]
    aux    = cache["aux"]

    _, _, folds = load_multilabel_dataset()
    folds     = np.array(folds)
    test_mask = folds == 10

    X_test = np.concatenate([embeds[test_mask], aux[test_mask]], axis=1)
    y_test = labels[test_mask].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MLHead().to(device)
    ckpt   = torch.load(CKPT_OUT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test).to(device))).cpu().numpy()

    preds  = (probs >= 0.5).astype(int)
    y_int  = y_test.astype(int)

    macro_f1   = f1_score(y_int, preds, average="macro",   zero_division=0)
    micro_f1   = f1_score(y_int, preds, average="micro",   zero_division=0)
    macro_auroc = roc_auc_score(y_int, probs, average="macro")

    print(f"\n{'='*56}")
    print(f"  ECG-FM Multilabel — Test Results")
    print(f"{'='*56}")
    print(f"  MacroF1  : {macro_f1:.3f}   (multilabel CNN baseline: 0.699)")
    print(f"  MicroF1  : {micro_f1:.3f}")
    print(f"  MacroAUROC: {macro_auroc:.3f}  (multilabel CNN baseline: 0.972)")
    print(f"\n  Per-class:")
    per_f1   = f1_score(y_int, preds, average=None, zero_division=0)
    per_auroc = roc_auc_score(y_int, probs, average=None)
    for i, code in enumerate(MULTILABEL_CODES):
        print(f"    {code:>6}: F1={per_f1[i]:.3f}  AUROC={per_auroc[i]:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--eval",       action="store_true")
    args = parser.parse_args()

    do_all = not (args.precompute or args.train or args.eval)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.precompute or do_all:
        if os.path.exists(EMBED_CACHE):
            print(f"Cache exists at {EMBED_CACHE} — skipping precompute (use --precompute to force)")
        else:
            paths, labels, folds = load_multilabel_dataset()
            # Build aux cache
            print(f"  Building aux features for {len(paths)} records ...")
            aux = np.zeros((len(paths), N_AUX), dtype=np.float32)
            for i, p in enumerate(paths):
                raw = _load_raw_signal(p)
                if raw is not None:
                    aux[i] = extract_voltage_features(raw)
                if (i + 1) % 500 == 0 or (i + 1) == len(paths):
                    print(f"    aux {i+1}/{len(paths)}", flush=True)
            precompute(paths, labels, aux, device)

    if args.train or do_all:
        print("\n" + "="*56)
        print("  Training MLP head on ECG-FM embeddings")
        print("="*56)
        train_head()

    if args.eval or do_all:
        evaluate()
