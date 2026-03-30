"""
cnn_classifier.py
=================
1D CNN classifier for 12-lead ECG superclass prediction.
Operates on raw signal (5000 x 12) -- no hand-crafted features needed.

Superclasses: NORM, MI, STTC, HYP, CD

v2 improvements:
  - Data augmentation (noise, scaling, time shift, lead dropout)
  - Oversampling of minority classes (HYP, STTC)
  - Squeeze-and-Excitation attention
  - 50 epochs with OneCycleLR
  - Label smoothing

Usage:
    # Train
    python cnn_classifier.py

    # From app.py:
    from cnn_classifier import load_cnn_classifier, predict_cnn
    model = load_cnn_classifier()
    result = predict_cnn(model, signal_12, fs=500)
"""

import ast
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------

SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]
SUPERCLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T Change",
    "HYP":  "Hypertrophy",
    "CD":   "Conduction Disturbance",
}
LABEL_TO_IDX = {label: i for i, label in enumerate(SUPERCLASS_LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

MODEL_PATH = "models/ecg_cnn.pt"
SIGNAL_LEN = 5000  # 10s at 500Hz
N_LEADS = 12

# v10: amplitude-preserving normalization constant.
# Dividing by 5 mV keeps the CNN input in [-4, +4] range while preserving
# absolute voltage relationships (Sokolow threshold 3.5 mV -> 0.70 after norm).
GLOBAL_NORM_SCALE = 5.0
N_CLASSES = 5

# Auxiliary features: 8 voltage + 2 demographic + 1 axis + 3 morphology (v10)
N_AUX = 14

# Standard 12-lead index mapping (PTB-XL WFDB order)
_LEAD_IDX = {
    "I": 0, "II": 1, "III": 2, "AVR": 3, "AVL": 4, "AVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11,
}


# -------------------------------------------------------------
# Voltage Feature Extraction
# (computed from raw mV signal before per-lead normalization)
# -------------------------------------------------------------

def _detect_r_peaks_simple(lead_sig, fs):
    """Return sample indices of R-peaks using threshold crossing. Lead signal in mV."""
    std_val = np.std(lead_sig)
    if std_val < 1e-6:
        return []
    threshold = 2.0 * std_val
    above = np.where(lead_sig > threshold)[0]
    if len(above) == 0:
        return []
    crossings = [above[0]]
    for i in range(1, len(above)):
        if above[i] - above[i - 1] > fs // 2:
            crossings.append(above[i])
    # Refine to local maximum within +/-100ms window
    win = fs // 10
    refined = []
    for c in crossings:
        start = max(0, c - win)
        end   = min(len(lead_sig), c + win)
        refined.append(int(start + np.argmax(lead_sig[start:end])))
    return refined


def _t_wave_strain_score(sig_12xN, fs):
    """
    LVH strain pattern: T-wave inversion in lateral leads (V5, V6, aVL).
    Uses lead II for R-peak detection. Returns score in [0, 1].
    1 = strongly inverted T waves in lateral leads (high HYP specificity).
    """
    lead_ii = sig_12xN[_LEAD_IDX["II"]]
    r_peaks = _detect_r_peaks_simple(lead_ii, fs)
    if len(r_peaks) < 1:
        return 0.0

    t_start = int(0.15 * fs)   # 150 ms post-R
    t_end   = int(0.38 * fs)   # 380 ms post-R (avoids next P wave)

    lateral = [_LEAD_IDX["V5"], _LEAD_IDX["V6"], _LEAD_IDX["AVL"]]
    lead_scores = []
    for li in lateral:
        lead = sig_12xN[li] - np.mean(sig_12xN[li])   # remove DC
        lead_range = max(float(np.max(lead) - np.min(lead)), 0.1)
        beat_t = []
        for pk in r_peaks:
            s, e = pk + t_start, pk + t_end
            if e < len(lead):
                beat_t.append(float(np.mean(lead[s:e])))
        if beat_t:
            median_t = float(np.median(beat_t))
            lead_scores.append(max(0.0, -median_t / lead_range))

    return float(np.clip(max(lead_scores) if lead_scores else 0.0, 0.0, 1.0))


def _qrs_duration_norm(sig_12xN, fs):
    """
    Estimate QRS duration in lead II, normalised by 200 ms (return in [0, 1]).
    Typical normal: 80-100 ms (0.40-0.50). LVH/BBB: 100-120+ ms (0.50-0.60+).
    """
    lead = sig_12xN[_LEAD_IDX["II"]].copy()
    lead -= np.mean(lead)
    r_peaks = _detect_r_peaks_simple(lead, fs)
    if len(r_peaks) < 1:
        return 0.5   # default ~100 ms

    win = int(0.08 * fs)   # +/-80 ms around each R peak
    durations = []
    for pk in r_peaks:
        s = max(0, pk - win)
        e = min(len(lead), pk + win)
        seg = lead[s:e]
        if len(seg) < 4:
            continue
        peak_amp = float(np.max(np.abs(seg)))
        if peak_amp < 1e-6:
            continue
        above = np.where(np.abs(seg) > 0.15 * peak_amp)[0]
        if len(above) >= 2:
            durations.append((above[-1] - above[0]) / fs)

    if not durations:
        return 0.5
    return float(np.clip(float(np.median(durations)) / 0.20, 0.0, 2.0))


def extract_voltage_features(sig_12xN, sex="M", age=50):
    """
    Extract 14-dim voltage + demographic feature vector from raw (12, N) signal in mV.

    Indices 0-10: same as v9 (backward compat -- old n_aux=11 models slice [:11])
      0: S(V1)              /3.0   S-wave depth in V1 (LVH marker)
      1: max(R_V5, R_V6)   /3.0   Tall R in lateral leads (LVH marker)
      2: Sokolow-Lyon value /5.0   S(V1)+max(R_V5,R_V6); threshold 3.5 mV
      3: Sokolow met                1.0 if > 3.5 mV, else 0.0
      4: R(aVL)            /2.0   R-wave in aVL (Cornell LVH)
      5: S(V3)             /2.0   S-wave in V3 (Cornell LVH)
      6: Cornell value     /4.0   R(aVL)+S(V3); threshold 2.8/2.0 mV
      7: R(V1)             /2.0   Dominant R in V1 (RVH marker)
      8: sex_female                0.0 (M) / 1.0 (F)  [overwritten per-sample]
      9: age_norm                  age/80 [0,1]        [overwritten per-sample]
     10: frontal QRS axis  /180   deg; LAD (<-30) common in LVH

    New in v10 (indices 11-13):
     11: T-wave strain score        LVH strain in V5/V6/aVL [0,1]; 1=inverted T
     12: QRS duration norm          QRS_ms/200; normal~0.45, LVH/BBB~0.55+
     13: Cornell VDP norm           (Cornell x QRS_ms)/2440; >1 = criterion met
    """
    def _rs(lead_idx):
        lead = sig_12xN[lead_idx]
        return float(np.max(lead)), float(abs(np.min(lead)))

    r_i,   s_i   = _rs(_LEAD_IDX["I"])
    r_avf, s_avf = _rs(_LEAD_IDX["AVF"])
    _, s_v1       = _rs(_LEAD_IDX["V1"])
    r_v5, _       = _rs(_LEAD_IDX["V5"])
    r_v6, _       = _rs(_LEAD_IDX["V6"])
    r_avl, _      = _rs(_LEAD_IDX["AVL"])
    _, s_v3       = _rs(_LEAD_IDX["V3"])
    r_v1, _       = _rs(_LEAD_IDX["V1"])

    sokolow = s_v1 + max(r_v5, r_v6)
    cornell = r_avl + s_v3

    # Frontal QRS axis
    net_i    = r_i   - s_i
    net_avf  = r_avf - s_avf
    axis_deg  = float(np.degrees(np.arctan2(net_avf, net_i)))
    axis_norm = float(np.clip(axis_deg / 180.0, -1.0, 1.0))

    # New morphology features (v10)
    n_samples = sig_12xN.shape[1]
    fs_est    = n_samples // 10   # assume 10-second recording -> 500 Hz
    t_strain  = _t_wave_strain_score(sig_12xN, fs_est)
    qrs_norm  = _qrs_duration_norm(sig_12xN, fs_est)
    qrs_ms    = qrs_norm * 200.0
    cvdp_norm = float(np.clip(cornell * qrs_ms / 2440.0, 0.0, 2.0))  # Cornell VDP

    feats = np.array([
        np.clip(s_v1 / 3.0, 0, 2.0),
        np.clip(max(r_v5, r_v6) / 3.0, 0, 2.0),
        np.clip(sokolow / 5.0, 0, 2.0),
        float(sokolow > 3.5),
        np.clip(r_avl / 2.0, 0, 2.0),
        np.clip(s_v3 / 2.0, 0, 2.0),
        np.clip(cornell / 4.0, 0, 2.0),
        np.clip(r_v1 / 2.0, 0, 2.0),
        float(sex == "F"),
        float(np.clip(float(age) / 80.0, 0, 1)),
        axis_norm,          # index 10
        t_strain,           # index 11 -- new v10
        qrs_norm,           # index 12 -- new v10
        cvdp_norm,          # index 13 -- new v10
    ], dtype=np.float32)

    return feats


# -------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss -- down-weights easy examples, focuses on hard misclassifications.
    Especially effective for rare classes like HYP where the model is often wrong.
    
    Phase 1 Improvement: Per-class gamma tuning.
    - Critical classes (MI, HYP): gamma=2.5 (focus harder on hard examples)
    - Moderate classes: gamma=2.0 (standard focal loss)
    - Easy classes (NORM): gamma=1.5 (less aggressive focusing)
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05, gamma_per_class=None):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        
        # Support both scalar gamma and per-class gamma dict
        if gamma_per_class is not None:
            # gamma_per_class should be dict: {class_idx: gamma_value}
            # Default: MI (1) and HYP (3) get 2.5, others get 2.0
            self.gamma_per_class = gamma_per_class
        else:
            # Phase 1: Tuned gamma values per class
            self.gamma_per_class = {
                0: 1.5,   # NORM: easy class, lower gamma
                1: 2.5,   # MI: critical (53% recall), higher gamma
                2: 2.0,   # STTC: moderate
                3: 2.5,   # HYP: critical (23% precision), higher gamma
                4: 2.0,   # CD: moderate
            }

    def forward(self, logits, targets):
        # Apply label smoothing manually
        n_classes = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal modulation: (1 - p_t)^gamma
        # Per-class gamma: apply different focusing strength per target class
        batch_size = logits.shape[0]
        focal_weights = torch.ones_like(probs)
        
        for class_idx in range(n_classes):
            gamma_class = self.gamma_per_class.get(class_idx, 2.0)
            class_mask = (targets == class_idx)
            if class_mask.sum() > 0:
                focal_weights[class_mask] = (1.0 - probs[class_mask]) ** gamma_class

        # Weighted focal cross-entropy
        loss = -focal_weights * smooth_targets * log_probs

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        return loss.sum(dim=1).mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al. 2021) adapted for single-label multi-class.

    Key idea: apply a *stricter* focusing penalty to non-target class predictions
    (gamma- > gamma+). This directly penalises samples where p(HYP) is high but the true
    label is NORM/MI/STTC/CD -- the dominant failure mode for HYP false positives.

    margin (m): clips easy negatives -- any p < m is treated as 0. This prevents
    the model from wasting gradient on trivially correct negatives.

    Recommended defaults: gamma+=0, gamma-=4, m=0.05 (from original paper).
    """
    def __init__(self, weight=None, gamma_pos=0.0, gamma_neg=4.0,
                 margin=0.05, label_smoothing=0.05):
        super().__init__()
        self.weight        = weight
        self.gamma_pos     = gamma_pos
        self.gamma_neg     = gamma_neg
        self.margin        = margin
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)          # (B, C)

        # Smooth one-hot targets
        with torch.no_grad():
            smooth_t = torch.zeros_like(logits)
            smooth_t.fill_(self.label_smoothing / (n_classes - 1))
            smooth_t.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Positive branch (target class): standard focal-style
        xs_pos   = probs
        log_pos  = torch.log(xs_pos.clamp(min=1e-8))
        focal_p  = (1.0 - xs_pos) ** self.gamma_pos

        # Negative branch (non-target classes): clip + harsher gamma
        xs_neg   = (probs - self.margin).clamp(min=0.0)
        log_neg  = torch.log((1.0 - xs_neg).clamp(min=1e-8))
        focal_n  = xs_neg ** self.gamma_neg

        loss = -(smooth_t * focal_p * log_pos +
                 (1.0 - smooth_t) * focal_n * log_neg)

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        return loss.sum(dim=1).mean()


class SqueezeExcitation(nn.Module):
    """Channel attention: learns which channels (features) matter most."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class ECGResBlock(nn.Module):
    """Residual block for 1D convolution with optional SE attention."""
    def __init__(self, channels, kernel_size=7, dropout=0.1, use_se=False):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return self.relu(x + out)


class ECGNet(nn.Module):
    """
    1D ResNet-style CNN with SE attention for 12-lead ECG classification.
    Input: (batch, 12, 5000)
    Output: (batch, 5) logits
    """
    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()

        # Stem: 12 leads -> 64 channels
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),  # 5000 -> 1250
        )

        # Block 1: 64 channels
        self.layer1 = nn.Sequential(
            ECGResBlock(64, kernel_size=7, dropout=0.1, use_se=True),
            ECGResBlock(64, kernel_size=7, dropout=0.1),
            nn.MaxPool1d(4),  # 1250 -> 312
        )

        # Block 2: 128 channels
        self.expand2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ECGResBlock(128, kernel_size=7, dropout=0.2, use_se=True),
            ECGResBlock(128, kernel_size=5, dropout=0.2),
            nn.MaxPool1d(4),  # 312 -> 78
        )

        # Block 3: 256 channels
        self.expand3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ECGResBlock(256, kernel_size=5, dropout=0.3, use_se=True),
            ECGResBlock(256, kernel_size=3, dropout=0.3),
            nn.AdaptiveAvgPool1d(1),  # -> 1
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.expand2(x)
        x = self.layer2(x)
        x = self.expand3(x)
        x = self.layer3(x)
        return self.head(x)


class ECGNetJoint(nn.Module):
    """
    Joint CNN + voltage/demographic auxiliary branch.
    Same backbone as ECGNet; voltage features fused at classifier head.

    Input:  signal (B, 12, 5000) + aux (B, N_AUX)
    Output: (B, 5) logits
    """
    def __init__(self, n_leads=12, n_classes=5, n_aux=N_AUX):
        super().__init__()

        # CNN backbone (identical to ECGNet)
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.layer1 = nn.Sequential(
            ECGResBlock(64, kernel_size=7, dropout=0.1, use_se=True),
            ECGResBlock(64, kernel_size=7, dropout=0.1),
            nn.MaxPool1d(4),
        )
        self.expand2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ECGResBlock(128, kernel_size=7, dropout=0.2, use_se=True),
            ECGResBlock(128, kernel_size=5, dropout=0.2),
            nn.MaxPool1d(4),
        )
        self.expand3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ECGResBlock(256, kernel_size=5, dropout=0.3, use_se=True),
            ECGResBlock(256, kernel_size=3, dropout=0.3),
            nn.AdaptiveAvgPool1d(1),
        )

        # Auxiliary branch: voltage + demographics -> 32-dim embedding
        self.aux_branch = nn.Sequential(
            nn.Linear(n_aux, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Fusion: 256 (CNN) + 32 (aux) -> classifier
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, aux):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.expand2(x)
        x = self.layer2(x)
        x = self.expand3(x)
        x = self.layer3(x)
        x = x.squeeze(-1)          # (B, 256)
        a = self.aux_branch(aux)   # (B, 32)
        return self.fusion(torch.cat([x, a], dim=1))


class ECGNetTransformer(nn.Module):
    """
    CNN-Transformer hybrid for 12-lead ECG classification.

    Architecture:
      CNN frontend: (B,12,5000) -> stem+layer1+2 -> (B,256,78)
      Transformer:  attends across 78 time positions (each = 64ms of ECG)
      Fusion:       Transformer mean-pool (256) + voltage aux (32) -> classifier

    The Transformer backend captures global dependencies the CNN misses:
    P-wave to T-wave relationships, beat-to-beat amplitude variation,
    and distributed voltage patterns critical for borderline HYP cases.

    Input:  signal (B, 12, 5000) + aux (B, N_AUX)
    Output: (B, n_classes) logits
    """
    # Sequence length after CNN downsampling: 5000/4/4/4 = 78
    SEQ_LEN = 78

    def __init__(self, n_leads=12, n_classes=5, n_aux=N_AUX,
                 n_heads=4, n_tf_layers=2, dim_ff=512, tf_dropout=0.1):
        super().__init__()

        # CNN frontend -- identical to ECGNetJoint up through expand3
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),            # 5000 -> 1250
        )
        self.layer1 = nn.Sequential(
            ECGResBlock(64, kernel_size=7, dropout=0.1, use_se=True),
            ECGResBlock(64, kernel_size=7, dropout=0.1),
            nn.MaxPool1d(4),            # 1250 -> 312
        )
        self.expand2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ECGResBlock(128, kernel_size=7, dropout=0.2, use_se=True),
            ECGResBlock(128, kernel_size=5, dropout=0.2),
            nn.MaxPool1d(4),            # 312 -> 78
        )
        self.expand3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        # Final conv blocks without pooling -- keep 78-step sequence
        self.layer3_conv = nn.Sequential(
            ECGResBlock(256, kernel_size=5, dropout=0.3, use_se=True),
            ECGResBlock(256, kernel_size=3, dropout=0.3),
        )

        # Learned positional embeddings (78 positions x 256 dims)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.SEQ_LEN, 256))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder: each of 78 positions attends to all others
        tf_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=tf_dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(tf_layer, num_layers=n_tf_layers)
        self.tf_norm = nn.LayerNorm(256)

        # Auxiliary branch: voltage + demographics -> 32-dim
        self.aux_branch = nn.Sequential(
            nn.Linear(n_aux, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Fusion: 256 (Transformer) + 32 (aux) -> classifier
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, aux):
        # CNN frontend
        x = self.stem(x)
        x = self.layer1(x)
        x = self.expand2(x)
        x = self.layer2(x)
        x = self.expand3(x)
        x = self.layer3_conv(x)             # (B, 256, 78)

        # Transformer backend
        x = x.transpose(1, 2)              # (B, 78, 256)
        x = x + self.pos_embed             # add positional encoding
        x = self.transformer(x)            # (B, 78, 256)  -- global attention
        x = self.tf_norm(x)
        x = x.mean(dim=1)                  # (B, 256)  -- mean pool

        # Aux branch + fusion
        a = self.aux_branch(aux)           # (B, 32)
        return self.fusion(torch.cat([x, a], dim=1))


# -------------------------------------------------------------
# Data Augmentation
# -------------------------------------------------------------

def augment_signal(sig):
    """
    Apply random augmentations to a (12, 5000) signal.
    Each augmentation is applied independently with some probability.
    """
    # Gaussian noise
    if np.random.random() < 0.5:
        noise_level = np.random.uniform(0.01, 0.15)
        sig = sig + np.random.randn(*sig.shape).astype(np.float32) * noise_level

    # Amplitude scaling per-lead
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.8, 1.2, size=(sig.shape[0], 1)).astype(np.float32)
        sig = sig * scale

    # Time shift (circular)
    if np.random.random() < 0.3:
        shift = np.random.randint(-250, 250)
        sig = np.roll(sig, shift, axis=1)

    # Lead dropout (zero 1-2 random leads)
    if np.random.random() < 0.2:
        n_drop = np.random.randint(1, 3)
        drop_leads = np.random.choice(sig.shape[0], n_drop, replace=False)
        sig[drop_leads] = 0.0

    # Baseline wander (low-frequency sinusoidal noise)
    if np.random.random() < 0.3:
        t = np.linspace(0, 1, sig.shape[1], dtype=np.float32)
        freq = np.random.uniform(0.1, 0.5)
        amplitude = np.random.uniform(0.05, 0.2)
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        sig = sig + wander[np.newaxis, :]

    return sig


# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------

def _load_raw_signal(rec_path):
    """
    Load a WFDB record from disk and return raw physical signal (mV) as (12, 5000) float32.
    No per-lead normalization applied -- values are in millivolts.
    """
    try:
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal  # (N, channels)
        fs  = rec.fs
    except Exception:
        return np.zeros((N_LEADS, SIGNAL_LEN), dtype=np.float32)

    if sig is None:
        return np.zeros((N_LEADS, SIGNAL_LEN), dtype=np.float32)

    # Handle non-12-lead signals
    n_ch = sig.shape[1]
    if n_ch < N_LEADS:
        sig = np.hstack([sig, np.zeros((sig.shape[0], N_LEADS - n_ch))])
    elif n_ch > N_LEADS:
        sig = sig[:, :N_LEADS]

    # Resample to 500 Hz if needed
    if fs != 500 and fs > 0:
        from scipy.signal import resample
        target_n = int(sig.shape[0] * 500 / fs)
        sig = resample(sig, target_n, axis=0)

    # Pad or truncate to SIGNAL_LEN
    if sig.shape[0] < SIGNAL_LEN:
        pad = np.zeros((SIGNAL_LEN - sig.shape[0], N_LEADS))
        sig = np.vstack([sig, pad])
    else:
        sig = sig[:SIGNAL_LEN]

    sig = sig.T.astype(np.float32)  # (12, 5000)
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig = np.clip(sig, -20.0, 20.0)
    return sig


def _normalize_signal_zscore(sig):
    """v9 and earlier: per-lead z-score. Preserves shape but destroys absolute amplitude."""
    mean = sig.mean(axis=1, keepdims=True)
    std  = sig.std(axis=1, keepdims=True) + 1e-8
    out  = (sig - mean) / std
    if not np.isfinite(out).all():
        out = np.zeros_like(sig)
    return out


def _normalize_signal(sig):
    """v10+: per-lead baseline removal + fixed global scale (amplitude-preserving).

    Key change: divide by GLOBAL_NORM_SCALE (5 mV) instead of per-lead std.
    This preserves absolute voltage relationships across leads and patients --
    critical for HYP where Sokolow-Lyon (>3.5 mV) and Cornell (>2.8 mV) are
    defined by absolute millivolt thresholds.

    After this normalization:
      Sokolow threshold 3.5 mV  ->  0.70
      Cornell threshold 2.8 mV  ->  0.56
      Typical QRS peak  1.0 mV  ->  0.20
    """
    mean = sig.mean(axis=1, keepdims=True)   # per-lead DC offset removal only
    out  = (sig - mean) / GLOBAL_NORM_SCALE
    if not np.isfinite(out).all():
        out = np.zeros_like(sig)
    return out


def _load_and_preprocess_signal(rec_path):
    """Load a single record from disk and preprocess to (12, 5000) float32 (normalized)."""
    return _normalize_signal(_load_raw_signal(rec_path))


def preload_all(all_paths):
    """
    Pre-load all signals into memory in a single disk-read pass.
    Also extracts voltage features (8-dim, signal-based) from raw mV signal.

    Returns:
        signal_cache: dict path -> (12, 5000) normalized float32
        voltage_cache: dict path -> (8,) float32 voltage features
    """
    signal_cache  = {}
    voltage_cache = {}
    unique_paths  = list(set(all_paths))
    print(f"\n  Pre-loading {len(unique_paths)} unique signals + voltage features...")
    t0 = time.time()
    for i, path in enumerate(unique_paths):
        if (i + 1) % 2000 == 0 or (i + 1) == len(unique_paths):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (len(unique_paths) - i - 1) / rate
            print(f"    {i+1}/{len(unique_paths)} ({rate:.0f}/s, ETA {eta:.0f}s)")
        raw = _load_raw_signal(path)
        voltage_cache[path] = extract_voltage_features(raw)  # full 11-dim (sex/age overwritten per-sample)
        signal_cache[path]  = _normalize_signal(raw)
    elapsed = time.time() - t0
    print(f"  Pre-loaded {len(signal_cache)} signals in {elapsed:.0f}s")
    return signal_cache, voltage_cache


def preload_signals(all_paths):
    """Backward-compatible wrapper -- returns signal_cache only."""
    signal_cache, _ = preload_all(all_paths)
    return signal_cache


class ECGDataset(Dataset):
    """
    Dataset backed by pre-loaded signal cache for fast training.
    If voltage_cache and demographics are provided, also returns aux features.
    """
    def __init__(self, record_paths, labels, signal_cache, augment=False,
                 voltage_cache=None, demographics=None):
        self.record_paths = record_paths
        self.labels       = labels
        self.signal_cache = signal_cache
        self.augment      = augment
        self.voltage_cache = voltage_cache        # path -> (8,) float32
        self.demographics  = demographics or {}   # path -> (sex_female, age_norm)

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        path = self.record_paths[idx]
        sig  = self.signal_cache[path].copy()

        if self.augment:
            sig = augment_signal(sig)

        label = self.labels[idx]
        sig_t = torch.from_numpy(sig)
        lbl_t = torch.tensor(label, dtype=torch.long)

        if self.voltage_cache is not None:
            aux = self.voltage_cache.get(path, np.zeros(N_AUX, dtype=np.float32)).copy()
            sex_f, age_n = self.demographics.get(path, (0.0, 0.625))  # 0.625 = 50/80
            aux[8] = sex_f   # overwrite default sex with actual
            aux[9] = age_n   # overwrite default age with actual
            return sig_t, torch.from_numpy(aux), lbl_t

        return sig_t, lbl_t


# -------------------------------------------------------------
# Data loading (reuses PTB-XL superclass mapping)
# -------------------------------------------------------------

def build_scp_to_superclass(scp_path):
    df = pd.read_csv(scp_path, index_col=0)
    diag = df[df["diagnostic"] == 1.0]
    return diag["diagnostic_class"].to_dict()


def get_primary_superclass(scp_codes_str, scp_map):
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return None
    best_class = None
    best_score = -1
    for code, likelihood in codes.items():
        superclass = scp_map.get(code)
        if superclass and likelihood > best_score:
            best_score = likelihood
            best_class = superclass
    return best_class


def load_dataset(base_path="ekg_datasets/ptbxl"):
    """Index all records and return paths + labels (no signal loading)."""
    base = Path(base_path)
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    scp_map = build_scp_to_superclass(str(base / "scp_statements.csv"))

    meta["superclass"] = meta["scp_codes"].apply(
        lambda x: get_primary_superclass(x, scp_map)
    )
    meta = meta.dropna(subset=["superclass"])
    meta = meta[meta["superclass"].isin(SUPERCLASS_LABELS)]

    paths, labels, folds = [], [], []

    print(f"  Indexing {len(meta)} records...")
    for ecg_id, row in meta.iterrows():
        rec_path = str(base / row["filename_hr"])
        if os.path.exists(rec_path + ".dat"):
            paths.append(rec_path)
            labels.append(LABEL_TO_IDX[row["superclass"]])
            folds.append(int(row["strat_fold"]))

    print(f"  Found {len(paths)} valid records on disk")
    return paths, labels, folds


def load_unified_dataset():
    """
    Load the unified multi-dataset index (if available).
    Falls back to PTB-XL only if unified index doesn't exist.

    Returns: paths, labels, folds, dataset_names
    PTB-XL records keep their original folds (1-10).
    External records get fold=0 (used for training only, never test).
    """
    index_path = Path("ekg_datasets/unified_index.csv")

    if not index_path.exists():
        print("  No unified index found. Building it now...")
        try:
            from dataset_pipeline import build_unified_index
            build_unified_index()
        except Exception as e:
            print(f"  Could not build unified index: {e}")
            print("  Falling back to PTB-XL only")
            p, l, f = load_dataset()
            return p, l, f, ["PTB-XL"] * len(p)

    if not index_path.exists():
        print("  Falling back to PTB-XL only")
        p, l, f = load_dataset()
        return p, l, f, ["PTB-XL"] * len(p)

    df = pd.read_csv(index_path)
    print(f"  Unified index: {len(df)} total records")

    paths, labels, folds, datasets = [], [], [], []
    skipped = 0

    for _, row in df.iterrows():
        rec_path = row["path"]
        superclass = row["superclass"]

        if superclass not in LABEL_TO_IDX:
            skipped += 1
            continue

        # Verify file exists (check .dat or .mat)
        if not (os.path.exists(rec_path + ".dat") or os.path.exists(rec_path + ".mat")):
            skipped += 1
            continue

        paths.append(rec_path)
        labels.append(LABEL_TO_IDX[superclass])
        datasets.append(row["dataset"])

        # PTB-XL has stratified folds; external datasets get fold=0
        if "strat_fold" in row and pd.notna(row.get("strat_fold")):
            folds.append(int(row["strat_fold"]))
        else:
            folds.append(0)  # External = training only

    print(f"  Loaded {len(paths)} valid records ({skipped} skipped)")
    for ds_name, count in pd.Series(datasets).value_counts().items():
        print(f"    {ds_name}: {count}")

    return paths, labels, folds, datasets


def load_dataset_demographics(base_path="ekg_datasets/ptbxl"):
    """
    Load age/sex from PTB-XL metadata.
    Returns dict: rec_path -> (sex_female: float, age_norm: float)
    PTB-XL sex encoding: 0=male, 1=female
    """
    base = Path(base_path)
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    demo = {}
    for ecg_id, row in meta.iterrows():
        rec_path  = str(base / row["filename_hr"])
        sex_raw   = row.get("sex", 0)
        age_raw   = row.get("age", 50)
        if pd.isna(sex_raw): sex_raw = 0
        if pd.isna(age_raw): age_raw = 50
        sex_female = float(int(sex_raw) == 1)
        age_norm   = float(np.clip(float(age_raw) / 80.0, 0, 1))
        demo[rec_path] = (sex_female, age_norm)
    return demo


# -------------------------------------------------------------
# Training
# -------------------------------------------------------------

def identify_hard_examples(model, data_loader, device, top_pct=15):
    """
    Phase 1 Improvement: Hard negative mining.
    Find most challenging examples per epoch to focus training.
    
    Args:
      model: trained model
      data_loader: DataLoader to evaluate
      device: cuda/cpu
      top_pct: percentage of samples to identify (15% = top 15% hardest)
    
    Returns:
      hard_indices: list of indices in data_loader with highest losses
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch_idx, (signals, labels) in enumerate(data_loader):
            signals = signals.to(device)
            labels = labels.to(device)
            
            logits = model(signals)
            
            # Per-sample loss (not averaged)
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -log_probs[torch.arange(len(labels)), labels]
            
            for sample_idx, loss_val in enumerate(ce_loss):
                global_idx = batch_idx * len(labels) + sample_idx
                losses.append((global_idx, loss_val.item()))
    
    # Sort by loss (descending) and take top percentile
    losses.sort(key=lambda x: x[1], reverse=True)
    n_hard = max(1, int(len(losses) * (top_pct / 100.0)))
    hard_indices = [idx for idx, _ in losses[:n_hard]]
    
    return hard_indices


def train(use_multi=False):
    version = "v10-ampnorm (multi)" if use_multi else "v10-ampnorm (PTB-XL)"
    print("\n" + "=" * 60)
    print(f"  ECG Joint CNN {version} -- Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if use_multi:
        paths, labels, folds, datasets = load_unified_dataset()
    else:
        paths, labels, folds = load_dataset()
        datasets = ["PTB-XL"] * len(paths)

    if not paths:
        print("  No records found.")
        return

    paths  = np.array(paths)
    labels = np.array(labels)
    folds  = np.array(folds)

    train_mask = (folds <= 8)
    val_mask   = (folds == 9)
    test_mask  = (folds == 10)

    # Stratified rebalancing
    train_paths  = paths[train_mask].tolist()
    train_labels = labels[train_mask].tolist()
    class_counts = np.bincount(train_labels, minlength=N_CLASSES)

    RESAMPLE_RATIOS = {
        0: 0.5,   # NORM: downsample
        1: 1.5,   # MI: upsample
        2: 0.8,   # STTC
        3: 1.2,   # HYP: upsample
        4: 0.9,   # CD
    }

    norm_count    = int(class_counts[0])
    target_counts = {
        cls: max(int(norm_count * RESAMPLE_RATIOS.get(cls, 1.0)), int(class_counts[cls]))
        for cls in range(N_CLASSES)
    }

    resampled_paths, resampled_labels = [], []
    for cls_idx in range(N_CLASSES):
        cls_indices   = [i for i, l in enumerate(train_labels) if l == cls_idx]
        cls_paths_lst = [train_paths[i] for i in cls_indices]
        current       = len(cls_indices)
        target        = target_counts[cls_idx]
        if target >= current:
            n_extra = target - current
            extras  = np.random.choice(len(cls_indices), n_extra, replace=True)
            resampled_paths.extend([cls_paths_lst[e] for e in extras])
            resampled_labels.extend([cls_idx] * n_extra)
            resampled_paths.extend(cls_paths_lst)
            resampled_labels.extend([cls_idx] * current)
        else:
            sel = np.random.choice(len(cls_indices), target, replace=False)
            resampled_paths.extend([cls_paths_lst[s] for s in sel])
            resampled_labels.extend([cls_idx] * target)

    oversampled_paths  = resampled_paths
    oversampled_labels = resampled_labels

    print(f"\n  Original / resampled distribution:")
    os_counts = np.bincount(oversampled_labels, minlength=N_CLASSES)
    for i, lbl in enumerate(SUPERCLASS_LABELS):
        print(f"    {lbl}: {int(class_counts[i])} -> {int(os_counts[i])}")
    print(f"  Total: {len(train_paths)} -> {len(oversampled_paths)}")

    # Load PTB-XL demographics (age / sex) for Cornell threshold
    print("  Loading demographics...")
    demographics = load_dataset_demographics()
    print(f"  Loaded demographics for {len(demographics)} records")

    # Single-pass preload: signal cache + voltage features
    all_unique_paths = list(set(
        oversampled_paths +
        paths[val_mask].tolist() +
        paths[test_mask].tolist()
    ))
    signal_cache, voltage_cache = preload_all(all_unique_paths)

    train_ds = ECGDataset(oversampled_paths, oversampled_labels, signal_cache, augment=True,
                          voltage_cache=voltage_cache, demographics=demographics)
    val_ds   = ECGDataset(paths[val_mask].tolist(), labels[val_mask].tolist(), signal_cache, augment=False,
                          voltage_cache=voltage_cache, demographics=demographics)
    test_ds  = ECGDataset(paths[test_mask].tolist(), labels[test_mask].tolist(), signal_cache, augment=False,
                          voltage_cache=voltage_cache, demographics=demographics)

    print(f"  Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # v10: ECGNetJoint -- best performing arch from v8; isolate normalization change
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: ECGNetJoint  ({param_count:,} params)")

    # Inverse-frequency weights on resampled distribution.
    # Resampling already handles class imbalance; these weights provide mild
    # additional correction. HYP and STTC get modest upweighting since they
    # remain smaller after resampling relative to MI.
    os_counts_f   = os_counts.astype(np.float32)
    class_weights = 1.0 / (os_counts_f + 1e-6)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    class_weights = torch.from_numpy(class_weights).to(device)
    print(f"  Class weights: {' '.join(f'{lbl}={class_weights[i]:.3f}' for i, lbl in enumerate(SUPERCLASS_LABELS))}")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    n_epochs   = 50
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs, pct_start=0.1,
    )
    criterion = AsymmetricLoss(weight=class_weights, gamma_pos=0.0, gamma_neg=4.0,
                               margin=0.05, label_smoothing=0.05)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch  = 0
    patience    = 10
    no_improve  = 0

    print(f"\n  Training for up to {n_epochs} epochs (patience={patience})...\n")
    print(f"  {'Epoch':>5} {'Loss':>8} {'TrainAcc':>9} {'ValAcc':>7} {'ValF1':>7} {'LR':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*9} {'-'*7} {'-'*7} {'-'*10}")

    for epoch in range(n_epochs):
        model.train()
        train_loss = train_correct = train_total = 0

        for batch in train_loader:
            batch_x, batch_aux, batch_y = batch
            batch_x   = batch_x.to(device)
            batch_aux = batch_aux.to(device)
            batch_y   = batch_y.to(device)

            if not torch.isfinite(batch_x).all():
                batch_x = torch.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad()
            logits = model(batch_x, batch_aux)
            loss   = criterion(logits, batch_y)

            if not torch.isfinite(loss):
                scheduler.step()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss    += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total   += batch_x.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        model.eval()
        val_correct = val_total = 0
        val_preds_all = []
        val_labels_all = []
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_aux, batch_y = batch
                batch_x   = batch_x.to(device)
                batch_aux = batch_aux.to(device)
                batch_y   = batch_y.to(device)
                logits    = model(batch_x, batch_aux)
                preds     = logits.argmax(1)
                val_correct += (preds == batch_y).sum().item()
                val_total   += batch_x.size(0)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(batch_y.cpu().numpy())

        val_acc = val_correct / val_total
        from sklearn.metrics import f1_score as compute_f1
        val_f1 = compute_f1(val_labels_all, val_preds_all, average="macro", zero_division=0)
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch  = epoch + 1
            no_improve  = 0
            marker = " *"
            torch.save({
                "model_state_dict":        model.state_dict(),
                "superclass_labels":       SUPERCLASS_LABELS,
                "superclass_descriptions": SUPERCLASS_DESCRIPTIONS,
                "val_accuracy":            best_val_acc,
                "val_f1_macro":            best_val_f1,
                "epoch":                   best_epoch,
                "n_params":                param_count,
                "use_aux_features":        True,
                "n_aux":                   N_AUX,
                "model_type":              "ECGNetJoint",
                "norm_mode":               "amplitude",  # v10+: amplitude-preserving
            }, MODEL_PATH)
        else:
            no_improve += 1

        print(f"  {epoch+1:>5} {train_loss:>8.4f} {train_acc:>9.1%} {val_acc:>7.1%} {val_f1:>7.3f} {lr:>10.6f}{marker}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    print(f"\n  Best val F1: {best_val_f1:.3f} | Acc: {best_val_acc:.1%} (epoch {best_epoch})")

    # Test evaluation
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_aux, batch_y = batch
            batch_x   = batch_x.to(device)
            batch_aux = batch_aux.to(device)
            logits    = model(batch_x, batch_aux)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc   = np.mean(all_preds == all_labels)

    print(f"  Test accuracy: {test_acc:.1%}\n  Classification report:")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(all_labels, all_preds, target_names=SUPERCLASS_LABELS, zero_division=0))
    print("  Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\n  Model saved to {MODEL_PATH}\n  Done.\n")


# -------------------------------------------------------------
# Inference (used by app.py)
# -------------------------------------------------------------

def load_cnn_classifier(path=MODEL_PATH):
    """Load the trained CNN model (ECGNet or ECGNetJoint, auto-detected)."""
    if not os.path.exists(path):
        return None

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    use_aux    = checkpoint.get("use_aux_features", False)
    n_aux      = checkpoint.get("n_aux", N_AUX)
    model_type = checkpoint.get("model_type", "ECGNetJoint" if use_aux else "ECGNet")

    if model_type == "ECGNetTransformer":
        model = ECGNetTransformer(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=n_aux)
    elif use_aux:
        model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=n_aux)
    else:
        model = ECGNet(n_leads=N_LEADS, n_classes=N_CLASSES)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return {
        "model":                   model,
        "device":                  device,
        "superclass_labels":       checkpoint["superclass_labels"],
        "superclass_descriptions": checkpoint["superclass_descriptions"],
        "val_accuracy":            checkpoint["val_accuracy"],
        "use_aux_features":        use_aux,
        "n_aux":                   n_aux,
        "model_type":              model_type,
        "norm_mode":               checkpoint.get("norm_mode", "zscore"),
    }


def predict_cnn(model_data, signal_12, fs=500, sex="M", age=50):
    """
    Predict ECG superclass from a 12-lead signal.

    Args:
        model_data: dict from load_cnn_classifier()
        signal_12:  (N, 12) numpy array in mV (physical units)
        fs:         sampling rate (used for resampling if needed)
        sex:        "M" or "F" (used by ECGNetJoint for Cornell threshold)
        age:        patient age in years (used by ECGNetJoint)

    Returns:
        dict: prediction, description, confidence, probabilities
    """
    model     = model_data["model"]
    device    = model_data["device"]
    use_aux   = model_data.get("use_aux_features", False)

    # --- prepare raw signal in mV for voltage feature extraction ---
    raw = signal_12.copy()
    if len(raw) < SIGNAL_LEN:
        raw = np.vstack([raw, np.zeros((SIGNAL_LEN - len(raw), raw.shape[1]))])
    else:
        raw = raw[:SIGNAL_LEN]

    # Resample to 500 Hz if needed
    if fs != 500 and fs > 0:
        from scipy.signal import resample
        raw = resample(raw, SIGNAL_LEN, axis=0)

    raw = raw.T.astype(np.float32)         # (12, 5000) raw mV
    raw = np.clip(raw, -20.0, 20.0)

    # Normalize for CNN input -- dispatch based on what the model was trained with
    norm_mode = model_data.get("norm_mode", "zscore")
    if norm_mode == "amplitude":
        sig = _normalize_signal(raw)           # amplitude-preserving (v10+)
    else:
        sig = _normalize_signal_zscore(raw)    # per-lead z-score (v9 and earlier)
    x   = torch.from_numpy(sig).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_aux:
            n_aux_model = model_data.get("n_aux", N_AUX)
            aux_feats = extract_voltage_features(raw, sex=sex, age=age)[:n_aux_model]
            aux_t     = torch.from_numpy(aux_feats).unsqueeze(0).to(device)
            logits    = model(x, aux_t)
        else:
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = SUPERCLASS_LABELS[pred_idx]
    prob_dict  = {SUPERCLASS_LABELS[i]: round(float(p), 3) for i, p in enumerate(probs)}

    return {
        "prediction":   pred_label,
        "description":  SUPERCLASS_DESCRIPTIONS.get(pred_label, pred_label),
        "confidence":   float(probs[pred_idx]),
        "probabilities": prob_dict,
    }


if __name__ == "__main__":
    import sys
    use_multi = "--multi" in sys.argv
    if use_multi:
        print("  Multi-dataset mode: training on all available datasets")
    train(use_multi=use_multi)
