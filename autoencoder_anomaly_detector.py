"""
autoencoder_anomaly_detector.py
================================
Variational Autoencoder (VAE) for ECG anomaly detection.

Strategy:
- Train ONLY on NORMAL heartbeats (supervised: class_idx=0)
- Model learns to reconstruct "normal" signals perfectly
- When fed abnormal signals (MI/HYP/etc) → reconstruction error SPIKES
- Use reconstruction_error as confidence filter for CNN predictions

Usage:
    # Train
    python autoencoder_anomaly_detector.py --train
    
    # Inference
    from autoencoder_anomaly_detector import load_vae_detector, compute_anomaly_score
    detector = load_vae_detector()
    error, is_anomaly = compute_anomaly_score(signal_12, threshold=2.0)
"""

import os
import time
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# =============================================================================
# Config
# =============================================================================

VAE_MODEL_PATH = "models/ecg_vae_detector.pt"
SIGNAL_LEN = 5000  # 10s at 500Hz
N_LEADS = 12
LATENT_DIM = 32
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

# =============================================================================
# VAE Architecture (trained on NORMAL heartbeats only)
# =============================================================================

class ECG_VAE(nn.Module):
    """
    Variational Autoencoder for ECG signals.
    Learns to compress/reconstruct normal heartbeats.
    High reconstruction error indicates anomaly.
    """
    
    def __init__(self, input_len=SIGNAL_LEN, n_leads=N_LEADS, latent_dim=LATENT_DIM):
        super().__init__()
        self.input_len = input_len
        self.n_leads = n_leads
        self.latent_dim = latent_dim
        
        # ── Encoder: Signal → Latent distribution ──────────────
        self.encoder = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=16, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=16, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        
        # Compute encoder output size
        with torch.no_grad():
            dummy = torch.zeros(1, n_leads, input_len)
            enc_out = self.encoder(dummy)
            self.encoder_out_dim = enc_out.shape[1]
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)
        
        # ── Decoder: Latent → Signal ──────────────────────────
        self.fc_decode = nn.Linear(latent_dim, 128 * 32)  # Expand back
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=16, stride=4, padding=7, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, n_leads, kernel_size=16, stride=4, padding=7, output_padding=0),
        )
    
    def encode(self, x):
        """Encode signal to latent distribution."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector back to signal."""
        h = self.fc_decode(z)
        h = h.view(-1, 128, 32)  # Reshape for transposed conv: (batch, 128, 32)
        x_recon = self.decoder(h)  # (batch, n_leads, ~5000)
        
        # Ensure output has exactly the right length using interpolation
        if x_recon.shape[2] != self.input_len:
            x_recon = F.interpolate(
                x_recon, 
                size=self.input_len, 
                mode='linear', 
                align_corners=False
            )
        
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def reconstruct(self, x):
        """Reconstruct signal without sampling (inference mode)."""
        mu, _ = self.encode(x)
        x_recon = self.decode(mu)
        return x_recon


def vae_loss(x_recon, x_true, mu, logvar, beta=0.1):
    """
    VAE loss = reconstruction loss + KL divergence.
    
    - Reconstruction loss: how well model reconstructs normal signals
    - KL divergence: regularize latent distribution to standard normal
    - beta: weight on KL term (0.1 = focus more on reconstruction fidelity)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x_true, reduction='mean')
    
    # KL divergence (analytical)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# =============================================================================
# Dataset (NORMAL heartbeats only)
# =============================================================================

class NormalECGDataset(Dataset):
    """Dataset of only NORMAL heartbeats for VAE training."""
    
    def __init__(self, record_paths, signal_cache):
        self.record_paths = record_paths
        self.signal_cache = signal_cache
    
    def __len__(self):
        return len(self.record_paths)
    
    def __getitem__(self, idx):
        sig = self.signal_cache[self.record_paths[idx]].copy()  # (5000, 12)
        
        # Ensure shape is (5000, 12), transpose to (12, 5000) for Conv1d
        if sig.shape[0] == 12:
            # Already (12, 5000), no transpose needed
            sig_out = sig
        else:
            # (5000, 12), need to transpose
            sig_out = sig.T  # Now (12, 5000)
        
        sig_out = sig_out.astype(np.float32)
        
        # Normalize per-lead (across time dimension)
        mean = sig_out.mean(axis=1, keepdims=True)
        std = sig_out.std(axis=1, keepdims=True) + 1e-8
        sig_out = (sig_out - mean) / std
        
        return torch.from_numpy(sig_out)  # (12, 5000)


# =============================================================================
# Training
# =============================================================================

def train_vae(use_multi=False):
    """Train VAE on NORMAL heartbeats only."""
    
    print("\n" + "="*70)
    print("  ECG VAE Anomaly Detector -- Training")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Load dataset
    from cnn_classifier import load_dataset, LABEL_TO_IDX
    paths, labels, folds = load_dataset()
    paths = np.array(paths)
    labels = np.array(labels)
    folds = np.array(folds)
    
    # Get NORMAL heartbeats only (class 0) from training folds
    train_mask = (folds < 10) & (labels == 0)  # Normal + training folds
    normal_paths = paths[train_mask].tolist()
    
    print(f"\n  Found {len(normal_paths)} NORMAL heartbeats for VAE training")
    print(f"  (using training folds 1-9 only, class='NORM')")
    
    # Pre-load signals
    from cnn_classifier import preload_signals
    print(f"\n  Pre-loading signals...")
    all_unique_paths = list(set(normal_paths))
    signal_cache = preload_signals(all_unique_paths)
    
    # Create dataset
    dataset = NormalECGDataset(normal_paths, signal_cache)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        collate_fn=lambda batch: torch.stack(batch)  # Ensure (batch, channels, timesteps)
    )
    
    # Model
    model = ECG_VAE(input_len=SIGNAL_LEN, n_leads=N_LEADS, latent_dim=LATENT_DIM).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {param_count:,}")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n  Training for {EPOCHS} epochs...")
    print(f"  {'Epoch':>5} {'Loss':>10} {'Recon':>10} {'KL':>10}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    
    # Debug: check first batch shape
    sample_batch = next(iter(loader))
    print(f"  DEBUG: First batch shape: {sample_batch.shape}")
    print(f"  Expected: (batch={BATCH_SIZE}, channels={N_LEADS}, timesteps={SIGNAL_LEN})")
    print()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch_idx, x in enumerate(loader):
            x = x.to(device)
            
            # Forward pass
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=0.1)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        # Average over batches
        n_batches = len(loader)
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        
        print(f"  {epoch+1:5d} {avg_loss:10.4f} {avg_recon:10.4f} {avg_kl:10.4f}", end="")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), VAE_MODEL_PATH)
            print(" ✓ (saved)")
        else:
            print()
    
    print(f"\n  Best model saved to {VAE_MODEL_PATH}")
    return model


# =============================================================================
# Inference
# =============================================================================

def load_vae_detector():
    """Load trained VAE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECG_VAE(input_len=SIGNAL_LEN, n_leads=N_LEADS, latent_dim=LATENT_DIM).to(device)
    
    if os.path.exists(VAE_MODEL_PATH):
        model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
        print(f"✓ Loaded VAE from {VAE_MODEL_PATH}")
    else:
        print(f"⚠️ VAE not found at {VAE_MODEL_PATH}. Run training first.")
        return None
    
    model.eval()
    return {"model": model, "device": device}


def compute_anomaly_score(signal_12, detector_data, fs=500, threshold=None):
    """
    Compute anomaly score for a signal.
    
    Args:
        signal_12: (N, 12) numpy array
        detector_data: dict from load_vae_detector()
        fs: sampling rate
        threshold: optional anomaly threshold (if None, only return score)
    
    Returns:
        (reconstruction_error, is_anomaly_bool)
        reconstruction_error: MSE between input and VAE reconstruction
        is_anomaly_bool: True if error > threshold (or None if threshold not provided)
    """
    if detector_data is None:
        return None, None
    
    model = detector_data["model"]
    device = detector_data["device"]
    
    # Pad/truncate to SIGNAL_LEN
    sig = signal_12.copy()
    if len(sig) < SIGNAL_LEN:
        pad = np.zeros((SIGNAL_LEN - len(sig), sig.shape[1]))
        sig = np.vstack([sig, pad])
    else:
        sig = sig[:SIGNAL_LEN]
    
    # Normalize per-lead
    sig = sig.T.astype(np.float32)  # (12, 5000)
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-8
    sig_norm = (sig - mean) / std
    
    # Forward pass
    x = torch.from_numpy(sig_norm).unsqueeze(0).to(device)  # (1, 12, 5000)
    
    with torch.no_grad():
        x_recon = model.reconstruct(x)
    
    # Compute reconstruction error (MSE)
    recon_error = F.mse_loss(x_recon, x).item()
    
    # Determine if anomaly
    is_anomaly = None
    if threshold is not None:
        is_anomaly = recon_error > threshold
    
    return recon_error, is_anomaly


# =============================================================================
# Hybrid CNN-VAE Filtering (for HYP class)
# =============================================================================

def predict_cnn_with_anomaly_filter(cnn_model_data, vae_model_data, signal_12, fs=500, 
                                     hyp_threshold=2.0):
    """
    Predict ECG class using CNN, then filter HYP predictions with VAE anomaly score.
    
    Logic:
    - If CNN predicts class X (not HYP): return X (no filter)
    - If CNN predicts HYP:
      - Compute anomaly score (reconstruction error)
      - If anomaly_score > hyp_threshold: ACCEPT HYP (truly abnormal)
      - If anomaly_score < hyp_threshold: REJECT HYP → downgrade to NORM or runner-up
    
    Returns:
        {
            "prediction": class_name,
            "confidence": float,
            "cnn_probabilities": dict,
            "anomaly_score": float or None,
            "vae_filtered": bool,
        }
    """
    from cnn_classifier import predict_cnn, SUPERCLASS_LABELS
    
    # CNN baseline prediction
    cnn_result = predict_cnn(cnn_model_data, signal_12, fs)
    pred_class = cnn_result["prediction"]
    cnn_probs = cnn_result["probabilities"]
    
    # Check if VAE filtering applies
    vae_filtered = False
    anomaly_score = None
    final_pred = pred_class
    
    if pred_class == "HYP" and vae_model_data is not None:
        # Compute anomaly score
        anomaly_score, _ = compute_anomaly_score(signal_12, vae_model_data)
        
        # If anomaly score is LOW, HYP is likely FALSE POSITIVE
        # (Signal looks "normal" to VAE but CNN thinks HYP)
        if anomaly_score is not None and anomaly_score < hyp_threshold:
            vae_filtered = True
            # Downgrade to top alternative class
            filtered_probs = dict(cnn_probs)
            filtered_probs["HYP"] = 0.0  # Zero out HYP
            final_pred = max(filtered_probs, key=filtered_probs.get)
    
    return {
        "prediction": final_pred,
        "confidence": cnn_result["confidence"],
        "cnn_probabilities": cnn_probs,
        "anomaly_score": anomaly_score,
        "vae_filtered": vae_filtered,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if "--train" in sys.argv:
        print("  Starting VAE training on NORMAL heartbeats...")
        train_vae()
    else:
        print("\nUsage: python autoencoder_anomaly_detector.py --train")
        print("\nOr import in code:")
        print("  from autoencoder_anomaly_detector import load_vae_detector, compute_anomaly_score")
        print("  detector = load_vae_detector()")
        print("  error, is_anomaly = compute_anomaly_score(signal_12, detector)")
