"""
ecgfm_encoder.py
================
Standalone PyTorch implementation of the ECG-FM wav2vec2_cmsc encoder.
Loads pretrained weights from the MIMIC-IV/PhysioNet checkpoint without fairseq.

Architecture (from checkpoint cfg):
  in_d=12 → 4x Conv1d(stride=2) → (B, 256, T//16)
  LayerNorm(256) + Linear(256→768)
  ConvPosEncoder (weight-normed Conv1d, k=128, groups=16)
  12-layer TransformerEncoder (d=768, heads=12, ffn=3072, post-LN)
  Mean pool → (B, 768)

State-dict key structure (from checkpoint):
  feature_extractor.conv_layers.0.0.weight   Conv1d weight  (256, 12, 2)
  feature_extractor.conv_layers.0.2.weight   GroupNorm w    (256,)
  feature_extractor.conv_layers.0.2.bias     GroupNorm b    (256,)
  feature_extractor.conv_layers.{1,2,3}.0.weight           (256, 256, 2)
  layer_norm.{weight,bias}                   LayerNorm(256)
  post_extract_proj.{weight,bias}            Linear(256→768)
  conv_pos.pos_conv.0.weight_{g,v}           weight-normed Conv1d(768,768,k=128,g=16)
  conv_pos.pos_conv.0.bias
  encoder.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}
  encoder.layers.N.{fc1,fc2}.{weight,bias}
  encoder.layers.N.{self_attn_layer_norm,final_layer_norm}.{weight,bias}
  encoder.layer_norm.{weight,bias}

Usage:
    from ecgfm_encoder import ECGFMEncoder
    enc = ECGFMEncoder.from_pretrained("models/ecgfm/mimic_iv_ecg_physionet_pretrained.pt")
    enc.eval()
    with torch.no_grad():
        emb = enc(signal_tensor)   # (B, 768)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvFeatureExtractor(nn.Module):
    """
    4-layer strided Conv1d frontend.  extractor_mode='default':
      layer-0 : Conv1d → GELU → GroupNorm(n_out, n_out)   [indices 0, 1, 2]
      layers 1-3: Conv1d → GELU                            [indices 0, 1]
    Index layout matches state dict keys conv_layers.N.M.*
    """
    def __init__(self, in_d: int, layers_cfg: list):
        super().__init__()
        seqs = []
        for i, (dim, k, stride) in enumerate(layers_cfg):
            in_c = in_d if i == 0 else layers_cfg[i - 1][0]
            conv = nn.Conv1d(in_c, dim, k, stride=stride, bias=False)
            if i == 0:
                seqs.append(nn.Sequential(
                    conv,
                    nn.GELU(),
                    nn.GroupNorm(dim, dim, affine=True),
                ))
            else:
                seqs.append(nn.Sequential(conv, nn.GELU()))
        self.conv_layers = nn.ModuleList(seqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        return x


class _ConvPosWrapper(nn.Module):
    """
    Convolutional positional encoder with weight normalisation.
    Matches state dict prefix 'conv_pos.pos_conv.0.*' (weight_g / weight_v / bias).
    """
    def __init__(self, embed_dim: int, kernel_size: int, groups: int):
        super().__init__()
        conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size,
            padding=kernel_size // 2, groups=groups,
        )
        # weight_norm adds weight_g / weight_v parameters
        self.pos_conv = nn.Sequential(
            nn.utils.weight_norm(conv, name="weight", dim=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x_t  = x.transpose(1, 2)                  # (B, D, T)
        pos  = self.pos_conv(x_t)                  # (B, D, T+1)  even kernel → +1
        pos  = pos[:, :, :x_t.size(2)]             # trim back to T
        return pos.transpose(1, 2)                 # (B, T, D)


class _MultiheadSelfAttn(nn.Module):
    """
    Self-attention with separate q/k/v projections to match fairseq state dict keys
    (self_attn.q_proj / k_proj / v_proj / out_proj).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q_proj    = nn.Linear(embed_dim, embed_dim)
        self.k_proj    = nn.Linear(embed_dim, embed_dim)
        self.v_proj    = nn.Linear(embed_dim, embed_dim)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh   = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, dh).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class _TransformerEncoderLayer(nn.Module):
    """
    Post-LN transformer layer (layer_norm_first=False, wav2vec2 default).
    Residual → Dropout → Add → Norm (both for attn and FFN sub-layers).
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.self_attn            = _MultiheadSelfAttn(embed_dim, num_heads, attn_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1                  = nn.Linear(embed_dim, ffn_dim)
        self.fc2                  = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm     = nn.LayerNorm(embed_dim)
        self.dropout              = nn.Dropout(dropout)
        self.activation           = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer
        x = x + self.dropout(self.self_attn(x))
        x = self.self_attn_layer_norm(x)
        # FFN sub-layer
        residual = x
        x = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = residual + self.dropout(x)
        x = self.final_layer_norm(x)
        return x


class _TransformerEncoder(nn.Module):
    """Wrapper so state dict keys match 'encoder.layers.N.*' and 'encoder.layer_norm.*'."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(embed_dim, ffn_dim, num_heads, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class ECGFMEncoder(nn.Module):
    """
    Standalone ECG-FM encoder for fine-tuning / feature extraction.
    Excludes pretraining heads (quantizer, final_proj, project_q, mask_emb).

    Args:
        in_d            : number of input channels (12 leads)
        embed_dim       : transformer hidden size (768)
        ffn_dim         : FFN intermediate size (3072)
        num_heads       : attention heads (12)
        num_layers      : transformer layers (12)
        dropout         : general dropout (0.1)
        attn_dropout    : attention-weight dropout (0.1)
        conv_layers_cfg : list of (channels, kernel, stride) for conv frontend
        conv_pos_kernel : positional conv kernel size (128)
        conv_pos_groups : positional conv groups (16)
    """

    def __init__(
        self,
        in_d: int = 12,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        conv_layers_cfg: list = None,
        conv_pos_kernel: int = 128,
        conv_pos_groups: int = 16,
    ):
        super().__init__()
        if conv_layers_cfg is None:
            conv_layers_cfg = [(256, 2, 2)] * 4
        extractor_dim = conv_layers_cfg[-1][0]  # 256

        self.feature_extractor = _ConvFeatureExtractor(in_d, conv_layers_cfg)
        self.layer_norm        = nn.LayerNorm(extractor_dim)
        self.post_extract_proj = nn.Linear(extractor_dim, embed_dim)
        self.dropout_input     = nn.Dropout(dropout)
        self.conv_pos          = _ConvPosWrapper(embed_dim, conv_pos_kernel, conv_pos_groups)
        self.encoder           = _TransformerEncoder(
            embed_dim, ffn_dim, num_heads, num_layers, dropout, attn_dropout
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        map_location: str = "cpu",
    ) -> "ECGFMEncoder":
        """Load weights from a fairseq wav2vec2_cmsc checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        cfg  = ckpt["cfg"]["model"]
        cfl  = eval(cfg.get("conv_feature_layers", "[(256, 2, 2)] * 4"))  # noqa: S307

        enc = cls(
            in_d            = cfg.get("in_d", 12),
            embed_dim       = cfg.get("encoder_embed_dim", 768),
            ffn_dim         = cfg.get("encoder_ffn_embed_dim", 3072),
            num_heads       = cfg.get("encoder_attention_heads", 12),
            num_layers      = cfg.get("encoder_layers", 12),
            dropout         = cfg.get("dropout", 0.1),
            attn_dropout    = cfg.get("attention_dropout", 0.1),
            conv_layers_cfg = cfl,
            conv_pos_kernel = cfg.get("conv_pos", 128),
            conv_pos_groups = cfg.get("conv_pos_groups", 16),
        )

        # Drop pretraining-only keys; keep only the encoder subset
        SKIP_PREFIXES = ("quantizer.", "final_proj.", "project_q.", "mask_emb")
        sd = {
            k: v for k, v in ckpt["model"].items()
            if not any(k.startswith(p) for p in SKIP_PREFIXES)
        }

        missing, unexpected = enc.load_state_dict(sd, strict=False)
        if missing:
            print(f"[ECGFMEncoder] Missing   keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[ECGFMEncoder] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        else:
            print(f"[ECGFMEncoder] Loaded {len(sd)} tensors — no unexpected keys.")

        return enc

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 12) float32 ECG in mV.
                T should be a multiple of 16 (e.g. 5000 for 10 s at 500 Hz → 312 frames).
        Returns:
            (B, 768) mean-pooled contextual embedding.
        """
        x = x.transpose(1, 2)                    # (B, 12, T)
        x = self.feature_extractor(x)            # (B, 256, T//16)
        x = x.transpose(1, 2)                    # (B, T//16, 256)
        x = self.layer_norm(x)
        x = self.post_extract_proj(x)            # (B, T//16, 768)
        x = self.dropout_input(x)
        x = x + self.conv_pos(x)                # add positional encoding
        x = self.encoder(x)                      # (B, T//16, 768)
        return x.mean(dim=1)                     # (B, 768)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).parent)

    CKPT = "models/ecgfm/mimic_iv_ecg_physionet_pretrained.pt"
    print(f"Loading ECG-FM encoder from {CKPT} ...")
    enc = ECGFMEncoder.from_pretrained(CKPT)
    enc.eval()
    total_params = sum(p.numel() for p in enc.parameters())
    print(f"  Parameters: {total_params:,}")

    # Synthetic 10-second 12-lead ECG @ 500 Hz
    x = torch.randn(2, 5000, 12)
    with torch.no_grad():
        emb = enc(x)
    print(f"  Input shape : {tuple(x.shape)}")
    print(f"  Output shape: {tuple(emb.shape)}  (expected (2, 768))")
    print("  Smoke-test PASSED." if emb.shape == (2, 768) else "  Smoke-test FAILED.")
