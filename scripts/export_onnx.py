"""
Export ECGNetJoint model from PyTorch .pt checkpoint to ONNX format.

Usage:
    python scripts/export_onnx.py                      # export best model
    python scripts/export_onnx.py --quantize            # also create int8 quantized version
    python scripts/export_onnx.py --verify              # verify ONNX matches PyTorch output
    python scripts/export_onnx.py --checkpoint <path>   # export specific checkpoint

Output:
    EKGMobile/assets/models/ecg_v3.onnx            (full precision)
    EKGMobile/assets/models/ecg_v3_int8.onnx       (quantized, if --quantize)
    EKGMobile/assets/models/thresholds_v3.json      (copied)
    EKGMobile/assets/models/model_manifest.json     (SHA-256 hashes)
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# Project root
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cnn_classifier import ECGNetJoint, N_AUX, N_LEADS, SIGNAL_LEN
from dataset_chapman import MERGED_CODES
from dataset_challenge import V3_CODES, N_V3

# Constants
N_CLASSES = N_V3  # 26
BEST_MODEL = ROOT / "models" / "ecg_multilabel_v3_best.pt"
FALLBACK_MODEL = ROOT / "models" / "ecg_multilabel_v3.pt"
OUTPUT_DIR = ROOT / "EKGMobile" / "assets" / "models"
THRESHOLDS_SRC = ROOT / "models" / "thresholds_v3.json"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """Load ECGNetJoint from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ECGNetJoint(n_leads=N_LEADS, n_classes=N_CLASSES, n_aux=N_AUX)

    # Handle potential state dict key variations
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Input: signal ({N_LEADS}, {SIGNAL_LEN}) + aux ({N_AUX},)")
    print(f"  Output: logits ({N_CLASSES},)")
    return model


def export_onnx(model: torch.nn.Module, output_path: Path):
    """Export model to ONNX format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy inputs matching the model's expected shapes
    dummy_signal = torch.randn(1, N_LEADS, SIGNAL_LEN, dtype=torch.float32)
    dummy_aux = torch.randn(1, N_AUX, dtype=torch.float32)

    print(f"\nExporting to ONNX: {output_path}")
    # PyTorch >= 2.5 added `dynamo` parameter; default may be True (requires onnxscript).
    # Force legacy TorchScript exporter (no onnxscript needed) via dynamo=False.
    export_kwargs = dict(
        input_names=["signal", "aux"],
        output_names=["logits"],
        dynamic_axes=None,  # Fixed batch size 1 for mobile inference
        opset_version=13,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, (dummy_signal, dummy_aux), str(output_path),
                          dynamo=False, **export_kwargs)
    except TypeError:
        # dynamo kwarg not supported in this PyTorch version (< 2.5) — use default
        torch.onnx.export(model, (dummy_signal, dummy_aux), str(output_path),
                          **export_kwargs)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  ONNX exported: {size_mb:.1f} MB")
    return output_path


def quantize_onnx(input_path: Path, output_path: Path):
    """Apply dynamic int8 quantization to reduce model size."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("WARNING: onnxruntime not installed. Skipping quantization.")
        print("  Install: pip install onnxruntime")
        return None

    print(f"\nQuantizing to int8: {output_path}")
    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Quantized: {size_mb:.1f} MB")
    return output_path


def verify_onnx(model: torch.nn.Module, onnx_path: Path, n_samples: int = 20):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("WARNING: onnxruntime not installed. Skipping verification.")
        print("  Install: pip install onnxruntime")
        return False

    print(f"\nVerifying ONNX parity ({n_samples} samples)...")
    session = ort.InferenceSession(str(onnx_path))

    max_diff = 0.0
    for i in range(n_samples):
        # Generate random input
        signal = torch.randn(1, N_LEADS, SIGNAL_LEN, dtype=torch.float32)
        aux = torch.randn(1, N_AUX, dtype=torch.float32)

        # PyTorch inference
        with torch.no_grad():
            pt_logits = model(signal, aux).numpy()

        # ONNX inference
        ort_inputs = {
            "signal": signal.numpy(),
            "aux": aux.numpy(),
        }
        ort_logits = session.run(["logits"], ort_inputs)[0]

        # Compare
        diff = np.max(np.abs(pt_logits - ort_logits))
        max_diff = max(max_diff, diff)

        if diff > 1e-4:
            print(f"  Sample {i}: max diff = {diff:.6f} WARNING")
        elif i % 5 == 0:
            print(f"  Sample {i}: max diff = {diff:.8f} OK")

    print(f"\n  Max absolute difference: {max_diff:.8f}")
    if max_diff < 1e-4:
        print("  PASS: ONNX output matches PyTorch within 1e-4 tolerance")
        return True
    else:
        print("  FAIL: ONNX output diverges from PyTorch")
        return False


def generate_manifest(output_dir: Path):
    """Generate model_manifest.json with SHA-256 hashes for integrity verification."""
    manifest = {
        "version": "v3",
        "n_classes": N_CLASSES,
        "n_leads": N_LEADS,
        "signal_length": SIGNAL_LEN,
        "n_aux": N_AUX,
        "label_codes": V3_CODES,
        "files": {},
    }

    for f in output_dir.iterdir():
        if f.suffix in (".onnx", ".json") and f.name != "model_manifest.json":
            manifest["files"][f.name] = {
                "sha256": sha256_file(f),
                "size_bytes": f.stat().st_size,
            }

    manifest_path = output_dir / "model_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nManifest written: {manifest_path}")
    for name, info in manifest["files"].items():
        print(f"  {name}: {info['sha256'][:16]}... ({info['size_bytes']/1e6:.1f} MB)")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Export ECGNetJoint to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint (default: models/ecg_multilabel_v3_best.pt)")
    parser.add_argument("--quantize", action="store_true",
                        help="Also create int8 quantized version")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX matches PyTorch output")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for ONNX files")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif BEST_MODEL.exists():
        ckpt_path = BEST_MODEL
    elif FALLBACK_MODEL.exists():
        ckpt_path = FALLBACK_MODEL
    else:
        print(f"ERROR: No checkpoint found at {BEST_MODEL} or {FALLBACK_MODEL}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(ckpt_path)

    # Export to ONNX
    onnx_path = output_dir / "ecg_v3.onnx"
    export_onnx(model, onnx_path)

    # Quantize (optional)
    if args.quantize:
        quantized_path = output_dir / "ecg_v3_int8.onnx"
        quantize_onnx(onnx_path, quantized_path)

    # Copy thresholds
    if THRESHOLDS_SRC.exists():
        dst = output_dir / "thresholds_v3.json"
        shutil.copy(THRESHOLDS_SRC, dst)
        print(f"\nThresholds copied: {dst}")
    else:
        print(f"\nWARNING: {THRESHOLDS_SRC} not found. Thresholds not copied.")

    # Verify (optional but recommended)
    if args.verify:
        verify_onnx(model, onnx_path)

    # Generate integrity manifest (SHA-256 hashes for HIPAA 1.4.2)
    generate_manifest(output_dir)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Output: {output_dir}")
    print(f"  Model:  ecg_v3.onnx ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Next:   Bundle into EKGMobile/assets/models/ for mobile app")
    print("=" * 60)


if __name__ == "__main__":
    main()
