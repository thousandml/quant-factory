#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=================================="
echo " ThousandML CQ Unified Installer"
echo "=================================="

if ! python -c 'import sys; assert sys.version_info[:2] == (3, 12)' ; then
    echo "ERROR: ThousandML currently requires Python 3.12."
    exit 1
fi

echo "Fetching ThousandML native libraries..."
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Installing Git LFS..."
    sudo apt-get update
    sudo apt-get install -y git-lfs
fi

git lfs install
git lfs pull

python -m pip install --upgrade pip setuptools wheel

echo "[1/5] Checking protected binaries..."

python - <<'PY'
from pathlib import Path
import sysconfig
suffix = sysconfig.get_config_var("EXT_SUFFIX")
required = {
    "Quantization": Path("lib/utils") / f"_cq_core{suffix}",
    "Inference runtime": Path("inference/cq") / f"runtime_ops{suffix}",
    "Inference kernels": Path("inference/cq") / f"cq_kernels{suffix}",
}

training = [
    "cq_linear",
    "kernel_backend",
    "runtime_geometry",
    "checkpoint_layout",
    "_cq_training_license",
]

for name in training:
    required["Training " + name] = (
        Path("training") / f"{name}{suffix}"
    )
missing = []
for label, path in required.items():
    if path.is_file():
        print(f"OK: {label}")
    else:
        print(f"MISSING: {label} ({path})")
        missing.append(str(path))

if missing:
    raise SystemExit(
        "Missing compiled binaries for this Python ABI. "
        "Build and distribute the matching protected customer binaries."
    )
PY

echo "[2/5] Installing CQ inference + CUDA support..."
python -m pip install ./inference
echo "[3/5] Installing shared training dependencies..."
python -m pip install \
    "peft>=0.20,<0.22" \
    "accelerate==1.10.1" \
    "datasets==4.1.1"

python -m  pip install --ignore-installed PyJWT

echo "[4/5] Checking dependency consistency..."
python -m pip check

echo "[5/5] Running installation checks..."
python - <<'PY'
import torch
import transformers
import peft
import cq.cq_kernels
import cq.runtime_ops
from lib.utils import _cq_core
from peft import LoraConfig, get_peft_model
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
print("ThousandML CQ imports: OK")
PY

echo "=================================="
echo " ThousandML installation complete"
echo "=================================="
echo "Quantize: ./quantize-model"
echo "Inference: vllm serve MODEL --quantization cq"
echo "Training: python training/train.py --help"
