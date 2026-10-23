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
    echo "ERROR: Git LFS is required."
    echo "Install it with: sudo apt-get install git-lfs"
    exit 1
fi

git lfs pull

python -m pip install --upgrade pip setuptools wheel

echo "[1/6] Checking protected binaries..."

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

echo "[2/6] Installing CQ inference + CUDA support..."
python -m pip install ./inference

echo "[3/6] Installing shared training dependencies..."
python -m pip install \
    "peft>=0.20,<0.22" \
    "accelerate==1.10.1" \
    "datasets==4.1.1"

echo "[4/6] Installing CQ-compatible training acceleration..."
# Build CUDA extensions against the PyTorch already installed by CQ inference.
# Avoid a dependency resolver upgrading/replacing the CUDA-enabled torch stack.
python -m pip install ninja packaging einops

# Select nvcc matching the installed PyTorch CUDA version, including pip CUDA toolkit.
CUDA_TOOLKIT_ROOT="$(python - <<'PYCUDA'
import sys, site, subprocess
from pathlib import Path
import torch
want = torch.version.cuda
if not want:
    raise SystemExit('CUDA-enabled PyTorch required for CQ CUDA extensions')
candidates = []
for root in (Path('/usr/local/cuda'), Path('/usr/local') / ('cuda-' + want)):
    candidates.append(root)
for root in site.getsitepackages():
    candidates.extend(Path(root).glob('nvidia/cu*/'))
for root in candidates:
    nvcc = root / 'bin/nvcc'
    if not nvcc.is_file():
        continue
    out = subprocess.run([str(nvcc), '--version'], capture_output=True, text=True)
    if out.returncode == 0 and ('release ' + want + ',') in out.stdout:
        print(root)
        break
else:
    raise SystemExit(f'ERROR: no nvcc matching PyTorch CUDA {want}. '
                     'Install the matching CUDA compiler; do not use CUDA 12.8 with cu130.')
PYCUDA
)"
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "Using CUDA toolkit: $CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version

# A40 is SM 8.6; override TORCH_CUDA_ARCH_LIST for other GPUs.
# Build from source because a matching cu130/torch2.13 wheel may not exist.
MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" \
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    python -m pip install causal-conv1d --no-build-isolation --no-cache-dir

# Keep torch/triton/transformers and CQ dependencies under CQ's control.
python -m pip install flash-linear-attention --no-deps

# Unsloth Zoo provides optional smart gradient checkpointing. Full Unsloth
# model patching is NOT enabled: it can replace CQ quantized projections.
# Do not allow the resolver to replace the CUDA 13 PyTorch stack.
python -m pip install unsloth-zoo --no-deps

echo "[5/6] Checking dependency consistency..."
python -m pip check

echo "[6/6] Running installation checks..."
python - <<'PY'
import torch
import transformers
import peft
import cq.cq_kernels
import cq.runtime_ops
import causal_conv1d
import fla
import unsloth_zoo.gradient_checkpointing as gc
assert callable(getattr(gc, 'patch_unsloth_smart_gradient_checkpointing', None)), (
    'Installed unsloth-zoo lacks the checkpointing API expected by training/unsloth_compat.py'
)
from lib.utils import _cq_core
from peft import LoraConfig, get_peft_model
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("causal_conv1d import: OK")
print("flash-linear-attention (fla) import: OK")
print("Unsloth Zoo checkpointing API: OK")
print("ThousandML CQ imports: OK")
PY

echo "=================================="
echo " ThousandML installation complete"
echo "=================================="
echo "Quantize: ./quantize-model"
echo "Inference: vllm serve MODEL --quantization cq"
echo "Training: python training/train.py --help"
echo "CQ-compatible acceleration defaults: Unsloth checkpointing + fused PyTorch AdamW"
echo "To disable: --checkpointing native --optimizer adamw_torch"
echo "NOTE: Unsloth fused LoRA, loss, RMSNorm and SwiGLU kernels are NOT wired into CQ training."
echo "      Installing a package alone cannot enable these kernels."
