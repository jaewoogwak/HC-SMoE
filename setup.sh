#!/usr/bin/env bash
# Recreate the Mixtral development environment captured by the lock files.
#
# These locks were generated from the working `mixtral` environment on
# Linux x86_64 with PyTorch CUDA 12.8 (cu128), suitable for Blackwell GPUs.
# Override MIXTRAL_ENV_NAME to use another name.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${MIXTRAL_ENV_NAME:-mixtral}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

CONDA_COMMAND="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
if [[ -z "$CONDA_COMMAND" && -x /opt/miniforge3/bin/conda ]]; then
    CONDA_COMMAND=/opt/miniforge3/bin/conda
fi

if [[ -z "$CONDA_COMMAND" ]]; then
    echo "conda is required. Install Miniforge/Conda, then run this script again." >&2
    exit 1
fi

eval "$("$CONDA_COMMAND" shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Creating Conda environment: $ENV_NAME"
    conda create --yes --name "$ENV_NAME" --file "$ROOT_DIR/conda-mixtral.lock"
else
    echo "Using existing Conda environment: $ENV_NAME"
fi

conda activate "$ENV_NAME"

echo "Installing exact Python packages from requirements-mixtral.lock"
python -m pip install \
    --extra-index-url "$PYTORCH_INDEX_URL" \
    --requirement "$ROOT_DIR/requirements-mixtral.lock"

python -m pip check
python - <<'PY'
import torch
import transformers

if torch.version.cuda != "12.8":
    raise RuntimeError(f"Expected the cu128 PyTorch build, got torch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"Python packages ready: torch={torch.__version__}, CUDA={torch.version.cuda}, transformers={transformers.__version__}")
PY

echo "Mixtral environment '$ENV_NAME' is ready."
