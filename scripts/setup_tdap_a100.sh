#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dcd6g-tdap}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required on the server for this setup script." >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" python=3.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

# TDAP was developed with PyTorch 1.7.1. This cu110 wheel is the closest
# practical A100-era target; if your cluster already has a working torch/PyG
# module, load that module instead of running this script.
python -m pip install \
  torch==1.7.1+cu110 \
  torchvision==0.8.2+cu110 \
  torchaudio==0.7.2 \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install \
  torch-scatter==2.0.7 \
  torch-sparse==0.6.9 \
  torch-cluster==1.5.9 \
  torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-1.7.1+cu110.html

python -m pip install \
  torch-geometric==2.0.2 \
  torch-geometric-temporal==0.51.0 \
  numpy==1.21.2 \
  scipy==1.7.3 \
  scikit-learn==1.0.1 \
  pandas==1.3.4 \
  matplotlib==3.5.0 \
  networkx==2.6.3 \
  tqdm==4.62.3 \
  pyyaml==6.0 \
  tensorboardx==2.4.1

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY

