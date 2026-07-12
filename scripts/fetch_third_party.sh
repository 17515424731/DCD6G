#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

if [ ! -d third_party/TDAP/.git ]; then
  git clone https://github.com/claws-lab/TDAP.git third_party/TDAP
fi

if [ ! -d third_party/TDAP/models/DySAT_pytorch/.git ]; then
  git clone https://github.com/FeiGSSS/DySAT_pytorch third_party/TDAP/models/DySAT_pytorch
fi

echo "Third-party repositories are ready."

