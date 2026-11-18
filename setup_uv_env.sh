#!/usr/bin/env bash
set -euo pipefail

# Create the python env with uv (no conda) for a fresh server, e.g. taylor3.
# Code comes from git (git clone / git pull). Data is NOT in git.
#
# Usage (from the repo root):
#   bash setup_uv_env.sh [ARTIFACTS_DIR]
# ARTIFACTS_DIR (optional): folder holding the untracked artifacts
#   source_models/  data_cams/  pretrained-imgnet/
# They are symlinked into the repo root if missing.
#
# Datasets: the code reads $DATASETSH/datasets when HOST_XXX=ESON, e.g.
#   export HOST_XXX=ESON
#   export DATASETSH=$HOME          # -> $HOME/datasets/{GLAS,CAMELYON17_512}
# Activate with: source .venv/bin/activate

cd "$(dirname "$0")"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

"$UV" venv --python 3.10 .venv
"$UV" pip install --python .venv/bin/python \
  torch==2.0.0+cu117 torchvision==0.15.1+cu117 \
  --index-url https://download.pytorch.org/whl/cu117 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
"$UV" pip install --python .venv/bin/python -r reqs_uv.txt

ART="${1:-}"
if [[ -n "$ART" ]]; then
  for d in source_models data_cams pretrained-imgnet; do
    if [[ ! -e "$d" && -d "$ART/$d" ]]; then
      ln -s "$ART/$d" "$d"
      echo "linked $d -> $ART/$d"
    fi
  done
fi

.venv/bin/python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
