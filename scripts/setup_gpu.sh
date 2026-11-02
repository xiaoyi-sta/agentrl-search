#!/usr/bin/env bash
# One-shot setup on a fresh GPU box (AutoDL / any CUDA 12 machine).
#   bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$HF_HOME"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/pip install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ ! -f /root/autodl-tmp/models/Qwen3-1.7B-ms/model.safetensors.index.json ]; then
  .venv/bin/pip install -q modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
  .venv/bin/modelscope download --model Qwen/Qwen3-1.7B \
    --local_dir /root/autodl-tmp/models/Qwen3-1.7B-ms
fi

.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "setup done"
