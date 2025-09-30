#!/bin/bash
export FAST=/fast/atatjer
export SOFT_FILELOCK=1
export HF_HOME=/fast/atatjer/hf_fast          # if download
export HOME=/home/atatjer
export UV_CACHE_DIR=/home/atatjer/.cache
export UV_PYTHON_INSTALL_DIR=/home/atatjer/.local/share/uv/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_TRUST_REMOTE_CODE=True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
source /home/atatjer/src/plainLM/.venv/bin/activate