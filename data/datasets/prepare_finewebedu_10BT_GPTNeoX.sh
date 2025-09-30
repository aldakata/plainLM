#!/bin/bash

# This script will download and preprocess FineWebEdu-100BT.
# Expect some token loss by batched concat_chunk.

export SOFT_FILELOCK=1
export HF_HOME=/fast/atatjer/hf_fast
export TMPDIR=/fast/atatjer/tmp
export HOME=/fast/atatjer/tmp 

mkdir -p /fast/atatjer/tmp

PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="/fast/atatjer/data/fwedu_sample_10BT" \
  --cache_path="/fast/atatjer/tmp" \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-10BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048 \
  --split_train_valid=True \
  --n_tokens_valid=10000000
