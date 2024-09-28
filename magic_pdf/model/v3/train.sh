#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
OUTPUT_DIR="${DIR}/checkpoint/v3/$(date +%F-%H)"
DATA_DIR="${DIR}/ReadingBank/"

mkdir -p "${OUTPUT_DIR}"

deepspeed train.py \
  --model_dir 'microsoft/layoutlmv3-large' \
  --dataset_dir "${DATA_DIR}" \
  --dataloader_num_workers 1 \
  --deepspeed ds_config.json \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --do_train \
  --do_eval \
  --logging_steps 100 \
  --bf16 \
  --seed 42 \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --warmup_steps 1000 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --remove_unused_columns False \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  "$@"
