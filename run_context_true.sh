#!/bin/bash
# Context推理 - 使用改进过滤后的数据
export CUDA_VISIBLE_DEVICES=3
python3 -u new_code/prompt.py \
  --dataset_name msnbc \
  --dataset_path datasets/msnbc/listwise_filter_true.jsonl \
  --model_name Zephyr \
  --model_path /data2/home/E24301323/code/llm/zephyr-7b-beta \
  --func_name context \
  --exp_name msnbc_context_zephyr_true