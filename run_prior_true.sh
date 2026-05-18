#!/bin/bash
# Prior推理 - 使用改进过滤后的数据
export CUDA_VISIBLE_DEVICES=1
python3 -u new_code/prompt.py \
  --dataset_name msnbc \
  --dataset_path datasets/msnbc/listwise_filter_true.jsonl \
  --model_name Zephyr \
  --model_path /data2/home/E24301323/code/llm/zephyr-7b-beta \
  --func_name prior \
  --exp_name msnbc_prior_zephyr_true