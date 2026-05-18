export CUDA_VISIBLE_DEVICES=2
python -u new_code/prompt.py \
  --dataset_name ace2004 \
  --dataset_path datasets/ace2004/listwise_filter_true.jsonl \
  --model_name Zephyr \
  --model_path /data2/home/E24301323/code/llm/zephyr-7b-beta \
  --func_name context \
  --exp_name  ace2004_context_zephyr
  #--test