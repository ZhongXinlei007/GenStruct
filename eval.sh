#result/wiki/wiki_test_prompt0_sum13B_13B_nocot_prompt1.jsonl
# python new_code/eval.py \
#   --dataset_name wiki \
#   --file_path result/wiki_mfj.jsonl \
#   --answer_key final_pred_num
python3 eval_recall_at_k.py \
  --dataset_name wiki \
  --file_path result/wiki_mfj_true.json \
  --answer_key final_pred_num \
  --k_list "1"