#!/usr/bin/env python3
"""
增强版评估脚本 - 支持Recall@K计算
包括: Accuracy, Recall@1, Recall@2, Recall@5, Recall@10
"""

import json
from tqdm import tqdm
import argparse


def recall_at_k_eval(file_path, answer_key='final_pred_num', k_list=[1, 2, 5, 10]):
    """
    计算Recall@K指标

    Args:
        file_path: 评估文件路径
        answer_key: 预测结果的字段名
        k_list: 要计算的K值列表，如[1, 2, 5, 10]
    """

    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    total = len(data)
    print(f"总提及数: {total}")
    print(f"评估字段: {answer_key}")
    print("="*60)
    print(f"{'K':<10} {'命中数':<10} {'Accuracy':<15} {'说明'}")
    print("="*60)

    results = {}

    for k in k_list:
        hit_count = 0

        for item in data:
            # 获取正确答案ID
            correct_id = item.get('ans_id')

            # 获取候选实体列表
            candidates = item.get('candidates', [])

            if not candidates:
                continue

            # 检查正确答案是否在前K个候选中
            # 假设candidates已经按相关性排序
            if len(candidates) >= k:
                top_k_candidates = candidates[:k]
            else:
                top_k_candidates = candidates

            # 检查正确答案是否在top_k中
            found = False
            for cand in top_k_candidates:
                cand_id = cand.get('wiki_id')

                # 处理ID类型可能不一致的问题（int vs str）
                try:
                    if int(cand_id) == int(correct_id):
                        found = True
                        break
                except:
                    if str(cand_id) == str(correct_id):
                        found = True
                        break

            if found:
                hit_count += 1

        # 计算Recall@K
        recall = hit_count / total if total > 0 else 0.0
        results[k] = {'hit': hit_count, 'recall': recall}

        # 打印结果
        if k == 1:
            description = "准确率（Hit@1）"
        else:
            description = f"正确答案在前{k}个候选中的比例"

        print(f"{k:<10} {hit_count:<10} {recall:<15.4f} {description}")

    print("="*60)

    # 额外统计：单候选情况
    single_cand = sum(1 for item in data if len(item.get('candidates', [])) == 1)
    print(f"\n额外统计:")
    print(f"单候选提及数: {single_cand} ({single_cand/total*100:.1f}%)")

    # 预测命中率
    pred_hit = 0
    total_pred = 0
    for item in data:
        pred_val = item.get(answer_key, -1)
        if pred_val != -1:
            total_pred += 1
            if pred_val == item.get('ans_id'):
                pred_hit += 1

    if total_pred > 0:
        pred_acc = pred_hit / total_pred
        print(f"预测准确率: {pred_hit}/{total_pred} ({pred_acc:.4f})")

    return results


def accuracy_eval(file_path, answer_key='final_pred_num'):
    """
    计算准确率（等价于Hit@1）
    与原eval.py的file_f1函数相同逻辑
    """
    all_num = 0
    hit_num = 0
    error_num = 0

    with open(file_path, 'r') as f:
        for line in f:
            all_num += 1
            item = json.loads(line)

            if answer_key not in item:
                error_num += 1
                continue

            pred_val = item[answer_key]

            # 处理数值ID
            if isinstance(pred_val, (int, float)) or (isinstance(pred_val, str) and pred_val.strip().lstrip('-').isdigit()):
                try:
                    pred_id = int(pred_val)
                except Exception:
                    error_num += 1
                    continue
                if pred_id == item.get('ans_id'):
                    hit_num += 1
                continue

            # 处理文本预测
            if not isinstance(pred_val, str) or len(pred_val) == 0:
                error_num += 1
                continue

            # 简单文本匹配（不做复杂解析）
            pred_lower = pred_val.lower().strip()

            # 检查是否在候选中
            found = False
            for cand in item.get('candidates', []):
                if 'name' in cand:
                    cand_name = cand['name'].lower().strip()
                    if cand_name in pred_lower or pred_lower in cand_name:
                        if cand.get('wiki_id') == item.get('ans_id'):
                            hit_num += 1
                            found = True
                            break

    acc = (hit_num / all_num) if all_num > 0 else 0.0
    print(f"\n准确率评估:")
    print(f"总提及数: {all_num}")
    print(f"命中数: {hit_num}")
    print(f"错误数: {error_num}")
    print(f"准确率: {acc:.4f}")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Evaluation with Recall@K')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--file_path', '-f', type=str, required=True, help="Path to Eval File")
    parser.add_argument('--answer_key', '-k', type=str, default='final_pred_num',
                       help="Answer Key in File (default: final_pred_num)")
    parser.add_argument('--k_list', type=str, default='1,2,5,10',
                       help="Comma-separated K values for Recall@K (default: 1,2,5,10)")
    parser.add_argument('--accuracy_only', action='store_true',
                       help="Only compute accuracy (same as original eval.py)")

    args = parser.parse_args()

    print("="*60)
    print(f"评估数据集: {args.dataset_name}")
    print(f"评估文件: {args.file_path}")
    print(f"答案字段: {args.answer_key}")
    print("="*60)
    print()

    if args.accuracy_only:
        # 只计算准确率
        accuracy_eval(args.file_path, args.answer_key)
    else:
        # 计算Recall@K
        k_list = [int(k.strip()) for k in args.k_list.split(',')]
        recall_at_k_eval(args.file_path, args.answer_key, k_list)
