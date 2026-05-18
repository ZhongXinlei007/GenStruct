#!/usr/bin/env python3
"""
处理LLM增强推理结果并评估
"""

import json
import argparse
from tqdm import tqdm


def process_result(dataset_name):
    """处理LLM增强context推理结果"""

    input_file = f'result/{dataset_name}_context_llm_enhanced.json'
    output_file = f'result/{dataset_name}_context_processed_llm_enhanced.jsonl'

    print(f"处理 {dataset_name} 的LLM增强context结果...")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print()

    with open(input_file) as input_f, \
        open(output_file, 'w') as output_f:

        for line in tqdm(input_f):
            line = json.loads(line.strip())

            # 提取预测结果
            llm_res = line.get('llm_response', '')

            # 简单解析：提取实体ID
            pred_num = -1
            pred_entity = None

            if llm_res:
                # 查找数字ID
                import re
                numbers = re.findall(r'\b\d+\b', llm_res)
                if numbers:
                    try:
                        pred_num = int(numbers[0])
                        # 查找对应的实体名称
                        for cand in line['candidates']:
                            if cand['wiki_id'] == pred_num:
                                pred_entity = cand['name']
                                break
                    except:
                        pass

            line['context_pred_num'] = pred_num
            line['context_pred_entity'] = pred_entity
            line['context_confidence'] = 0.8  # 默认置信度

            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"处理完成！结果保存到: {output_file}")
    return output_file


def evaluate_result(dataset_name):
    """评估LLM增强结果"""

    input_file = f'result/{dataset_name}_context_processed_llm_enhanced.jsonl'

    print(f"\n评估 {dataset_name} 的LLM增强context结果...")

    total = 0
    correct = 0

    with open(input_file) as f:
        for line in f:
            data = json.loads(line.strip())
            total += 1

            pred_num = data.get('context_pred_num', -1)
            correct_num = data.get('ans_id', -1)

            if pred_num == correct_num:
                correct += 1

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"总提及数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')

    args = parser.parse_args()

    # 处理结果
    process_result(args.dataset_name)

    # 评估结果
    evaluate_result(args.dataset_name)

    print(f"\n下一步: 与原始context结果对比")
    print(f"python3 new_code/eval.py --dataset_name {args.dataset_name} --file_path result/{args.dataset_name}_context_processed_llm_enhanced.jsonl --answer_key context_pred_num")