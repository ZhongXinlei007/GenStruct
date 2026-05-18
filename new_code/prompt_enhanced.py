#!/usr/bin/env python3
"""
增强版Context推理 - 支持LLM上下文增强
基于LLMAEL的上下文增强策略
"""

from LLM_calls import load_llm, llm_call
import json
from tqdm import tqdm
import random
import argparse


def read_prompt(file_name):
    prompt_dict = {}
    with open(file_name) as prompt_f:
        for line in prompt_f:
            line = json.loads(line.strip())
            prompt_dict[line['id']] = line
    return prompt_dict


def read_cot(cot_file_name):
    cot_index_dict = {}
    i = 0
    with open(cot_file_name) as input_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            prompt = line['context']
            prompt += 'Answer: {}\n\n'.format(line['gpt_ans'].replace('\n',''))
            cot_index_dict[i] = prompt
            i += 1
    return cot_index_dict


def context_prompt_enhanced(src_dict, datasets, cot_index_dict, instruction_dict, prompt_id=0, ent_des='summary'):
    """
    增强版Context Prompt - 支持LLM上下文增强
    """
    system_content = "You're an entity disambiguator. I'll give you the description of entity disambiguation and some tips on entity disambiguation, you should pay attention to these textual features:\n\n"
    system_content += instruction_dict[prompt_id]['prompt']

    '''category and sentence sim'''
    cot_index = src_dict['cot_index']
    cot_case = cot_index_dict[cot_index]

    content = 'The following example will help you understand the task:\n\n'
    content += cot_case

    content += "Now, I'll give you a mention, a context (with LLM-enhanced description), and a list of candidates entities.\n\n"
    content += f'Mention: {src_dict["mention"]}\n'

    # 检查是否有LLM增强上下文
    if 'llm_enhanced_context' in src_dict and src_dict['llm_enhanced_context']:
        # 使用LLM增强上下文
        original_context = f"{src_dict['left_context']} {src_dict['mention']} {src_dict['right_context']}"
        original_context = ' '.join(original_context.split())

        llm_context = src_dict['llm_enhanced_context']

        # 根据join_strategy组合上下文
        join_strategy = src_dict.get('join_strategy', 4)

        if join_strategy == 0:
            # 仅LLM上下文
            context = llm_context
        elif join_strategy == 1:
            # LLM上下文 + 原始上下文
            context = f"{llm_context}\n{original_context}"
        elif join_strategy == 2:
            # LLM上下文 + 原始上下文
            context = f"{llm_context}\n{original_context}"
        elif join_strategy == 3:
            # 原始上下文 + LLM上下文
            context = f"{original_context}\n{llm_context}"
        else:  # join_strategy == 4 (推荐)
            # 原始上下文 + LLM上下文
            context = f"{original_context}\n\nLLM Enhanced Context: {llm_context}"
    else:
        # 使用原始上下文
        context = src_dict['left_context'] + ' ###' + src_dict['mention'] + '### ' + src_dict['right_context']
        context = context.strip()
        context = ' '.join(context.split())

    content += f'Context: {context}\n'

    # 添加候选实体
    candidates = random.sample(src_dict['candidates'], len(src_dict['candidates']))
    i = 1
    for cand in candidates:
        if datasets == 'zeshel':
            cand_entity = f'{cand["title"]}.{cand[ent_des]}'
            content += f'Entity {cand["document_id"]}:{cand_entity}\n'
        else:
            cand_entity = f'{cand["name"]}.{cand[ent_des]}'
            content += f'Entity {i}:{cand_entity}\n'
        i += 1

    content += '\n'
    content += """You need to determine which candidate entity is more likely to be the mention. Please refer to the above example, give your reasons, and finally answer id of the entity and the name of the entity. If all candidate entities are not appropriate, you can answer '-1.None'. In addition, provide your confidence score from 1 to 5 as 'Confidence: <number>'."""

    return system_content, content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Context Entity Linking')
    parser.add_argument('--dataset_name', '-d', type=str, required=True, help="Dataset Name")
    parser.add_argument('--dataset_path', type=str, help="Dataset Path", default=None)
    parser.add_argument('--model_name', '-m', type=str, required=True, help='Model Name')
    parser.add_argument('--exp_name','-e',type=str, required=True, default='test', help='Exp Name')
    parser.add_argument('--model_path','-p',type=str, required=True, help="Path to model")
    parser.add_argument('--func_name','-f',type=str, required=True, help="Function(prompt) used")
    parser.add_argument('--output_key', type=str, help='output json key', default='llm_response')
    parser.add_argument('--instruction_dict', type=str, help='instruction file path', default='prompt/prompt.jsonl')
    parser.add_argument('--COT_pool', type=str, help='COT file path', default='datasets/aida_train_COT_pool.jsonl')
    parser.add_argument('--test', action='store_true', help="if Test", default=False)

    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    model_name = args.model_name
    exp_name = args.exp_name
    func_name = args.func_name
    full_flag = False if args.test else True

    print(f"Loading model '{model_name}' from path: {args.model_path}")
    pipeline = load_llm(model_name, args.model_path)

    input_file_name = args.dataset_path
    output_file_name = 'result/{}.json'.format(args.exp_name)
    output_key = args.output_key

    if func_name == 'context':
        instruction_dict = read_prompt(args.instruction_dict)
        cot_index_dict = read_cot(args.COT_pool)
    elif func_name in ['merge', 'point_wise']:
        instruction_dict = read_prompt(args.instruction_dict)

    if not full_flag:
        # 测试模式 - 只处理前几条
        with open(input_file_name) as input_f, \
            open(output_file_name, 'w') as output_f:
            for i, line in tqdm(enumerate(input_f)):
                if i >= 3:
                    break
                line = json.loads(line.strip())

                if func_name == 'context':
                    system_prompt, prompt = context_prompt_enhanced(
                        line, dataset_name, cot_index_dict=cot_index_dict,
                        instruction_dict=instruction_dict
                    )
                elif func_name == 'summary':
                    prompt = summary_prompt(line)
                elif func_name == 'category':
                    prompt = category_prompt(line)
                elif func_name == 'prior':
                    system_prompt, prompt = prior_prompt(line, dataset_name)
                elif func_name == 'merge':
                    system_prompt, prompt = merge_prompt(line, dataset_name, instruction_dict)

                response = llm_call(pipeline, prompt, system_prompt)
                line[output_key] = response
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
    else:
        # 完整处理
        with open(input_file_name) as input_f, \
            open(output_file_name, 'w') as output_f:
            for line in tqdm(input_f):
                line = json.loads(line.strip())

                if func_name == 'context':
                    system_prompt, prompt = context_prompt_enhanced(
                        line, dataset_name, cot_index_dict=cot_index_dict,
                        instruction_dict=instruction_dict
                    )
                elif func_name == 'summary':
                    prompt = summary_prompt(line)
                elif func_name == 'category':
                    prompt = category_prompt(line)
                elif func_name == 'prior':
                    system_prompt, prompt = prior_prompt(line, dataset_name)
                elif func_name == 'merge':
                    system_prompt, prompt = merge_prompt(line, dataset_name, instruction_dict)

                response = llm_call(pipeline, prompt, system_prompt)
                line[output_key] = response
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"Processing complete! Results saved to: {output_file_name}")