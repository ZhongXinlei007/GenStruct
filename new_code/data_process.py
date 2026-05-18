import json
from tqdm import tqdm
import os
import copy
import re
import numpy as np


def fomulate_data(line:dict):
    context = line['context']['text']
    context = context.split()
    left_context = ' '.join(context[:line['start_index']])
    right_context = ' '.join(context[line['end_index'] + 1:])
    return left_context, right_context

def concat_cands(mention_file_name, cand_id_file_name, doc_file_name, original_doc_file_name ,output_file_name):
    '''change json style from zeshel to aida'''
    id2doc = {}
    print('load cands')
    with open(doc_file_name) as doc_file:
        for line in doc_file:
            line = json.loads(line.strip())
            id2doc[line['document_id']] = line
    print('load context')
    id2doc_large = {}
    with open(original_doc_file_name) as doc_file:
        for line in doc_file:
            line = json.loads(line.strip())
            id2doc_large[line['document_id']] = line

    print('load cands id')
    id2cand = {}
    with open(cand_id_file_name) as cand_id_file:
        for line in cand_id_file:
            line = json.loads(line.strip())
            id2cand[line['mention_id']] = line
    output_f = open(output_file_name, 'w')

    print('concating...')
    with open(mention_file_name) as mention_file:
        for line in tqdm(mention_file):
            line = json.loads(line.strip())
            line['context'] = id2doc_large[line['context_document_id']]
            line['label'] = id2doc_large[line['label_document_id']]
            left_contex, right_context = fomulate_data(line)
            line['left_context'] = left_contex
            line['mention'] = line['text']
            line['right_context'] = right_context
            line['output'] = line['label']['title']
            cand_list = id2cand[line['mention_id']]['tfidf_candidates']
            cand_word_list = []
            for cand in cand_list:
                cand_word_list.append(id2doc[cand])
            line['candidates'] = cand_word_list
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            # exit()

def select_hit_entity_in_all(cand_id_file_name, doc_file_name, output_file_name):
    '''select entity used'''
    hit_set = set()
    with open(cand_id_file_name) as cand_id_file:
        for line in cand_id_file:
            line = json.loads(line.strip())
            cand_list = line['tfidf_candidates']
            for cand_id in cand_list:
                hit_set.add(cand_id)
    
    with open(doc_file_name) as doc_file,\
        open(output_file_name, 'w') as output_file:
        for line in tqdm(doc_file):
            line = json.loads(line)
            if line['document_id'] in hit_set:
                output_file.write(json.dumps(line, ensure_ascii=False) + '\n')

def split_file(input_file_path, input_file_name, num):
    output_file_list = []
    for i in range(num):
        output_f = open(input_file_path + 'split/' + input_file_name + '_{}.json'.format(i), 'w')
        output_file_list.append(output_f)
    
    with open(input_file_path + input_file_name + '.json') as input_f:
        for i, line in tqdm(enumerate(input_f)):
            index = i % num
            output_f = output_file_list[index]
            output_f.write(line)

def summary_llm_process(llm_response:str):
    return llm_response.split('\n\n')[-1]

def merge_file(input_file_list, output_file_name):
    output_file = open(output_file_name, 'w')
    mention_set = set()
    for input_file_name in input_file_list:
        with open(input_file_name) as input_file:
            for line in tqdm(input_file):
                line = json.loads(line.strip())
                if line['mention_id'] in mention_set:
                    continue
                else:
                    mention_set.add(line['mention_id'])
                # line['summary'] = summary_llm_process(line['summary'])
                output_file.write(json.dumps(line, ensure_ascii=False) + '\n')

def list2point(input_file_name, output_file_name):
    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:

        for line in tqdm(input_f):
            line = json.loads(line.strip())
            for cand in line['candidates']:
                new_line = copy.deepcopy(line)
                new_line['candidates'] = cand
                output_f.write(json.dumps(new_line, ensure_ascii=False) + '\n')

def phrase_pointwise_ans(llm_response:str):
    '''
    1. my answer is / I answer
    2. pos of yes & no, last pos
    '''
    pattern_list = ['[Aa]nswer is[:\s\n\*\"\']*[A-Za-z]+[\s\n\*\"\'\.]*', 
                    '[Aa]nswer[:\s\n\*\"\']*[A-Za-z]+[\s\n\*\"\'\.]*',
                    '[Cc]onclusion[:\s\n\*\"\']*[A-Za-z]+[\s\n\*\"\'\.]*',
                    '[Cc]onclusion is[:\s\n\*\"\']*[A-Za-z]+[\s\n\*\"\'\.]*']
    for pattern in pattern_list:
        res = re.findall(pattern, llm_response)
        if len(res) != 0:
            break
    if len(res) != 0: # find ans
        ans = res[-1]
        
        if 'no' in ans.lower():
            return False
        else:
            return True
    
    negative_list = ['"no"', "'no'", 'not related', 'cannot establish a relationship', '"No"', "'No'", ': no', ': No', '"NO"', "'NO'", ': NO']
    for neg_word in negative_list:
        if neg_word in llm_response:
            return False
    
    return True

def extract_confidence(llm_response: str) -> float:
    """从 LLM 输出中解析 1-5 的置信度，返回 [0,1] 归一化分数。找不到则返回 0.0。
    支持格式："Confidence: 4"、"confidence = 3"、"confidence 5/5"、或百分比如 "80%"。
    """
    if not isinstance(llm_response, str) or len(llm_response) == 0:
        return 0.0
    text = llm_response.strip()
    # 1) 显式 Confidence: N
    m = re.search(r"[Cc]onfidence\s*[:=]\s*(\d(?:\.\d)?)", text)
    if m:
        try:
            val = float(m.group(1))
            # 常规范围 1-5 归一化
            if 0.0 <= val <= 1.0:
                return max(0.0, min(1.0, val))
            return max(0.0, min(1.0, (val - 1.0) / 4.0))
        except Exception:
            pass
    # 2) N/5 形式
    m = re.search(r"(\d(?:\.\d)?)[\s]*/[\s]*5", text)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, (val - 1.0) / 4.0))
        except Exception:
            pass
    # 3) 百分比
    m = re.search(r"(\d{1,3})(?:\.\d+)?%", text)
    if m:
        try:
            val = float(m.group(1)) / 100.0
            return max(0.0, min(1.0, val))
        except Exception:
            pass
    return 0.0

def extract_confidence_raw_1to5(llm_response: str) -> float:
    """从 LLM 输出中解析原始 1-5 的置信度。找不到则返回 0.0（表示缺失）。"""
    if not isinstance(llm_response, str) or len(llm_response) == 0:
        return 0.0
    text = llm_response.strip()
    # 1) 显式 Confidence: N
    m = re.search(r"[Cc]onfidence\s*[:=]\s*(\d(?:\.\d)?)", text)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return max(1.0, min(5.0, 1.0 + 4.0 * val))
            return max(1.0, min(5.0, val))
        except Exception:
            pass
    # 2) N/5 形式
    m = re.search(r"(\d(?:\.\d)?)[\s]*/[\s]*5", text)
    if m:
        try:
            val = float(m.group(1))
            return max(1.0, min(5.0, val))
        except Exception:
            pass
    # 3) 百分比 -> 映射到 1-5
    m = re.search(r"(\d{1,3})(?:\.\d+)?%", text)
    if m:
        try:
            prob = float(m.group(1)) / 100.0
            return max(1.0, min(5.0, 1.0 + 4.0 * prob))
        except Exception:
            pass
    return 0.0

def phrase_category(llm_ans:str):
    category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']
    llm_ans = llm_ans.lower()
    hit_cat_dict = {}
    for category in category_list:
        category = category.lower()
        if category in llm_ans:
            pos = llm_ans.find(category)
            hit_cat_dict[category] = pos
    if len(hit_cat_dict) > 0:
        category, _ = sorted(hit_cat_dict.items(), key = lambda kv:(kv[1], kv[0]))[0]
    else:
        category = 'Any'
    return category

def point_wise_filter(input_file_name, output_file_name, fail_file_name):
    '''llm_pointwise_response'''
    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f, \
        open(fail_file_name, 'w') as fail_f:

        for line in tqdm(input_f):
            line = json.loads(line.strip())
            llm_res = line['llm_pointwise_response']
            if len(llm_res) == 0:
                fail_f.write(json.dumps(line, ensure_ascii=False) + '\n')
            elif phrase_pointwise_ans(llm_res):
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

def cut_context(left_context, right_context):
    '''
    for zeshel, some context too long
    so if output is None, pls cut context first
    '''
    left_pos = left_context.rfind('.')
    right_pos = right_context.find('.')
    first_pos = left_context.find('.')

    if left_pos == -1:
        cut_left_c = left_context
    else:
        cut_left_c = left_context[:first_pos + 1] + left_context[left_pos + 1:]
    
    if right_pos == -1:
        cut_right_c = right_context
    else:
        cut_right_c = right_context[:right_pos + 1]
    
    return cut_left_c, cut_right_c

def sentence_embed(file_path, sentence_file_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('hf_models/all-mpnet-base-v2')
    os.environ['CUDA_VISIBLE_DEVICES']='4'
    import torch
    torch.set_num_threads(10)
    
    emb_list = []
    with open(file_path) as input_f:
        for line in tqdm(input_f):
            src_dict = json.loads(line.strip())
            if 'cut_left_context' in src_dict.keys():
                context = src_dict['cut_left_context'] + src_dict['mention'] + src_dict['cut_right_context']
            else:
                context = src_dict['left_context'] + src_dict['mention'] + src_dict['right_context']
            context = context.strip()
            context = ' '.join(context.split())
            emb = model.encode([context])[0]
            emb_list.append(emb)

    emb_list = np.array(emb_list)
    print(emb_list.shape)
    np.save('EL_datasets/embedding/{}.npy'.format(sentence_file_name), emb_list)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def cot_select(src_file_name, src_embed_name, output_file_name, alpha = 0.5):
    category_dict = {}
    category_list = ['General reference','Culture and the arts','Geography and places','Health and fitness', 'History and events', 'Human activities', 'Mathematics and logic', 'Natural and physical sciences', 'People and self', 'Philosophy and thinking', 'Religion and belief systems', 'Society and social sciences', 'Technology and applied sciences']

    for cate in category_list:
        category_dict[cate.lower()] = []
    cot_embed_arr = np.load('EL_datasets/embedding/aida_train_merge.npy')
    cot_file = open('EL_datasets/COT_sample/final/aida_train_merge_listwise_repeated.jsonl')

    src_embed_list = np.load(src_embed_name)
    map_file = open(output_file_name, 'w')
    src_file = open(src_file_name)

    for line in cot_file:
        line = json.loads(line.strip())
        line_category = line['llm_category']

        for cate in category_list:
            if line_category.lower() == cate.lower():
                cate_list = category_dict[cate.lower()]
                cate_list.append(1)
                category_dict[cate.lower()] = cate_list
            else:
                cate_list = category_dict[cate.lower()]
                cate_list.append(0)
                category_dict[cate.lower()] = cate_list

    category_dict['any'] = [0]*65
    category_embed_dict = {}
    for k, v in category_dict.items():
        category_embed_dict[k] = np.array(v)

    src_embed_list = src_embed_list.tolist()
    src_file_list = src_file.readlines()
    
    for src_embed, line in tqdm(zip(src_embed_list, src_file_list)):
        line = json.loads(line.strip())
        line_category = line['llm_category']
        cate_embed = category_embed_dict[line_category.lower()]
        src_embed = np.array(src_embed)     # d
        src_embed = src_embed[:,np.newaxis]
        score_embed = np.dot(cot_embed_arr, src_embed)
        score_embed = np.squeeze(score_embed)
        score_embed = normalization(score_embed)
        score_embed = alpha * score_embed + (1 - alpha) * cate_embed
        score_embed = np.argsort(-score_embed)
        # print(score_embed[:10])
        score_list = score_embed.tolist()
        line['cot_index'] = score_list[0]
        # print(line['cot_index'])
        try:
            del line['llm_pointwise_response']
        except:
            pass
        map_file.write(json.dumps(line, ensure_ascii=False) + '\n')
# 增加一个新的参数 dataset_name
def process_file_line(input_file_name, output_file_name, func, dataset_name):
    with open(input_file_name) as input_f, \
        open(output_file_name, 'w') as output_f:

        for line_num, line in enumerate(tqdm(input_f), 1):
            try:
                line = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num} in {input_file_name}")
                print(f"Error: {e}")
                print(f"Line preview (first 200 chars): {line[:200] if len(line) > 200 else line}")
                continue
            if func == 'cut_context':
                cut_left, cut_right = cut_context(line["left_context"], line["right_context"])
                line['cut_left_context'] = cut_left
                line['cut_right_context'] = cut_right
            if func == 'category':
                category = phrase_category(line['llm_category'])
                line['llm_category'] = category
            if func == 'context':
                # --- 在调用时传入 dataset_name ---
                pred_ent, pred_num = line_f1(line, 'llm_response', dataset_name)
                line['context_pred_entity'] = pred_ent
                line['context_pred_num'] = pred_num
                line['context_confidence'] = extract_confidence(line.get('llm_response', ''))
                line['context_confidence_raw'] = extract_confidence_raw_1to5(line.get('llm_response', ''))
            if func == 'prior':
                # --- 在调用时传入 dataset_name ---
                pred_ent, pred_num = line_f1(line, 'llm_response', dataset_name)
                line['prior_pred_entity'] = pred_ent
                line['prior_pred_num'] = pred_num
                line['prior_confidence'] = extract_confidence(line.get('llm_response', ''))
                line['prior_confidence_raw'] = extract_confidence_raw_1to5(line.get('llm_response', ''))
            if func == 'summary':
                new_cand = []
                for cand in line['candidates']:
                    if len(cand['summary']) == 0:
                        cand['summary'] = add_summary(cand['text'])
                    new_cand.append(cand)
                line['candidates'] = new_cand
            if func == 'merge':
                pred_ent, pred_num = line_f1(line, 'llm_response')
                line['final_pred_entity'] = pred_ent
                line['final_pred_num'] = pred_num
            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

def point2list(point_file, list_file):
    with open(point_file) as input_f, \
        open(list_file, 'w') as output_f:
        mention_id_set = set()
        list_dict = {}
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            if line['mention_id'] in mention_id_set:
                m_dict = list_dict[line['mention_id']]
                cands = m_dict['candidates']
                cands.append(line['candidates'])
                m_dict['candidates'] = cands
                list_dict[line['mention_id']] = m_dict
            else:
                mention_id_set.add(line['mention_id'])
                m_dict = copy.deepcopy(line)
                m_dict['candidates'] = [line['candidates']]
                list_dict[line['mention_id']] = m_dict

        for k, v in list_dict.items():
            output_f.write(json.dumps(v, ensure_ascii=False) + '\n')

def add_nocands(raw_file, filter_file, output_file):
    with open(raw_file) as raw_f, \
        open(filter_file) as src_f, \
        open(output_file, 'w') as output_f:

        mention_id_set = set()
        for line in tqdm(src_f):
            line = json.loads(line.strip())
            mention_id_set.add(line['mention_id'])
            output_f.write(json.dumps(line, ensure_ascii=False) +'\n')
        
        for line in tqdm(raw_f):
            line = json.loads(line.strip())
            if line['mention_id'] in mention_id_set:
                continue
            line['candidates'] = []
            output_f.write(json.dumps(line, ensure_ascii=False) +'\n')

def result_decode(llm_pred:str, map_dict, first_or_last='first', have_id=True, id_dict=None):
    llm_pred = llm_pred.lstrip('<|assistant|>\n')
    llm_pred = llm_pred.lower()
    '''llm result decode
        1. decode id (if have)
        2. if fail, use the last entity name in sentence
    '''
    index_dict = {}
    if have_id:
        for ent_id, v in id_dict.items():
            # if v == 'none':
            #     continue
            k = ent_id.lower()
            if k in llm_pred:
                if first_or_last == 'first':
                    pos = llm_pred.find(k)
                elif first_or_last == 'last':    
                    pos = llm_pred.rfind(k)
                index_dict[ent_id] = pos
    if len(index_dict) > 0:
        if first_or_last == 'first':
            pred_id, _ = sorted(index_dict.items(), key = lambda kv:(kv[1], len(kv[0]), kv[0]), reverse=False)[0]
        elif first_or_last == 'last':
            pred_id, _ = sorted(index_dict.items(), key = lambda kv:(kv[1], len(kv[0]), kv[0]))[0]
        return id_dict[pred_id], pred_id

    # index_dict = {}
    llm_pred = llm_pred.replace(' ', '')
    for name in map_dict.keys():
        if name in llm_pred:
            if first_or_last == 'first':
                pos = llm_pred.find(name) - len(name)
            elif first_or_last == 'last':    
                pos = llm_pred.rfind(name) + len(name)
            
            index_dict[name] = pos

    if len(index_dict) > 0:
        if first_or_last == 'first':
            pred_entity, _ = sorted(index_dict.items(), key = lambda kv:(kv[1], len(kv[0]), kv[0]), reverse=False)[0]
        elif first_or_last == 'last':
            pred_entity, _ = sorted(index_dict.items(), key = lambda kv:(kv[1], len(kv[0]), kv[0]))[0]
    else:
        pred_entity = 'none'
    
    return pred_entity, map_dict[pred_entity]

# 增加一个新的参数 dataset_name
def line_f1(line:dict, test_field, dataset_name:str):
    if len(line.get(test_field, "")) == 0:
        return None, -1

    map_dict = {'none': '-1'}
    inverse_map_dict = {'-1': 'none'}
    for cand in line['candidates']:
        # 适配实体名称的键 ('name' for wiki-based, 'title' for zeshel)
        if dataset_name == 'zeshel':
            cand_name_key = 'title'
        else:
            cand_name_key = 'name'
        
        if cand_name_key not in cand:
            print(f"Warning: Candidate missing name key '{cand_name_key}'. Candidate data: {cand}")
            continue

        # 类型安全：确保name是字符串
        cand_name_value = cand[cand_name_key]
        if not isinstance(cand_name_value, str):
            print(f"Warning: Candidate name is not a string (type: {type(cand_name_value)}). Candidate data: {cand}")
            # 如果name是-1，说明是无效候选，跳过
            if cand_name_value == -1 or cand_name_value == '-1':
                continue
            # 尝试转换为字符串
            cand_name_value = str(cand_name_value)

        cand_name = cand_name_value.lower().replace(' ','')
        
        # 适配实体 ID 的键 ('wiki_id' for wiki-based, 'document_id' for zeshel)
        if dataset_name == 'zeshel':
            cand_id_key = 'document_id'
        else:
            cand_id_key = 'wiki_id' 
        
        if cand_id_key not in cand:
            print(f"Warning: Candidate missing ID key '{cand_id_key}'. Candidate data: {cand}")
            continue

        cand_id = cand[cand_id_key]

        map_dict[cand_name] = cand_id
        # 使用已确保为字符串的cand_name_value，而不是直接访问cand[cand_name_key]
        inverse_map_dict[cand_id] = cand_name_value
    
    pred_entity, pred_num = result_decode(line[test_field], map_dict, first_or_last='first', id_dict=inverse_map_dict)
    
    return inverse_map_dict.get(pred_num, None), pred_num

def merge_context_and_prior(context_file_name, prior_file_name, output_file_name, dataset_name:str):
    with open(context_file_name) as context_file, \
        open(prior_file_name) as prior_file, \
        open(output_file_name, 'w') as output_f:

        context_res_dict = {}
        context_response_dict = {}
        context_conf_dict = {}
        context_conf_raw_dict = {}
        for line in tqdm(context_file):
            line = json.loads(line.strip())
            # 确保 mention_id 存在，如果不存在，可能需要用其他唯一标识符
            mention_id = line.get('mention_id', line.get('id'))
            if mention_id is None: continue
            context_res_dict[mention_id] = line['context_pred_num']
            context_response_dict[mention_id] = line['llm_response']
            context_conf_dict[mention_id] = line.get('context_confidence', 0.0)
            context_conf_raw_dict[mention_id] = line.get('context_confidence_raw', 0.0)
        
        for line in tqdm(prior_file):
            line = json.loads(line.strip())
            mention_id = line.get('mention_id', line.get('id'))
            if mention_id is None or mention_id not in context_res_dict: continue

            prior_num = line['prior_pred_num']
            context_num = context_res_dict[mention_id]
            cands = line['candidates']
            new_cands = []

            # 适配实体 ID 的键
            if dataset_name == 'zeshel':
                cand_id_key = 'document_id'
            else:
                cand_id_key = 'wiki_id'

            for cand in cands:
                # 检查 cand 中是否有 id_key，避免再次出错
                if cand_id_key not in cand: continue
                
                if (cand[cand_id_key] == prior_num) or (cand[cand_id_key] == context_num):
                    new_cands.append(cand)
            # 如果两路预测都为 -1 或均不在候选中，避免清空候选，回退到保留原候选
            if len(new_cands) == 0:
                new_cands = cands
            
            line['candidates'] = new_cands
            line['prior_response'] = line['llm_response']
            line['context_pred_num'] = context_num
            line['context_response'] = context_response_dict[mention_id]
            line['context_confidence'] = context_conf_dict.get(mention_id, 0.0)
            line['context_confidence_raw'] = context_conf_raw_dict.get(mention_id, 0.0)
            # 兼容：在 prior 阶段也解析置信度
            if 'prior_confidence' not in line:
                line['prior_confidence'] = extract_confidence(line.get('llm_response', ''))
            if 'prior_confidence_raw' not in line:
                line['prior_confidence_raw'] = extract_confidence_raw_1to5(line.get('llm_response', ''))
            del line['llm_response']

            output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

def long_context_filter(input_file_name, output_file_name):
    with open(input_file_name) as input_f,\
        open(output_file_name, 'w') as output_f:
        for line in tqdm(input_f):
            line = json.loads(line.strip())
            if 'cut_left_context' in line.keys():
                continue
            elif len(line['llm_response']) > 0:
                continue
            else:
                cut_left, cut_right = cut_context(line["left_context"], line["right_context"])
                line['cut_left_context'] = cut_left
                line['cut_right_context'] = cut_right
                output_f.write(json.dumps(line, ensure_ascii=False) + '\n')

def concat_cot(src_file_name, input_file_name, output_file_name, zeshel=False):
    with open(src_file_name) as src_file, \
        open(input_file_name) as input_file, \
        open(output_file_name, 'w') as output_file:
        '''cot_index'''
        if zeshel:
            '''for zeshel as zeshel have mention id'''
            mention2cot = {}
            for line in tqdm(src_file):
                line = json.loads(line.strip())
                mention2cot[line['mention_id']] = line['cot_index']

            for line in tqdm(input_file):
                line = json.loads(line.strip())
                line['cot_index'] = mention2cot[line['mention_id']]
                output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            '''for other datasets'''
            for line, cot in tqdm(zip(input_file, src_file)):
                line = json.loads(line.strip())
                cot = json.loads(cot.strip())
                line['cot_index'] = cot['cand_index'][0]
                output_file.write(json.dumps(line, ensure_ascii=False) + '\n')

def add_summary(des:str):
    return des.split('.')[0] + '.'

def dataset_static(input_file_name, model):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    cand_num = []
    des_len = []
    context_len = []
    des_set = set()

    with open(input_file_name) as inpuf_f:
        for line in tqdm(inpuf_f):
            line = json.loads(line.strip())
            cands = line['candidates']
            cand_num.append(len(cands))
            context = line['left_context'] + ' ' + line['mention'] + ' ' + line['right_context']
            context_len.append(len(tokenizer(context)['input_ids']))
            for cand in cands:
                if cand['wiki_id'] in des_set:
                    continue
                des_len.append(len(tokenizer(str(cand['text']))['input_ids']))
                des_set.add(cand['wiki_id'])
    
    print(sum(cand_num) / len(cand_num))
    print(sum(des_len) / len(des_len))
    print(sum(context_len) / len(context_len))

if __name__ == '__main__':
    # input_file_path = 'EL_datasets/zeshel/documents'
    # file_list = os.listdir(input_file_path)
    # output_file_name = 'EL_datasets/zeshel/all.json'
    
    # concat_cands('mentions/test.json', 'tfidf_candidates/test.json', 'LLM_process/summary/summary.json', 'documents/all.json','mentions/listwise_raw.json')

    # select_hit_entity_in_all('tfidf_candidates/test.json', 'documents/all.json', 'documents/select.json')

    # split_file('mentions/', 'listwise_final_addsum', 3)

    # input_data_path = 'EL_datasets/zeshel/LLM_process/raw/text/'
    # input_file_list = []
    # for i in range(4):
    #     input_file_list.append(input_data_path + 'listwise_raw_cot_{}.json'.format(i))
    # output_file_name = input_data_path + 'listwise_raw_cot.json'
    # merge_file(input_file_list=input_file_list, output_file_name=output_file_name)

    # list2point('mentions/listwise_raw.json', 'mentions/pointwise_raw.json')

    # print(phrase_pointwise_ans(llm_res))

    # data_path = 'EL_datasets/zeshel/'
    # for i in range(4):
    #     point_wise_filter(data_path + 'LLM_process/pointwise_filter_fail/pointwise_filter_fail_{}.json'.format(i),
    #                       data_path + 'process_data/pointwise_filter_{}.json'.format(i),
    #                       data_path + 'process_data/pointwise_filter_{}_fail.json'.format(i))

    # process_file_line('datasets/zeshel/process_data/pointwise_raw.json',
    #                   'datasets/zeshel/process_data/pointwise_cut.json',
    #                   'cut_context')

    # point2list('EL_datasets/zeshel/mentions/pointwise_filter.json', 'EL_datasets/zeshel/mentions/listwise_filter.json')
    # add_nocands('mentions/listwise_raw.json', 'mentions/listwise_filter.json', 'mentions/listwise_filter_full.json')
    # sentence_embed('EL_datasets/zeshel/process_data/category.json', 'zeshel')
    # cot_select('EL_datasets/zeshel/process_data/category.json',
    #            'EL_datasets/embedding/zeshel.npy',
    #            'EL_datasets/zeshel/process_data/listwise_final.json')

    # merge_context_and_prior('EL_datasets/zeshel/process_data/context_id_first.json',
    #                         'EL_datasets/zeshel/process_data/prior_id_first.json',
    #                         'EL_datasets/zeshel/mentions/listwise_merge.json')

    # long_context_filter('EL_datasets/zeshel/LLM_process/prior_id.json',
    #                     'EL_datasets/zeshel/tmp/prior_long.json')

    # concat_cot('EL_datasets/embedding/wiki_test_prompt0_cot.jsonl',
    #            'EL_datasets/datasets_recall/listwise_input/wiki.jsonl',
    #            'datasets/wiki/listwise_filter.jsonl')

    # dataset_static('EL_datasets/datasets_recall/wiki_test_13B_top10g.jsonl',
    #                'LLMs/models/Meta-Llama-3-8B-Instruct')

    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description='处理Context和Prior结果并合并')
    parser.add_argument('--dataset', '-d', type=str, default='aida',
                       help='数据集名称 (默认: aida)')
    parser.add_argument('--use_true', '-t', action='store_true', default=True,
                       help='使用改进的过滤数据 (默认: True)')
    parser.add_argument('--no_true', action='store_true', default=False,
                       help='不使用改进的过滤数据')
    args = parser.parse_args()

    # 定义当前要处理的数据集名称
    CURRENT_DATASET = args.dataset
    USE_TRUE_VERSION = args.use_true and not args.no_true  # 是否使用改进的过滤数据

    # 根据版本设置后缀
    version_suffix = '_true' if USE_TRUE_VERSION else ''

    # Step 1: Process context results
    process_file_line(
        input_file_name=f'result/{CURRENT_DATASET}_context_zephyr{version_suffix}.json',
        output_file_name=f'result/{CURRENT_DATASET}_context_processed{version_suffix}.jsonl',
        func='context',
        dataset_name=CURRENT_DATASET # <-- 传入参数
    )

    # Step 2: Process prior results
    process_file_line(
        input_file_name=f'result/{CURRENT_DATASET}_prior_zephyr{version_suffix}.json',
        output_file_name=f'result/{CURRENT_DATASET}_prior_processed{version_suffix}.jsonl',
        func='prior',
        dataset_name=CURRENT_DATASET # <-- 传入参数
    )

    # Step 3: Merge them to create the input for the merge step
    merge_context_and_prior(
        context_file_name=f'result/{CURRENT_DATASET}_context_processed{version_suffix}.jsonl',
        prior_file_name=f'result/{CURRENT_DATASET}_prior_processed{version_suffix}.jsonl',
        output_file_name=f'datasets/{CURRENT_DATASET}/listwise_merge{version_suffix}.jsonl',
        dataset_name=CURRENT_DATASET # <-- 传入参数
    )