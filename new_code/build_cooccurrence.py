"""
从训练集构建实体共现关系（PMI）的工具。

用法：
    python -m new_code.build_cooccurrence \
        --train_file datasets/aida/aida_train.jsonl \
        --output_file cooccurrence/aida_pmi.json \
        --dataset_name aida \
        --min_cooccurrence 2
"""
import json
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
import math
from typing import Dict, List, Set


def _get_entity_id_key(dataset_name: str) -> str:
    """获取实体ID字段名"""
    return 'document_id' if dataset_name == 'zeshel' else 'wiki_id'


def _get_entity_name_key(dataset_name: str) -> str:
    """获取实体名称字段名"""
    return 'title' if dataset_name == 'zeshel' else 'name'


def extract_entities_from_train(train_file: str, dataset_name: str) -> Dict[str, List[Set[str]]]:
    """
    从训练集中提取每个文档的实体集合。
    返回: {doc_id: [set of entity_ids]}
    注意：如果数据中没有 doc_id，使用 context 的哈希作为文档标识。
    """
    doc_entities: Dict[str, Set[str]] = defaultdict(set)
    entity_id_key = _get_entity_id_key(dataset_name)
    
    with open(train_file) as f:
        for line in tqdm(f, desc="Reading training data"):
            item = json.loads(line.strip())
            
            # 获取文档ID（如果有的话）
            doc_id = item.get('doc_id') or item.get('document_id') or item.get('context_document_id')
            if not doc_id:
                # 使用完整上下文的哈希作为文档ID
                full_context = (item.get('left_context', '') + ' ' + 
                               item.get('mention', '') + ' ' + 
                               item.get('right_context', ''))
                doc_id = str(hash(full_context))
            
            # 获取答案实体ID或名称
            ans_id = (item.get('ans_id') or 
                     item.get('label_document_id') or 
                     item.get('wiki_id'))
            
            if not ans_id:
                # 尝试从 candidates 中匹配 output 字段
                output_name = item.get('output', '').strip()
                if output_name:
                    for cand in item.get('candidates', []):
                        cand_name = cand.get(_get_entity_name_key(dataset_name), '').strip()
                        if cand_name == output_name:
                            ans_id = cand.get(entity_id_key)
                            break
            
            # 如果还是没有ID，使用实体名称作为标识符（归一化处理）
            if not ans_id:
                output_name = item.get('output', '').strip()
                if output_name:
                    # 使用归一化的实体名称作为ID（小写，去除多余空格）
                    ans_id = 'NAME:' + output_name.lower().strip()
            
            if ans_id:
                doc_entities[doc_id].add(str(ans_id))
    
    # 转换为列表格式（每个文档一个实体集合）
    result: Dict[str, List[Set[str]]] = defaultdict(list)
    for doc_id, entities in doc_entities.items():
        if len(entities) > 1:  # 只保留有多个实体的文档
            result[doc_id] = [entities]
    
    return result


def compute_pmi(doc_entities: Dict[str, List[Set[str]]], min_cooccurrence: int = 2) -> Dict[str, Dict[str, float]]:
    """
    计算实体对的 PMI (Pointwise Mutual Information)。
    
    PMI(e1, e2) = log(P(e1, e2) / (P(e1) * P(e2)))
                = log(count(e1, e2) * N / (count(e1) * count(e2)))
    
    返回: {entity_id: {other_entity_id: pmi_value, ...}, ...}
    """
    # 统计实体出现次数和共现次数
    entity_counts: Counter = Counter()
    cooccurrence_counts: Dict[str, Counter] = defaultdict(Counter)
    total_docs = 0
    
    for doc_id, entity_sets in tqdm(doc_entities.items(), desc="Counting co-occurrences"):
        for entity_set in entity_sets:
            total_docs += 1
            entity_list = list(entity_set)
            # 更新实体计数
            for eid in entity_list:
                entity_counts[eid] += 1
            # 更新共现计数（无序对）
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i+1:]:
                    cooccurrence_counts[e1][e2] += 1
                    cooccurrence_counts[e2][e1] += 1
    
    if total_docs == 0:
        return {}
    
    # 计算 PMI
    pmi_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    for e1, cooc_counter in tqdm(cooccurrence_counts.items(), desc="Computing PMI"):
        count_e1 = entity_counts[e1]
        if count_e1 < min_cooccurrence:
            continue
        
        for e2, count_both in cooc_counter.items():
            if count_both < min_cooccurrence:
                continue
            
            count_e2 = entity_counts[e2]
            if count_e2 < min_cooccurrence:
                continue
            
            # PMI = log(P(e1,e2) / (P(e1) * P(e2)))
            #     = log((count_both / N) / ((count_e1 / N) * (count_e2 / N)))
            #     = log(count_both * N / (count_e1 * count_e2))
            if count_e1 > 0 and count_e2 > 0:
                pmi = math.log((count_both * total_docs) / (count_e1 * count_e2))
                # 只保留正PMI（表示正相关）
                if pmi > 0:
                    pmi_dict[str(e1)][str(e2)] = float(pmi)
    
    return dict(pmi_dict)


def build_cooccurrence_from_train(
    train_file: str,
    output_file: str,
    dataset_name: str,
    min_cooccurrence: int = 2
):
    """主函数：从训练集构建共现关系"""
    print(f"Extracting entities from training file: {train_file}")
    doc_entities = extract_entities_from_train(train_file, dataset_name)
    
    print(f"Found {len(doc_entities)} documents with multiple entities")
    print(f"Computing PMI with min_cooccurrence={min_cooccurrence}")
    pmi_dict = compute_pmi(doc_entities, min_cooccurrence=min_cooccurrence)
    
    print(f"Computed PMI for {len(pmi_dict)} entities")
    total_pairs = sum(len(v) for v in pmi_dict.values())
    print(f"Total entity pairs: {total_pairs}")
    
    # 保存结果
    import os
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(pmi_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Saved PMI dictionary to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build co-occurrence PMI from training data')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Output PMI JSON file path')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (aida, ace2004, etc.)')
    parser.add_argument('--min_cooccurrence', type=int, default=2, help='Minimum co-occurrence count (default: 2)')
    
    args = parser.parse_args()
    build_cooccurrence_from_train(
        train_file=args.train_file,
        output_file=args.output_file,
        dataset_name=args.dataset_name.lower(),
        min_cooccurrence=args.min_cooccurrence
    )

