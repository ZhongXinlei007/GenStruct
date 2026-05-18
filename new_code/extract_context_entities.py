"""
从测试数据中提取文档级别的上下文实体（已消歧的实体）。

用法：
    python -m new_code.extract_context_entities \
        --input_file datasets/aida/listwise_merge_zephyr.jsonl \
        --output_file datasets/aida/listwise_merge_with_context_entities.jsonl \
        --dataset_name aida
"""
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Set


def _get_entity_id_key(dataset_name: str) -> str:
    """获取实体ID字段名"""
    return 'document_id' if dataset_name == 'zeshel' else 'wiki_id'


def extract_context_entities(
    input_file: str,
    output_file: str,
    dataset_name: str,
    doc_id_field: str = None
):
    """
    从测试数据中提取每个文档的已消歧实体作为上下文实体。
    
    策略：
    1. 按文档分组（使用 doc_id 或 context 的哈希）
    2. 对于每个文档，收集所有已消歧的实体（context_pred_num 或 prior_pred_num 不为 -1）
    3. 将上下文实体列表添加到每个样本中
    """
    entity_id_key = _get_entity_id_key(dataset_name)
    
    # 第一步：按文档分组，收集每个文档的已消歧实体
    doc_entities: Dict[str, Set[str]] = defaultdict(set)
    all_items: List[Dict] = []
    
    with open(input_file) as f:
        for line in tqdm(f, desc="Reading input file"):
            item = json.loads(line.strip())
            all_items.append(item)
            
            # 获取文档ID
            doc_id = None
            if doc_id_field and doc_id_field in item:
                doc_id = str(item[doc_id_field])
            else:
                # 尝试常见的文档ID字段
                doc_id = (item.get('doc_id') or 
                         item.get('document_id') or 
                         item.get('context_document_id'))
            
            if not doc_id:
                # 使用完整上下文的哈希作为文档ID
                full_context = (item.get('left_context', '') + ' ' + 
                               item.get('mention', '') + ' ' + 
                               item.get('right_context', ''))
                doc_id = str(hash(full_context))
            
            # 收集已消歧的实体（优先使用 context，其次 prior）
            context_pred = item.get('context_pred_num', '-1')
            prior_pred = item.get('prior_pred_num', '-1')
            
            # 只添加有效的实体ID（不为 -1 且在候选中）
            for pred_id in [context_pred, prior_pred]:
                if pred_id and str(pred_id) != '-1':
                    # 验证实体ID是否在候选中（可选，但更安全）
                    cand_ids = [str(c.get(entity_id_key)) for c in item.get('candidates', []) 
                               if entity_id_key in c]
                    if str(pred_id) in cand_ids or len(cand_ids) == 0:
                        doc_entities[doc_id].add(str(pred_id))
    
    print(f"Found {len(doc_entities)} documents")
    print(f"Average entities per document: {sum(len(v) for v in doc_entities.values()) / len(doc_entities) if doc_entities else 0:.2f}")
    
    # 第二步：为每个样本添加上下文实体（排除当前样本的实体）
    with open(output_file, 'w') as f:
        for item in tqdm(all_items, desc="Adding context entities"):
            # 获取文档ID（与上面相同的逻辑）
            doc_id = None
            if doc_id_field and doc_id_field in item:
                doc_id = str(item[doc_id_field])
            else:
                doc_id = (item.get('doc_id') or 
                         item.get('document_id') or 
                         item.get('context_document_id'))
            
            if not doc_id:
                full_context = (item.get('left_context', '') + ' ' + 
                               item.get('mention', '') + ' ' + 
                               item.get('right_context', ''))
                doc_id = str(hash(full_context))
            
            # 获取当前样本的实体ID（用于排除）
            current_entity_ids = set()
            context_pred = item.get('context_pred_num', '-1')
            prior_pred = item.get('prior_pred_num', '-1')
            for pred_id in [context_pred, prior_pred]:
                if pred_id and str(pred_id) != '-1':
                    current_entity_ids.add(str(pred_id))
            
            # 获取文档中其他已消歧的实体（排除当前样本）
            doc_entity_set = doc_entities.get(doc_id, set())
            context_entities = sorted(list(doc_entity_set - current_entity_ids))
            
            item['context_entities'] = context_entities
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved output to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract context entities from test data')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file (listwise_merge)')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSONL file with context_entities')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--doc_id_field', type=str, default=None, help='Field name for document ID (optional)')
    
    args = parser.parse_args()
    extract_context_entities(
        input_file=args.input_file,
        output_file=args.output_file,
        dataset_name=args.dataset_name.lower(),
        doc_id_field=args.doc_id_field
    )

