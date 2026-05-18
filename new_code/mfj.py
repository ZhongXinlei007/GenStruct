import json
import argparse
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

from .cooccurrence import CooccurrenceScorer


def _get_candidate_id_key(dataset_name: str) -> str:
    return 'document_id' if dataset_name == 'zeshel' else 'wiki_id'


def _normalize_weight(v: float) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _final_score(
    cand_id: str,
    context_pred_id: str,
    prior_pred_id: str,
    context_conf: float,
    prior_conf: float,
    cooc_score: float,
    w1: float,
    w2: float,
    w3: float,
) -> float:
    score_context = context_conf if str(cand_id) == str(context_pred_id) else 0.0
    score_prior = prior_conf if str(cand_id) == str(prior_pred_id) else 0.0
    return w1 * score_context + w2 * score_prior + w3 * cooc_score


def mfj_decide(
    input_file_name: str,
    output_file_name: str,
    dataset_name: str,
    w1: float,
    w2: float,
    w3: float,
    cooc_pmi_path: str = None,
    context_entities_field: str = 'context_entities'
):
    scorer = CooccurrenceScorer(cooc_pmi_path)
    cand_id_key = _get_candidate_id_key(dataset_name)

    with open(input_file_name) as input_f, open(output_file_name, 'w') as output_f:
        for line in tqdm(input_f):
            item = json.loads(line.strip())

            candidates: List[Dict[str, Any]] = item.get('candidates', [])
            # 无候选时也给出预测（回退到 context/prior）

            context_pred_id = item.get('context_pred_num', '-1')
            prior_pred_id = item.get('prior_pred_num', '-1')
            context_conf = float(item.get('context_confidence', 0.0))
            prior_conf = float(item.get('prior_confidence', 0.0))

            # 获取文档上下文实体（已消歧）ID 列表（若没有则为空）
            context_entity_ids: List[str] = [str(e) for e in item.get(context_entities_field, [])]

            cand_ids: List[str] = []
            for c in candidates:
                if cand_id_key in c:
                    cand_ids.append(str(c[cand_id_key]))

            # 共现打分
            cooc_scores: Dict[str, float] = scorer.score_all(cand_ids, context_entity_ids) if cand_ids else {}

            # 融合打分并选最大
            w1v, w2v, w3v = _normalize_weight(w1), _normalize_weight(w2), _normalize_weight(w3)
            best_id, best_score = None, float('-inf')
            per_cand_scores: Dict[str, float] = {}
            if cand_ids:
                for cid in cand_ids:
                    s = _final_score(cid, context_pred_id, prior_pred_id, context_conf, prior_conf, cooc_scores.get(cid, 0.0), w1v, w2v, w3v)
                    per_cand_scores[cid] = s
                    if s > best_score:
                        best_id, best_score = cid, s
            else:
                # 回退策略：无候选时优先 context，其次 prior，否则 -1
                if str(context_pred_id).strip() != '-1':
                    best_id = str(context_pred_id)
                elif str(prior_pred_id).strip() != '-1':
                    best_id = str(prior_pred_id)
                else:
                    best_id = '-1'

            # 写回结果
            item['mfj_scores'] = per_cand_scores
            item['mfj_weights'] = {'w1': w1v, 'w2': w2v, 'w3': w3v}
            item['final_pred_num'] = best_id
            # 取实体名
            name_key = 'title' if dataset_name == 'zeshel' else 'name'
            final_name = None
            for c in candidates:
                if str(c.get(cand_id_key)) == str(best_id):
                    final_name = c.get(name_key)
                    break
            item['final_pred_entity'] = final_name

            output_f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-source Fusion Judger (MFJ)')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--w1', type=float, default=1.0, help='weight for Contextual score')
    parser.add_argument('--w2', type=float, default=1.0, help='weight for Prior score')
    parser.add_argument('--w3', type=float, default=0.0, help='weight for Co-occurrence score')
    parser.add_argument('--cooc_pmi_path', type=str, default=None, help='path to PMI json file')
    parser.add_argument('--context_entities_field', type=str, default='context_entities')

    args = parser.parse_args()
    mfj_decide(
        input_file_name=args.input_path,
        output_file_name=args.output_path,
        dataset_name=args.dataset_name.lower(),
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        cooc_pmi_path=args.cooc_pmi_path,
        context_entities_field=args.context_entities_field,
    )


