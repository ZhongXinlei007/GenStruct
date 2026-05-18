import json
from typing import Dict, List, Optional


class CooccurrenceScorer:
    """PMI 共现打分器。
    - 期望加载一个 JSON 文件：{"entity_id":{"other_entity_id":PMI_value,...},...}
    - 如果资源缺失或未命中，返回 0.0。
    """

    def __init__(self, pmi_path: Optional[str] = None):
        self.pmi: Dict[str, Dict[str, float]] = {}
        if pmi_path:
            try:
                with open(pmi_path) as f:
                    self.pmi = json.load(f)
            except Exception:
                # 资源不可用时，保持为空
                self.pmi = {}

    def score_candidate(self, candidate_id: str, context_entity_ids: List[str], aggregate: str = "mean") -> float:
        if not self.pmi or not context_entity_ids:
            return 0.0
        scores = []
        cand_table = self.pmi.get(str(candidate_id), {})
        for eid in context_entity_ids:
            if eid == candidate_id:
                continue
            val = cand_table.get(str(eid))
            if isinstance(val, (int, float)):
                scores.append(float(val))
        if not scores:
            return 0.0
        if aggregate == "sum":
            return float(sum(scores))
        # 缺省使用 mean
        return float(sum(scores) / len(scores))

    def score_all(self, candidate_ids: List[str], context_entity_ids: List[str], aggregate: str = "mean") -> Dict[str, float]:
        result: Dict[str, float] = {}
        for cid in candidate_ids:
            result[cid] = self.score_candidate(cid, context_entity_ids, aggregate=aggregate)
        return result


