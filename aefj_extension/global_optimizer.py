import math
import re
from typing import Dict, List, Optional, Tuple

from new_code.cooccurrence import CooccurrenceScorer


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [tok for tok in re.split(r"[^A-Za-z0-9]+", text.lower()) if len(tok) > 3]


def _text_overlap(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class GlobalCoherenceOptimizer:
    """对整篇文档的实体选择进行迭代重排。"""

    def __init__(
        self,
        cooc_scorer: Optional[CooccurrenceScorer],
        compat_weight: float = 0.3,
        max_iter: int = 3,
        top_k: int = 3,
    ):
        self.cooc_scorer = cooc_scorer
        self.compat_weight = compat_weight
        self.max_iter = max_iter
        self.top_k = top_k

    def optimize(self, doc_groups: Dict[str, List[Dict]]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """返回 (final_selection_map, coherence_notes)"""
        selection: Dict[str, Dict] = {}
        notes: Dict[str, str] = {}

        for doc_id, mentions in doc_groups.items():
            doc_data = [m for m in mentions if m.get("aefj_ranked_candidates")]
            if len(doc_data) <= 1:
                # 单 mention 无需优化
                for mention in doc_data:
                    top = mention["aefj_ranked_candidates"][0]
                    selection[mention["_aefj_uid"]] = {
                        "candidate": top,
                        "bonus": 0.0,
                    }
                continue

            # 初始化为局部最佳
            current_choice: Dict[str, Dict] = {}
            for mention in doc_data:
                current_choice[mention["_aefj_uid"]] = mention["aefj_ranked_candidates"][0]

            for _ in range(self.max_iter):
                changed = False
                for mention in doc_data:
                    uid = mention["_aefj_uid"]
                    best_cand = current_choice[uid]
                    best_score = self._total_score(best_cand, uid, current_choice, doc_data)

                    for cand in mention["aefj_ranked_candidates"][: self.top_k]:
                        total = self._total_score(cand, uid, current_choice, doc_data)
                        if total > best_score + 1e-6:
                            best_score = total
                            best_cand = cand

                    if best_cand is not current_choice[uid]:
                        current_choice[uid] = best_cand
                        changed = True
                if not changed:
                    break

            for mention in doc_data:
                uid = mention["_aefj_uid"]
                final_cand = current_choice[uid]
                bonus, note = self._coherence_bonus_and_note(final_cand, uid, current_choice, doc_data)
                selection[uid] = {"candidate": final_cand, "bonus": bonus}
                if note:
                    notes[uid] = note

        return selection, notes

    def _total_score(self, cand: Dict, uid: str, current_choice: Dict[str, Dict], doc_data: List[Dict]) -> float:
        local = cand.get("score", 0.0)
        compat_sum = 0.0
        for other in doc_data:
            other_uid = other["_aefj_uid"]
            if other_uid == uid:
                continue
            other_cand = current_choice.get(other_uid)
            if not other_cand:
                continue
            compat_sum += self._compat(cand, other_cand)
        return local + self.compat_weight * compat_sum

    def _compat(self, cand_a: Dict, cand_b: Dict) -> float:
        pmi_score = 0.0
        if self.cooc_scorer:
            try:
                pmi_score = self.cooc_scorer.score_candidate(
                    cand_a.get("candidate_id"),
                    [cand_b.get("candidate_id")],
                    aggregate="mean",
                )
            except Exception:
                pmi_score = 0.0
        text_score = _text_overlap(cand_a.get("summary", ""), cand_b.get("summary", ""))
        # 两个分数范围差异较大，先进行缩放
        normalized_pmi = math.tanh(pmi_score)
        return 0.6 * normalized_pmi + 0.4 * text_score

    def _coherence_bonus_and_note(
        self,
        cand: Dict,
        uid: str,
        current_choice: Dict[str, Dict],
        doc_data: List[Dict],
    ) -> Tuple[float, str]:
        compat_pairs: List[str] = []
        compat_sum = 0.0
        for other in doc_data:
            other_uid = other["_aefj_uid"]
            if other_uid == uid:
                continue
            other_cand = current_choice.get(other_uid)
            if not other_cand:
                continue
            score = self._compat(cand, other_cand)
            if score > 0.0:
                compat_pairs.append(f"{other.get('mention')}→{other_cand.get('name')} ({score:.2f})")
            compat_sum += score
        bonus = self.compat_weight * compat_sum
        note = "; ".join(compat_pairs)
        return bonus, note


