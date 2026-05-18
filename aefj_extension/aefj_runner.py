import argparse
import json
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from new_code.cooccurrence import CooccurrenceScorer

from .explainer import DecisionReporter
from .global_optimizer import GlobalCoherenceOptimizer
from .llm_wrapper import build_llm_caller, LLMWrapper
from .meta_learner import AdaptiveWeightMetaLearner, MetaLearnerConfig


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


class AdaptiveExplainableFusionRunner:
    def __init__(
        self,
        dataset_name: str,
        doc_field: Optional[str],
        context_entities_field: str,
        cooc_scorer: Optional[CooccurrenceScorer],
        meta_learner: AdaptiveWeightMetaLearner,
        reporter: DecisionReporter,
        optimizer: Optional[GlobalCoherenceOptimizer],
        top_k: int = 3,
    ):
        self.dataset_name = dataset_name.lower()
        self.doc_field = doc_field
        self.context_entities_field = context_entities_field
        self.cooc_scorer = cooc_scorer
        self.meta_learner = meta_learner
        self.reporter = reporter
        self.optimizer = optimizer
        self.top_k = max(1, top_k)
        self._uid_counter = 0

    def run(self, input_path: str, output_path: str):
        records: List[Dict] = []
        doc_groups: Dict[str, List[Dict]] = {}

        with open(input_path) as input_f:
            for line in tqdm(input_f, desc="AEFJ-Stage1"):
                item = json.loads(line.strip())
                processed = self._process_item(item)
                records.append(processed)
                doc_key = processed["_doc_key"]
                doc_groups.setdefault(doc_key, []).append(processed)

        selection_map, coherence_notes = {}, {}
        if self.optimizer:
            selection_map, coherence_notes = self.optimizer.optimize(doc_groups)

        with open(output_path, "w") as output_f:
            for record in tqdm(records, desc="AEFJ-Stage2"):
                uid = record["_aefj_uid"]
                final_candidate = None
                bonus = 0.0
                if selection_map and uid in selection_map:
                    final_candidate = selection_map[uid]["candidate"]
                    bonus = selection_map[uid]["bonus"]
                elif record.get("aefj_ranked_candidates"):
                    final_candidate = record["aefj_ranked_candidates"][0]
                elif record.get("aefj_fallback"):
                    final_candidate = record["aefj_fallback"]

                if final_candidate:
                    record["final_pred_num"] = final_candidate.get("candidate_id")
                    record["final_pred_entity"] = final_candidate.get("name")
                    record["global_coherence_bonus"] = bonus
                    record["global_coherence_note"] = coherence_notes.get(uid, "")
                    explanation = self.reporter.generate(
                        record,
                        final_candidate.get("candidate_id"),
                        final_candidate.get("name"),
                        record.get("global_coherence_note", ""),
                        bonus,
                    )
                    record["aefj_explanation"] = explanation
                else:
                    record["aefj_explanation"] = "No candidate available."

                output_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----- internal helpers -----

    def _process_item(self, item: Dict) -> Dict:
        cand_id_key = "document_id" if self.dataset_name == "zeshel" else "wiki_id"
        name_key = "title" if self.dataset_name == "zeshel" else "name"

        candidates = item.get("candidates", [])
        cand_lookup: Dict[str, Dict] = {}
        for cand in candidates:
            cid = str(cand.get(cand_id_key, cand.get("cand_wiki_id", "")))
            if not cid:
                continue
            cand_lookup[cid] = {
                "candidate_id": cid,
                "name": cand.get(name_key) or cand.get("cand_name"),
                "summary": cand.get("summary") or cand.get("text") or cand.get("cand_summary", ""),
            }

        context_entities = [str(e) for e in item.get(self.context_entities_field, [])]
        cooc_scores = {}
        if cand_lookup and self.cooc_scorer:
            cooc_scores = self.cooc_scorer.score_all(list(cand_lookup.keys()), context_entities)

        meta_payload = self._build_meta_payload(item, cand_lookup, cooc_scores)
        weights, rationale = self.meta_learner.predict(meta_payload)

        context_pred_id = str(item.get("context_pred_num", "-1"))
        prior_pred_id = str(item.get("prior_pred_num", "-1"))
        context_conf = float(item.get("context_confidence", 0.0) or 0.0)
        prior_conf = float(item.get("prior_confidence", 0.0) or 0.0)

        per_cand_scores: Dict[str, float] = {}
        ranked_candidates: List[Dict] = []

        for cid, meta in cand_lookup.items():
            score_context = context_conf if cid == context_pred_id else 0.0
            score_prior = prior_conf if cid == prior_pred_id else 0.0
            score_cooc = cooc_scores.get(cid, 0.0)
            final_score = (
                weights[0] * score_context + weights[1] * score_prior + weights[2] * score_cooc
            )
            per_cand_scores[cid] = final_score
            ranked_candidates.append(
                {
                    "candidate_id": cid,
                    "name": meta.get("name"),
                    "summary": meta.get("summary"),
                    "score": final_score,
                    "cooc": score_cooc,
                }
            )

        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)
        if self.top_k and len(ranked_candidates) > self.top_k:
            ranked_candidates = ranked_candidates[: self.top_k]

        fallback_candidate = None
        if not ranked_candidates:
            fallback_candidate = self._fallback_candidate(item, cand_lookup, name_key)

        item["aefj_weights"] = {
            "context": weights[0],
            "prior": weights[1],
            "cooc": weights[2],
        }
        item["aefj_meta_reason"] = rationale
        item["aefj_scores"] = per_cand_scores
        item["aefj_ranked_candidates"] = ranked_candidates
        item["aefj_cooc_scores"] = cooc_scores
        item["aefj_fallback"] = fallback_candidate

        uid = self._allocate_uid(item)
        item["_aefj_uid"] = uid
        item["_doc_key"] = self._extract_doc_key(item)

        return item

    def _build_meta_payload(self, item: Dict, cand_lookup: Dict[str, Dict], cooc_scores: Dict[str, float]) -> Dict:
        context = (
            item.get("left_context", "")
            + " ###"
            + item.get("mention", "")
            + "### "
            + item.get("right_context", "")
        )
        context = _normalize_text(context)

        def resolve_name(cid: str, default: Optional[str]) -> Optional[str]:
            if cid in cand_lookup:
                return cand_lookup[cid].get("name")
            return default

        context_linker = {
            "entity_id": item.get("context_pred_num"),
            "name": resolve_name(str(item.get("context_pred_num")), item.get("context_pred_entity")),
            "confidence": item.get("context_confidence"),
            "reason": item.get("context_response"),
        }
        prior_linker = {
            "entity_id": item.get("prior_pred_num"),
            "name": resolve_name(str(item.get("prior_pred_num")), item.get("prior_pred_entity")),
            "confidence": item.get("prior_confidence"),
            "reason": item.get("prior_response"),
        }

        cooc_topk = []
        for cid, score in sorted(cooc_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            cooc_topk.append((cid, score, cand_lookup.get(cid, {}).get("name", "unknown")))

        candidate_pool = []
        for cid, meta in list(cand_lookup.items())[:5]:
            candidate_pool.append(
                {
                    "wiki_id": cid,
                    "name": meta.get("name"),
                    "summary": meta.get("summary"),
                }
            )

        payload = {
            "mention": item.get("mention"),
            "context_snippet": context,
            "context_linker": context_linker,
            "prior_linker": prior_linker,
            "cooc_topk": cooc_topk,
            "candidate_pool": candidate_pool,
        }
        return payload

    def _fallback_candidate(self, item: Dict, cand_lookup: Dict[str, Dict], name_key: str) -> Optional[Dict]:
        context_id = str(item.get("context_pred_num", "-1"))
        prior_id = str(item.get("prior_pred_num", "-1"))

        if context_id in cand_lookup:
            meta = cand_lookup[context_id]
            return {"candidate_id": context_id, "name": meta.get("name"), "summary": meta.get("summary"), "score": 0.0}
        if prior_id in cand_lookup:
            meta = cand_lookup[prior_id]
            return {"candidate_id": prior_id, "name": meta.get("name"), "summary": meta.get("summary"), "score": 0.0}

        if item.get("candidates"):
            first = item["candidates"][0]
            cid = str(first.get("wiki_id") or first.get("cand_wiki_id") or first.get("document_id"))
            if cid:
                return {
                    "candidate_id": cid,
                    "name": first.get(name_key),
                    "summary": first.get("summary"),
                    "score": 0.0,
                }
        return None

    def _allocate_uid(self, item: Dict) -> str:
        if "global_id" in item:
            return str(item["global_id"])
        if "id" in item:
            return f"{item.get('id')}"
        self._uid_counter += 1
        return f"auto_{self._uid_counter}"

    def _extract_doc_key(self, item: Dict) -> str:
        candidates = []
        if self.doc_field:
            candidates.append(self.doc_field)
        candidates.extend(
            ["doc_name", "docid", "document_id", "doc_title", "source_doc", "doc_key", "paragraph_id"]
        )
        for key in candidates:
            if key and key in item:
                return str(item[key])
        # 退化到每个 mention 自成文档
        return f"__docless__{item.get('_aefj_uid')}"


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive & Explainable Fusion Judger (AEFJ)")
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--meta_model_name", type=str, help="用于元学习器的模型名称")
    parser.add_argument("--meta_model_path", type=str, help="用于元学习器的模型路径")
    parser.add_argument("--report_model_name", type=str, help="用于解释生成的模型名称")
    parser.add_argument("--report_model_path", type=str, help="用于解释生成的模型路径")
    parser.add_argument("--share_model", action="store_true", help="解释与元学习共享同一模型")
    parser.add_argument("--cooc_pmi_path", type=str, default=None)
    parser.add_argument("--doc_field", type=str, default="doc_id")
    parser.add_argument("--context_entities_field", type=str, default="context_entities")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--coherence_weight", type=float, default=0.3)
    parser.add_argument("--coherence_iter", type=int, default=3)
    parser.add_argument("--enable_global", action="store_true")
    parser.add_argument("--disable_meta", action="store_true")
    parser.add_argument("--meta_default_weights", type=str, default="0.5,0.3,0.2")
    return parser.parse_args()


def build_runner_from_args(args) -> AdaptiveExplainableFusionRunner:
    cooc_scorer = CooccurrenceScorer(args.cooc_pmi_path) if args.cooc_pmi_path else None

    meta_llm: Optional[LLMWrapper] = None
    report_llm: Optional[LLMWrapper] = None
    if not args.disable_meta:
        meta_llm = build_llm_caller(args.meta_model_name, args.meta_model_path)

    if args.share_model:
        report_llm = meta_llm
    else:
        report_llm = build_llm_caller(args.report_model_name, args.report_model_path)

    defaults = tuple(float(x) for x in args.meta_default_weights.split(","))
    meta_config = MetaLearnerConfig(default_weights=defaults)
    meta_learner = AdaptiveWeightMetaLearner(meta_llm, meta_config)
    reporter = DecisionReporter(llm=report_llm, fallback_template=True)

    optimizer = None
    if args.enable_global:
        optimizer = GlobalCoherenceOptimizer(
            cooc_scorer=cooc_scorer,
            compat_weight=args.coherence_weight,
            max_iter=args.coherence_iter,
            top_k=args.top_k,
        )

    runner = AdaptiveExplainableFusionRunner(
        dataset_name=args.dataset_name,
        doc_field=args.doc_field,
        context_entities_field=args.context_entities_field,
        cooc_scorer=cooc_scorer,
        meta_learner=meta_learner,
        reporter=reporter,
        optimizer=optimizer,
        top_k=args.top_k,
    )
    return runner


if __name__ == "__main__":
    args = parse_args()
    runner = build_runner_from_args(args)
    runner.run(args.input_path, args.output_path)


