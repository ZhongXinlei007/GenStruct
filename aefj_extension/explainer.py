from typing import Dict, Optional

from .llm_wrapper import LLMWrapper


def _short(text, length: int = 240) -> str:
    if not text:
        return ""

    # Handle non-string inputs (e.g., dict, list, etc.)
    if not isinstance(text, str):
        if isinstance(text, dict):
            # For dictionaries, try to get a meaningful string representation
            text = str(text.get("reason", "")) if "reason" in text else str(text)
        else:
            text = str(text)

    text = " ".join(text.strip().split())
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


class DecisionReporter:
    """根据融合得分与全局一致性结果生成可解释文本。"""

    def __init__(self, llm: Optional[LLMWrapper], fallback_template: bool = True):
        self.llm = llm
        self.fallback_template = fallback_template

    def generate(
        self,
        record: Dict,
        final_candidate_id: str,
        final_candidate_name: Optional[str],
        coherence_note: Optional[str],
        coherence_bonus: float,
    ) -> str:
        payload = self._collect_facts(record, final_candidate_id, final_candidate_name, coherence_note, coherence_bonus)

        if self.llm and self.llm.is_available():
            try:
                prompt = self._build_prompt(payload)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "你是一个智能判决解释器，需要将多源融合结果总结成简洁的中文说明。"
                            "请突出最终实体、各信息源贡献和全局一致性。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                return self.llm.call(messages, do_sample=False, max_new_tokens=400)
            except Exception:
                pass

        if self.fallback_template:
            return self._template_report(payload)
        return ""

    def _collect_facts(
        self,
        record: Dict,
        final_candidate_id: str,
        final_candidate_name: Optional[str],
        coherence_note: Optional[str],
        coherence_bonus: float,
    ) -> Dict:
        weights = record.get("aefj_weights", {})
        w_context = weights.get("context", 0.0)
        w_prior = weights.get("prior", 0.0)
        w_cooc = weights.get("cooc", weights.get("coocc", 0.0))

        context_support = float(record.get("context_confidence", 0.0)) if str(record.get("context_pred_num")) == str(final_candidate_id) else 0.0
        prior_support = float(record.get("prior_confidence", 0.0)) if str(record.get("prior_pred_num")) == str(final_candidate_id) else 0.0
        cooc_scores = record.get("aefj_cooc_scores", {})
        cooc_support = float(cooc_scores.get(str(final_candidate_id), 0.0))

        return {
            "entity_name": final_candidate_name or "未知",
            "entity_id": final_candidate_id,
            "weights": (w_context, w_prior, w_cooc),
            "supports": (context_support, prior_support, cooc_support),
            "context_reason": _short(record.get("context_response", "")),
            "prior_reason": _short(record.get("prior_response", "")),
            "coherence_note": coherence_note,
            "coherence_bonus": coherence_bonus,
            "meta_reason": _short(record.get("aefj_meta_reason", ""), 400),
        }

    def _build_prompt(self, payload: Dict) -> str:
        w_context, w_prior, w_cooc = payload["weights"]
        s_context, s_prior, s_cooc = payload["supports"]
        lines = [
            f"最终实体：{payload['entity_name']} (ID: {payload['entity_id']})",
            f"自适应权重：Context {w_context:.2f}, Prior {w_prior:.2f}, Co-occurrence {w_cooc:.2f}",
            f"对应贡献：Context {w_context * s_context:.3f}, Prior {w_prior * s_prior:.3f}, Co-occurrence {w_cooc * s_cooc:.3f}",
            f"Coherence bonus: {payload['coherence_bonus']:.3f}",
            f"Context 理由：{payload['context_reason'] or '无'}",
            f"Prior 理由：{payload['prior_reason'] or '无'}",
            f"全局一致性：{payload['coherence_note'] or '无'}",
            f"元学习器提示：{payload['meta_reason'] or '无'}",
            "请将这些要点整合为一段说明，强调为什么最终实体最合理，并确保输出中文。",
        ]
        return "\n".join(lines)

    def _template_report(self, payload: Dict) -> str:
        w_context, w_prior, w_cooc = payload["weights"]
        s_context, s_prior, s_cooc = payload["supports"]
        lines = []
        lines.append(f"最终链接实体：{payload['entity_name']}（ID: {payload['entity_id']}）。")
        lines.append(
            f"自适应权重 -> Context: {w_context:.2f}, Prior: {w_prior:.2f}, Co-occurrence: {w_cooc:.2f}。"
        )
        lines.append(
            f"贡献得分 -> Context: {w_context * s_context:.3f}, Prior: {w_prior * s_prior:.3f}, "
            f"Co-occurrence: {w_cooc * s_cooc:.3f}, Coherence bonus: {payload['coherence_bonus']:.3f}。"
        )
        if payload["context_reason"]:
            lines.append(f"Context 证据：{payload['context_reason']}")
        if payload["prior_reason"]:
            lines.append(f"Prior 证据：{payload['prior_reason']}")
        if payload["coherence_note"]:
            lines.append(f"全局一致性：{payload['coherence_note']}")
        if payload["meta_reason"]:
            lines.append(f"元学习器提示：{payload['meta_reason']}")
        return "\n".join(lines)



