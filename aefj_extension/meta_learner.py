import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from .llm_wrapper import LLMWrapper


def _truncate_text(text: str, max_len: int = 600) -> str:
    if not text:
        return ""
    text = " ".join(text.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@dataclass
class MetaLearnerConfig:
    default_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)
    temperature: float = 0.0
    max_new_tokens: int = 512


class AdaptiveWeightMetaLearner:
    """基于 LLM 的自适应权重估计器"""

    def __init__(self, llm: Optional[LLMWrapper], config: MetaLearnerConfig = MetaLearnerConfig()):
        self.llm = llm
        self.config = config

    def is_enabled(self) -> bool:
        return self.llm is not None and self.llm.is_available()

    def predict(self, payload: Dict) -> Tuple[Tuple[float, float, float], str]:
        """返回 (w_context, w_prior, w_cooc), rationale"""
        if not self.is_enabled():
            return self.config.default_weights, "LLM meta learner disabled; fallback to default weights."

        prompt = self._build_prompt(payload)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个实体链接融合器，需要阅读不同打分来源（上下文、先验、共现）的描述，"
                    "并输出三个权重，分别对应 Context、Prior、Co-occurrence。"
                    "请严格返回 JSON，格式为 {\"weights\":{\"context\":0.5,\"prior\":0.3,\"coocc\":0.2},"
                    "\"reason\":\"...\"}。权重需为非负实数，并可归一化。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.llm.call(
                messages,
                do_sample=self.config.temperature > 0,
                max_new_tokens=self.config.max_new_tokens,
            )
        except Exception as exc:
            return self.config.default_weights, f"Meta learner failed ({exc}); fallback to default."

        weights, rationale = self._parse_response(response)
        return weights, rationale

    def _build_prompt(self, payload: Dict) -> str:
        mention = payload.get("mention", "")
        context = _truncate_text(payload.get("context_snippet", ""))
        context_linker = payload.get("context_linker", {})
        prior_linker = payload.get("prior_linker", {})
        cooc_stats = payload.get("cooc_topk", [])
        candidates = payload.get("candidate_pool", [])[:5]

        lines: List[str] = []
        lines.append(f"Mention: {mention}")
        lines.append(f"Context snippet: {context or 'N/A'}\n")

        lines.append("Contextual linker view:")
        lines.append(
            f"- entity: {context_linker.get('name', 'unknown')} ({context_linker.get('entity_id', 'N/A')})"
        )
        lines.append(f"- confidence: {context_linker.get('confidence', 'N/A')}")
        context_reason = _truncate_text(context_linker.get("reason", ""))
        if context_reason:
            lines.append(f"- rationale: {context_reason}")

        lines.append("\nPrior linker view:")
        lines.append(f"- entity: {prior_linker.get('name', 'unknown')} ({prior_linker.get('entity_id', 'N/A')})")
        lines.append(f"- confidence: {prior_linker.get('confidence', 'N/A')}")
        prior_reason = _truncate_text(prior_linker.get("reason", ""))
        if prior_reason:
            lines.append(f"- rationale: {prior_reason}")

        lines.append("\nCo-occurrence signals (entity_id:score):")
        if cooc_stats:
            for cid, score, name in cooc_stats:
                lines.append(f"- {cid} ({name}): {score:.4f}")
        else:
            lines.append("- No reliable co-occurrence evidence.")

        lines.append("\nCandidate summaries:")
        for cand in candidates:
            summary = _truncate_text(cand.get("summary", ""))
            lines.append(f"- {cand.get('name', 'unknown')} ({cand.get('wiki_id', 'N/A')}): {summary}")

        lines.append(
            "\n请结合不同信息源的可靠性，输出 Context、Prior、Co-occurrence 三个权重并说明原因。"
            "如果某一来源缺失，请降低它的权重。权重不必总和为 1，程序会自动归一化。"
        )

        return "\n".join(lines)

    def _parse_response(self, response: str) -> Tuple[Tuple[float, float, float], str]:
        cleaned = response.strip()
        data = {}
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试截取 JSON 片段
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(cleaned[start : end + 1])
                except Exception:
                    return self.config.default_weights, f"Unparsable meta response: {cleaned}"
            else:
                return self.config.default_weights, f"Unparsable meta response: {cleaned}"

        weights = data.get("weights", {})
        rationale = data.get("reason") or data.get("explanation") or cleaned

        # --- [Fix Start] 安全转换函数，防止 dict 类型导致 float() 报错 ---
        def _safe_float(val: Any, default: float = 0.0) -> float:
            if val is None:
                return default
            if isinstance(val, (float, int)):
                return float(val)
            if isinstance(val, dict):
                # 遇到 LLM 输出嵌套字典的情况，直接返回默认值，防止崩溃
                # 可以选择打印日志：print(f"Warning: Unexpected dict in weights: {val}")
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        # --- [Fix End] ---

        # 使用安全函数获取权重
        wc = max(_safe_float(weights.get("context")), 0.0)
        wp = max(_safe_float(weights.get("prior")), 0.0)
        # 处理 coocc 可能的别名
        raw_cooc = weights.get("coocc", weights.get("cooc"))
        wco = max(_safe_float(raw_cooc), 0.0)

        total = wc + wp + wco
        if total == 0:
            return self.config.default_weights, rationale
        
        normalized = (wc / total, wp / total, wco / total)
        return normalized, rationale