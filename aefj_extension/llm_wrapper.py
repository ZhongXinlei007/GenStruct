from typing import Optional

from new_code.LLM_calls import load_llm, llm_call


class LLMWrapper:
    """统一封装 LLM 的加载与调用，便于在多个模块间共享。"""

    def __init__(self, model_name: Optional[str], model_path: Optional[str]):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        if not model_name or not model_path:
            # 允许禁用 LLM
            return

        resource = load_llm(model_name, model_path)
        if isinstance(resource, tuple):
            # (model, tokenizer)
            self.model, self.tokenizer = resource
        else:
            self.pipeline = resource

    def is_available(self) -> bool:
        return self.model_name is not None and (
            self.model is not None or self.pipeline is not None or self.tokenizer is not None
        )

    def call(self, messages, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("LLMWrapper 未初始化，无法发起调用。")
        return llm_call(
            messages,
            self.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            pipeline=self.pipeline,
            **kwargs,
        )


def build_llm_caller(model_name: Optional[str], model_path: Optional[str]) -> Optional[LLMWrapper]:
    if not model_name or not model_path:
        return None
    return LLMWrapper(model_name=model_name, model_path=model_path)


