"""
===========================================
        Module: Open-source LLM Setup
===========================================
"""
from langchain.llms import CTransformers

from team_red.config import CONFIG


def build_llm() -> CTransformers:  # type: ignore[no-any-unimported]
    # Local CTransformers model
    llm = CTransformers(
        model=CONFIG.model.name,
        model_file=CONFIG.model.file,
        model_type=CONFIG.model.type,
        config={
            "max_new_tokens": CONFIG.model.max_new_tokens,
            "temperature": CONFIG.model.max_new_tokens,
        },
    )

    return llm
