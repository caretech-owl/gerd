"""
===========================================
        Module: Open-source LLM Setup
===========================================
"""

import logging

from langchain_community.llms import CTransformers

from gerd.models.model import ModelConfig

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def build_llm(model: ModelConfig) -> CTransformers:  # type: ignore[no-any-unimported]
    # Local CTransformers model
    llm = CTransformers(
        model=model.name,
        model_file=model.file,
        model_type=model.type,
        config={
            "max_new_tokens": model.max_new_tokens,
            "temperature": model.temperature,
            "context_length": model.context_length,
            "top_p": model.top_p,
            "top_k": model.top_k,
            "repetition_penalty": model.repetition_penalty,
            "last_n_tokens": model.last_n_tokens,
        },
    )

    return llm
