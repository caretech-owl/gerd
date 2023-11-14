"""
===========================================
        Module: Open-source LLM Setup
===========================================
"""
import logging

from langchain.llms import CTransformers

from team_red.models.model import ModelConfig

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
        },
    )

    return llm
