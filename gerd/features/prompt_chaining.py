"""The prompt chaining extension"""

import logging

from pydantic import BaseModel

from gerd.backends.loader import LLM
from gerd.models.model import PromptConfig

_LOGGER = logging.getLogger(__name__)


class PromptChainingConfig(BaseModel):
    """Configuration for"""

    prompts: list[PromptConfig]


class PromptChaining:
    def __init__(
        self, config: PromptChainingConfig, llm: LLM, prompt: PromptConfig
    ) -> None:
        self.llm = llm
        self.config = config
        self.prompt = prompt

    def generate(self, parameters: dict[str, str]) -> str:
        for i, prompt in enumerate(self.config.prompts, 1):
            resolved = self.prompt.format({"prompt": prompt.format(parameters)})
            _LOGGER.debug("\n===== Input =====\n\n%s\n\n====================", resolved)
            res = self.llm.generate(resolved).strip()
            _LOGGER.debug("\n===== Response =====\n\n%s\n\n====================", res)
            parameters[f"response_{i}"] = res
        return res
