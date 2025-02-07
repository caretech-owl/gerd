"""The prompt chaining extension.

Prompt chaining is a method to improve the factual accuracy of the model's output.
To do this, the model generates a series of prompts and uses the output of
each prompt as the input for the next prompt. This allows the model to
reflect on its own output and generate a more coherent response.
"""

import logging

from pydantic import BaseModel

from gerd.loader import LLM
from gerd.models.model import PromptConfig

_LOGGER = logging.getLogger(__name__)


class PromptChainingConfig(BaseModel):
    """Configuration for prompt chaining.

    Note that prompts should contain placeholders for the responses to be inserted.
    The initial question can be used with `{question}` if it is passed as a
    parameter with this key. The responses will be indexed as `response_1`,
    `response_2`, etc. This way any prompt can refer to all previous responses
    and the initial question if needed.
    """

    prompts: list[PromptConfig]
    """The list of prompts to chain."""


class PromptChaining:
    """The prompt chaining extension."""

    def __init__(
        self, config: PromptChainingConfig, llm: LLM, prompt: PromptConfig
    ) -> None:
        """The service is initialized with a chaining configuration and an LLM.

        Parameters:
            config: The configuration for the prompt chaining
            llm: The language model to use for the generation
            prompt: The prompt that is used to wrap the questions
        """
        self.llm = llm
        self.config = config
        self.prompt = prompt

    def generate(self, parameters: dict[str, str]) -> str:
        """Generate text based on the prompt configuration and use chaining.

        Parameters:
            parameters: The parameters to format the prompt with

        Returns:
            The result of the last prompt that was chained
        """
        res = ""
        for i, prompt in enumerate(self.config.prompts, 1):
            resolved = self.prompt.format({"prompt": prompt.format(parameters)})
            _LOGGER.debug("\n===== Input =====\n\n%s\n\n====================", resolved)
            res = self.llm.generate(resolved).strip()
            _LOGGER.debug("\n===== Response =====\n\n%s\n\n====================", res)
            parameters[f"response_{i}"] = res
        return res
