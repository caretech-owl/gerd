import logging
from typing import Dict, Optional

from ctransformers import AutoModelForCausalLM
from transformers.pipelines.base import Pipeline

from team_red.models.gen import GenerationConfig
from team_red.models.model import PromptConfig
from team_red.transport import GenResponse

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

class GenerationService:
    def __init__(self, config: GenerationConfig) -> None:
        self._config = config
        self._pipeline: Optional[Pipeline] = None  # type: ignore[no-any-unimported]

    def set_model(self) -> AutoModelForCausalLM:  # type: ignore[no-any-unimported]
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=self._config.model.name,
            model_file=self._config.model.file,
            model_type=self._config.model.type,
        )
        return model

    def set_prompt(self, config: PromptConfig) -> PromptConfig:
        self._config.model.prompt = config
        return self._config.model.prompt

    def get_prompt(self) -> PromptConfig:
        return self._config.model.prompt

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        resolved = self._config.model.prompt.text.format(**parameters)
        _LOGGER.debug(
            "\n====== Resolved prompt =====\n\n%s\n\n=============================",
            resolved,
        )
        response = self.set_model()(
            resolved,
            stop="<|im_end|>",
            max_new_tokens=self._config.model.max_new_tokens,
            top_p=self._config.model.top_p,
            top_k=self._config.model.top_k,
            temperature=self._config.model.temperature,
            repetition_penalty =self._config.model.repetition_penalty,
        )
        _LOGGER.debug(
            "\n====== Response =====\n\n%s\n\n=============================", response
        )
        return GenResponse(text=response)
