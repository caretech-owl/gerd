import logging
from typing import Dict, Optional

from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
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

    @property
    def pipeline(self) -> Pipeline:  # type: ignore[no-any-unimported]
        if not self._pipeline:
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_repo_id=self._config.model.name,
                model_file=self._config.model.file,
                model_type=self._config.model.type,
                context_length=self._config.model.context_length,
                temperature=self._config.model.temperature,
                repetition_penalty =self._config.model.repetition_penalty,
                last_n_tokens = self._config.model.last_n_tokens,
                top_k = self._config.model.top_k,
                top_p = self._config.model.top_p,
                hf=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model)
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                return_full_text=False,
            )
        return self._pipeline

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

        response = self.pipeline(
            resolved,
            do_sample=True,
            top_p=0.95,
            max_new_tokens=self._config.model.max_new_tokens,
        )
        _LOGGER.debug(
            "\n====== Response =====\n\n%s\n\n=============================", response
        )
        response_content = response[0]["generated_text"]
        return GenResponse(text=response_content)
