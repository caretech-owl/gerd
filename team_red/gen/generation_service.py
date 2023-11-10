import logging
from string import Formatter
from typing import Dict, Optional

from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Pipeline, pipeline

from team_red.config import CONFIG
from team_red.transport import GenResponse, PromptConfig

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class GenerationService:
    def __init__(self) -> None:
        self._prompt_config: Optional[PromptConfig] = None
        self._pipeline: Optional[Pipeline] = None

    @property
    def pipeline(self) -> Pipeline:
        if not self._pipeline:
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_repo_id=CONFIG.model.name,
                model_file=CONFIG.model.file,
                model_type=CONFIG.model.type,
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
        names = {
            fn: "" for _, fn, _, _ in Formatter().parse(config.text) if fn is not None
        }
        config.parameters = names
        self._prompt_config = config
        return self._prompt_config

    def get_prompt(self) -> PromptConfig:
        return self._prompt_config or PromptConfig(text="")

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        if self._prompt_config:
            prompt = self._prompt_config.text.format(**parameters)
        else:
            # if there is no prompt set,
            # we will just forward the passed dictionary values
            _LOGGER.warning("No prompt was set before query '%s'", parameters)
            prompt = " ".join(parameters.values())
        _LOGGER.debug(
            "\n====== Resolved prompt =====\n\n%s\n\n=============================",
            prompt,
        )

        response = self.pipeline(
            prompt,
            do_sample=True,
            top_p=0.95,
            max_new_tokens=CONFIG.model.max_new_tokens,
        )
        _LOGGER.debug(
            "\n====== Response =====\n\n%s\n\n=============================", response
        )
        response_content = response[0]["generated_text"]
        return GenResponse(text=response_content)
