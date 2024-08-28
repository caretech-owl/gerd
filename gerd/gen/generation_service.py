import logging
import string
from typing import Any, Dict

import gerd.backends.loader as gerd_loader
from gerd.models.gen import GenerationConfig
from gerd.models.model import PromptConfig
from gerd.transport import GenResponse

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class PartialFormatter(string.Formatter):
    def __init__(self, missing: str = "", bad_fmt: str = "!!") -> None:
        self.missing, self.bad_fmt = missing, bad_fmt

    def get_field(self, field_name: str, args: Any, kwargs: Any) -> Any:  # noqa: ANN401
        try:
            val = super(PartialFormatter, self).get_field(field_name, args, kwargs)
            return val
        except (KeyError, AttributeError):
            return None, field_name

    def format_field(self, value: Any, spec: Any) -> Any:  # noqa: ANN401
        if value is None:
            return self.missing
        try:
            return super(PartialFormatter, self).format_field(value, spec)
        except ValueError:
            if self.bad_fmt is not None:
                return self.bad_fmt
            else:
                raise


class GenerationService:
    def __init__(self, config: GenerationConfig) -> None:
        self._config = config
        self._model = gerd_loader.load_model_from_config(self._config.model)

    def set_prompt(self, config: PromptConfig) -> PromptConfig:
        self._config.model.prompt = config
        return self._config.model.prompt

    def get_prompt(self) -> PromptConfig:
        return self._config.model.prompt

    def generate(
        self, parameters: Dict[str, str], add_prompt: bool = False
    ) -> GenResponse:
        if self._config.features.prompt_chaining:
            from gerd.features.prompt_chaining import PromptChaining

            response = PromptChaining(
                self._config.features.prompt_chaining,
                self._model,
                self._config.model.prompt,
            ).generate(parameters)
        else:
            template = self._config.model.prompt.template
            resolved = (
                template.render(**parameters)
                if template
                else self._config.model.prompt.text.format(**parameters)
            )
            _LOGGER.debug(
                "\n====== Resolved prompt =====\n\n%s\n\n=============================",
                resolved,
            )
            response = self._model.generate(resolved)
            _LOGGER.debug(
                "\n====== Response =====\n\n%s\n\n=============================",
                response,
            )
        return GenResponse(text=response, prompt=resolved if add_prompt else None)

    def gen_continue(self, parameters: Dict[str, str]) -> GenResponse:
        fmt = PartialFormatter()
        if not self._config.features.continuation:
            return GenResponse(
                status=400,
                error_msg="Continuation feature is not configured for this model.",
            )
        continue_prompt = self._config.features.continuation.model.prompt.text
        resolved = fmt.format(continue_prompt, **parameters)
        _LOGGER.debug(
            "\n====== Resolved prompt =====\n\n%s\n\n=============================",
            resolved,
        )
        response = self._model.generate(resolved)
        _LOGGER.debug(
            "\n====== Response =====\n\n%s\n\n=============================", response
        )
        return GenResponse(text=response)
