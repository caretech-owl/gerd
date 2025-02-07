"""Implements the Generation class.

The generation services is meant to generate text based on a prompt and/or the
continuation of a provided text.
"""

import logging
import string
from typing import Any, Dict

import gerd.backends.loader as gerd_loader
from gerd.models.gen import GenerationConfig
from gerd.models.model import PromptConfig
from gerd.transport import GenResponse

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class GenerationService:
    """Service to generate text based on a prompt."""

    def __init__(self, config: GenerationConfig) -> None:
        """Initialize the generation service and loads the model.

        Parameters:
            config: The configuration for the generation service
        """
        self.config = config
        self._model = gerd_loader.load_model_from_config(self.config.model)

    def set_prompt_config(
        self,
        config: PromptConfig,
    ) -> PromptConfig:
        """Sets the prompt configuration.

        Parameters:
            config: The prompt configuration
        Returns:
            The prompt configuration; Should be the same as the input in most cases
        """
        self.config.model.prompt_config = config
        return self.config.model.prompt_config

    def get_prompt_config(self) -> PromptConfig:
        """Get the prompt configuration.

        Returns:
            The prompt configuration
        """
        return self.config.model.prompt_config

    def generate(
        self, parameters: Dict[str, str], add_prompt: bool = False
    ) -> GenResponse:
        """Generate text based on the prompt configuration.

        The actual prompt is provided by the prompt configuration.
        The list of parameters is used to format the prompt
        and replace the placeholders. The list can be empty if
        the prompt does not contain any placeholders.

        Parameters:
            parameters: The parameters to format the prompt with
            add_prompt: Whether to add the prompt to the response

        Returns:
            The generation result
        """
        if self.config.features.prompt_chaining:
            from gerd.features.prompt_chaining import PromptChaining

            response = PromptChaining(
                self.config.features.prompt_chaining,
                self._model,
                self.config.model.prompt_config,
            ).generate(parameters)
        else:
            template = self.config.model.prompt_config.template
            resolved = (
                template.render(**parameters)
                if template
                else self.config.model.prompt_config.text.format(**parameters)
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
