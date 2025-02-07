"""Models for the generation and chat service."""

from typing import Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from gerd.features.prompt_chaining import PromptChainingConfig
from gerd.models.model import ModelConfig


class GenerationFeaturesConfig(BaseModel):
    """Configuration for the generation-specific features."""

    prompt_chaining: PromptChainingConfig | None = None
    """Configuration for prompt chaining."""


class GenerationConfig(BaseSettings):
    """Configuration for the generation services.

    A configuration can be used for the
    [GenerationService][gerd.services.generation.GenerationService] or
    the [ChatService][gerd.services.chat.ChatService].
    Both support to generate text based on a prompt.
    """

    model_config = SettingsConfigDict(
        env_prefix="gerd_gen_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    model: ModelConfig = ModelConfig()
    """The model to be used for the generation service."""
    features: GenerationFeaturesConfig = GenerationFeaturesConfig()
    """The extra features to be used for the generation service."""

    @classmethod
    def settings_customise_sources(
        cls,
        _: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Customize the settings sources used by pydantic-settings.

        The order of the sources is important.
        The first source has the highest priority.

        Parameters:
            cls: The class of the settings.
            init_settings: The settings from the initialization.
            env_settings: The settings from the environment.
            dotenv_settings: The settings from the dotenv file.
            file_secret_settings: The settings from the secret file.

        Returns:
            The customized settings sources.
        """
        return (
            file_secret_settings,
            env_settings,
            dotenv_settings,
            init_settings,
        )
