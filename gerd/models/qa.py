"""Data definitions for QA model configuration."""

from typing import Optional, Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from gerd.models.model import ModelConfig


class EmbeddingConfig(BaseModel):
    """Embedding specific model configuration."""

    chunk_size: int
    """The size of the chunks stored in the database."""
    chunk_overlap: int
    """The overlap between chunks."""
    model: ModelConfig
    """The model used for the embedding.

    This model should be rather small and fast to compute.
    Furthermore, not every model is suited for this task."""
    db_path: Optional[str] = None
    """The path to the database file."""


class AnalyzeConfig(BaseModel):
    """The configuration for the analyze service."""

    model: ModelConfig
    """The model to be used for the analyze service."""


class QAFeaturesConfig(BaseModel):
    """Configuration for the QA-specific features."""

    analyze: AnalyzeConfig
    """Configuration to extract letter of discharge information from the text."""
    analyze_mult_prompts: AnalyzeConfig
    """Configuration to extract predefined infos with multiple prompts from the text."""
    return_source: bool
    """Whether to return the source in the response."""


class QAConfig(BaseSettings):
    """Configuration for the QA services.

    This model can be used to retrieve parameters from a variety of sources.
    The main source are YAML files (loaded as [`Settings`][gerd.config.Settings]) but
    dotenv files and environment variables can be used to
    situatively overwrite the values.
    Environment variables have to be prefixed with `gerd_qa_` to be recognized.
    """

    model_config = SettingsConfigDict(
        env_prefix="gerd_qa_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    model: ModelConfig
    """The model to be used for the QA service."""
    embedding: EmbeddingConfig
    """The configuration for the embedding service."""
    features: QAFeaturesConfig
    """The configuration for the QA-specific features."""
    device: str = "cpu"
    """The device to run the model on."""

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
