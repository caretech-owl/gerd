from typing import Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from gerd.features.prompt_chaining import PromptChainingConfig
from gerd.models.model import ModelConfig


class ContinuationConfig(BaseModel):
    model: ModelConfig


class GenerationFeaturesConfig(BaseModel):
    continuation: ContinuationConfig | None = None
    prompt_chaining: PromptChainingConfig | None = None


class GenerationConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="gerd_gen_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    model: ModelConfig = ModelConfig()
    features: GenerationFeaturesConfig = GenerationFeaturesConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        _: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            file_secret_settings,
            env_settings,
            dotenv_settings,
            init_settings,
        )
