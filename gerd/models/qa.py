from typing import Optional, Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from gerd.models.model import ModelConfig


class EmbeddingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    vector_count: int
    model: ModelConfig
    db_path: Optional[str] = None


class FactCheckingConfig(BaseModel):
    enabled: bool
    model: ModelConfig


class AnalyzeConfig(BaseModel):
    model: ModelConfig


class QAFeaturesConfig(BaseModel):
    fact_checking: FactCheckingConfig
    analyze: AnalyzeConfig
    analyze_mult_prompts: AnalyzeConfig
    return_source: bool


class QAConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="gerd_qa_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    model: ModelConfig
    embedding: EmbeddingConfig
    features: QAFeaturesConfig
    device: str = "cpu"

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
