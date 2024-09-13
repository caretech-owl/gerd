from pathlib import Path
from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from yaml import safe_load

from gerd.models.gen import GenerationConfig
from gerd.models.logging import LoggingConfig
from gerd.models.qa import QAConfig
from gerd.models.server import ServerConfig

PROJECT_DIR = Path(__file__).parent.parent


class YamlConfig(PydanticBaseSettingsSource):
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        raise NotImplementedError()

    def __call__(self) -> Dict[str, Any]:
        with Path(PROJECT_DIR, "config", "config.yml").open("r", encoding="utf-8") as f:
            d: Dict[str, Any] = safe_load(f)
        return d


class EnvVariables(BaseModel):
    api_token: SecretStr | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )
    logging: LoggingConfig
    server: ServerConfig
    env: EnvVariables | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            file_secret_settings,
            dotenv_settings,
            env_settings,
            init_settings,
            YamlConfig(settings_cls),
        )


def load_gen_config(config: str = "gen_default") -> GenerationConfig:
    """Load the LLM model configuration.

    :param config: The name of the configuration.
    :return: The model configuration.
    """
    config_path = (
        Path(config)
        if config.endswith("yml")
        else Path(PROJECT_DIR, "config", f"{config}.yml")
    )
    with config_path.open("r", encoding="utf-8") as f:
        conf = GenerationConfig.model_validate(safe_load(f))
    if CONFIG.env and CONFIG.env.api_token and conf.model.endpoint:
        conf.model.endpoint.key = CONFIG.env.api_token
    return conf


def load_qa_config(config: str = "qa_default") -> QAConfig:
    """Load the LLM model configuration.

    :param config: The name of the configuration.
    :return: The model configuration.
    """
    config_path = (
        Path(config)
        if config.endswith("yml")
        else Path(PROJECT_DIR, "config", f"{config}.yml")
    )
    with config_path.open("r", encoding="utf-8") as f:
        conf = QAConfig.model_validate(safe_load(f))
    if CONFIG.env and CONFIG.env.api_token and conf.model.endpoint:
        conf.model.endpoint.key = CONFIG.env.api_token
    return conf


CONFIG = Settings()  # type: ignore[call-arg]
