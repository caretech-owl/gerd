"""Configuration for the application."""

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
    """YAML configuration source."""

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        """Overrides a method from `PydanticBaseSettingsSource`.

        Fails if it should ever be called.
        Parameters:
            field: The field to get the value for.
            field_name: The name of the field.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()

    def __call__(self) -> Dict[str, Any]:
        """Load the configuration from a YAML file."""
        with Path(PROJECT_DIR, "config", "config.yml").open("r", encoding="utf-8") as f:
            d: Dict[str, Any] = safe_load(f)
        return d


class EnvVariables(BaseModel):
    """Environment variables."""

    api_token: SecretStr | None = None


class Settings(BaseSettings):
    """Settings for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="gerd_config_",
        extra="ignore",
    )
    logging: LoggingConfig
    server: ServerConfig
    env: EnvVariables = EnvVariables()
    kiosk_mode: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
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
            YamlConfig(settings_cls),
        )


def load_gen_config(config: str = "gen_default") -> GenerationConfig:
    """Load the LLM model configuration.

    Parameters:
        config: The name of the configuration.

    Returns:
        The model configuration.
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

    Parameters:
        config: The name of the configuration.

    Returns:
        The model configuration.
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
"""The global configuration object."""
