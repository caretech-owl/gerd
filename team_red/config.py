from pathlib import Path
from typing import Any, Dict, Tuple, Type

from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from yaml import safe_load

from team_red.models.gen import GenerationConfig
from team_red.models.logging import LoggingConfig
from team_red.models.qa import QAConfig

PROJECT_DIR = Path(__file__).parent.parent


class YamlConfig(PydanticBaseSettingsSource):
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        raise NotImplementedError()

    # def prepare_field_value(
    #     self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    # ) -> Any:
    #     raise NotImplementedError()

    def __call__(self) -> Dict[str, Any]:
        with Path(PROJECT_DIR, "config", "config.yml").open("r", encoding="utf-8") as f:
            d: Dict[str, Any] = safe_load(f)
        return d


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )
    device: str
    logging: LoggingConfig
    gen: GenerationConfig
    qa: QAConfig

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


CONFIG = Settings()  # type: ignore[call-arg]
