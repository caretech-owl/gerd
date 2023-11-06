from pathlib import Path
from typing import Any, Dict, Tuple, Type

from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from yaml import safe_load

from team_red.models.backend import BackendConfig
from team_red.models.data import DataConfig
from team_red.models.features import FeaturesConfig
from team_red.models.logging import LoggingConfig
from team_red.models.model import ModelConfig

PROJECT_DIR = Path(__file__).parent.parent


class YamlConfig(PydanticBaseSettingsSource):
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        raise NotImplementedError()

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        raise NotImplementedError()

    def __call__(self) -> Dict[str, Any]:
        with Path(PROJECT_DIR, "config", "config.yml").open() as f:
            d: Dict[str, Any] = safe_load(f)
        return d


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8")

    backend: BackendConfig
    data: DataConfig
    features: FeaturesConfig
    logging: LoggingConfig
    model: ModelConfig
    device: str

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
            init_settings,
            YamlConfig(settings_cls),
            env_settings,
            file_secret_settings,
        )


CONFIG = Settings()
