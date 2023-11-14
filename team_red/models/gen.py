from pydantic import BaseModel

from team_red.models.model import ModelConfig


class GenerationConfig(BaseModel):
    model: ModelConfig
