from pydantic import BaseModel

from team_red.models.model import ModelConfig


class ContinuationConfig(BaseModel):
    model: ModelConfig

class GenerationFeaturesConfig(BaseModel):
    continuation: ContinuationConfig

class GenerationConfig(BaseModel):
    model: ModelConfig
    features: GenerationFeaturesConfig