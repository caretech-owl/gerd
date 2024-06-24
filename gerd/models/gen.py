from pydantic import BaseModel

from gerd.models.model import ModelConfig


class ContinuationConfig(BaseModel):
    model: ModelConfig


class GenerationFeaturesConfig(BaseModel):
    continuation: ContinuationConfig | None = None


class GenerationConfig(BaseModel):
    model: ModelConfig
    features: GenerationFeaturesConfig
