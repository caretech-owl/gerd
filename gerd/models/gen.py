from pydantic import BaseModel

from gerd.features.prompt_chaining import PromptChainingConfig
from gerd.models.model import ModelConfig


class ContinuationConfig(BaseModel):
    model: ModelConfig


class GenerationFeaturesConfig(BaseModel):
    continuation: ContinuationConfig | None = None
    prompt_chaining: PromptChainingConfig | None = None


class GenerationConfig(BaseModel):
    model: ModelConfig
    features: GenerationFeaturesConfig = GenerationFeaturesConfig()
