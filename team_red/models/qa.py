from typing import Optional

from pydantic import BaseModel

from team_red.models.model import ModelConfig


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


class QAConfig(BaseModel):
    model: ModelConfig
    embedding: EmbeddingConfig
    features: QAFeaturesConfig
    device: str = "cpu"
