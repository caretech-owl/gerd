from typing import Optional

from pydantic import BaseModel

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


class QAFeaturesConfig(BaseModel):
    fact_checking: FactCheckingConfig
    return_source: bool


class QAConfig(BaseModel):
    model: ModelConfig
    embedding: EmbeddingConfig
    features: QAFeaturesConfig
    device: str = "cpu"
