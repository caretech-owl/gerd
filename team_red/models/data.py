from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    vector_count: int
    model: str
    db_path: str


class DataConfig(BaseModel):
    path: str
    chunk_size: int
    chunk_overlap: int
    embedding: EmbeddingConfig
