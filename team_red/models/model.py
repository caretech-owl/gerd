from typing import Optional

from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    type: str
    file: Optional[str]
    temperature: float
    max_new_tokens: int
