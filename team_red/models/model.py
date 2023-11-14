from typing import List, Optional

from pydantic import BaseModel


# Default values chosen by https://github.com/marella/ctransformers#config
class ModelConfig(BaseModel):
    name: str
    type: Optional[str] = None
    file: Optional[str] = None
    top_k: int = 40
    top_p: float = 0.95
    temperature: float = 0.8
    repetition_penalty: float = 1.1
    last_n_tokens: int = 64
    seed: int = -1
    max_new_tokens: int = 256
    stop: Optional[List[str]] = None
    stream: bool = False
    reset: bool = False
    batch_size: int = 8
    threads: int = -1
    context_length: int = -1  # Currently only LLaMA, MPT and Falcon
    gpu_layers: int = 0
