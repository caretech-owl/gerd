from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError, computed_field


class PromptConfig(BaseModel):
    text: str = ""
    path: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if not self.text and self.path is not None:
            if Path(self.path).exists():
                with Path(self.path).open("r", encoding="utf-8") as f:
                    self.text = f.read()
            else:
                msg = f"Prompt text is not set and '{self.path}' does not exist!"
                raise ValidationError(msg)

    @computed_field  # type: ignore[misc]
    @property
    def parameters(self) -> List[str]:
        field_names = {fn for _, fn, _, _ in Formatter().parse(self.text)
                       if fn is not None}
        return sorted(field_names)


# Default values chosen by https://github.com/marella/ctransformers#config
class ModelConfig(BaseModel):
    name: str
    prompt: PromptConfig = PromptConfig()
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
