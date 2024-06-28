from functools import cached_property
from pathlib import Path
from string import Formatter
from typing import Any, List, Optional

from jinja2 import Environment, FileSystemLoader, Template, meta
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    computed_field,
)


class PromptConfig(BaseModel):
    text: str = ""
    path: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if not self.text and self.path is not None:
            path = Path(self.path)
            if Path(self.path).exists():
                with path.open("r", encoding="utf-8") as f:
                    self.text = f.read()
            else:
                msg = f"Prompt text is not set and '{self.path}' does not exist!"
                raise ValidationError(msg)

    @cached_property
    def template(self) -> Template | None:
        if not self.path:
            return Template(self.text)
        path = Path(self.path)
        if path.suffix == ".jinja":
            loader = FileSystemLoader(searchpath=str(path.parent))
            env = Environment(loader=loader, autoescape=True)
            return env.get_template(path.name)
        return None

    @computed_field  # type: ignore[misc]
    @property
    def parameters(self) -> List[str]:
        field_names = (
            {fn for _, fn, _, _ in Formatter().parse(self.text) if fn is not None}
            if not self.template
            else meta.find_undeclared_variables(
                Environment(autoescape=True).parse(self.text)
            )
        )
        custom_order = [
            "attending_physician",
            "hospital",
            "patient_name",
            "patient_birth_date",
            "patient_address",
            "date_of_stay",
            "anamnesis",
            "diagnosis",
            "treatment",
            "medication",
        ]
        return sorted(
            field_names,
            key=lambda key: (
                custom_order.index(key) if key in custom_order else len(custom_order),
                list(field_names).index(key),
            ),
        )


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
