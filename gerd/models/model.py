import os
import sys
from pathlib import Path
from string import Formatter
from typing import Any, List, Literal, Mapping, Optional, Tuple

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from jinja2 import Environment, FileSystemLoader, Template, meta, select_autoescape
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
)

ChatRole = Literal["system", "user", "assistant"]
EndpointType = Literal["llama.cpp", "openai"]


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


class PromptConfig(BaseModel):
    text: str = "{message}"
    template: Optional[Template] = Field(
        exclude=True,
        default=None,
    )
    path: Optional[str] = None
    is_template: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def format(
        self, parameters: Mapping[str, str | list[ChatMessage]] | None = None
    ) -> str:
        if parameters is None:
            parameters = {}
        return (
            self.template.render(**parameters)
            if self.template
            else (
                self.text.format(**parameters)
                if self.text
                else "".join(str(parameters.values()))
            )
        )

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        if self.path:
            # reset self.text when path is set
            self.text = ""
            path = Path(self.path)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    self.text = f.read()
                    if self.is_template or path.suffix == ".jinja2":
                        self.is_template = True
                        loader = FileSystemLoader(path.parent)
                        env = Environment(
                            loader=loader,
                            autoescape=select_autoescape(
                                disabled_extensions=(".jinja2",),
                                default_for_string=True,
                                default=True,
                            ),
                        )
                        self.template = env.get_template(path.name)
            else:
                msg = f"'{self.path}' does not exist!"
                raise ValueError(msg)
        elif self.text and self.is_template:
            self.template = Environment(autoescape=True).from_string(self.text)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def parameters(self) -> List[str]:
        field_names = (
            {fn for _, fn, _, _ in Formatter().parse(self.text or "") if fn is not None}
            if not self.is_template
            else meta.find_undeclared_variables(
                Environment(autoescape=True).parse(self.text or "")
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
        sorted_field_names = sorted(field_names)
        return sorted(
            field_names,
            key=lambda key: (
                custom_order.index(key) if key in custom_order else len(custom_order),
                sorted_field_names.index(key),
            ),
        )


class ModelEndpoint(BaseModel):
    url: str
    type: EndpointType
    key: Optional[SecretStr] = None


# Default values chosen by https://github.com/marella/ctransformers#config
class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt_setup: List[Tuple[Literal["system", "user", "assistant"], PromptConfig]] = []
    prompt_config: PromptConfig = PromptConfig()
    endpoint: Optional[ModelEndpoint] = None
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
    batch_size: int = 8
    threads: Optional[int] = None
    context_length: int = 0  # Currently only LLaMA, MPT and Falcon
    gpu_layers: int = 0
    torch_dtype: Optional[str] = None
    loras: set[Path] = set()
    extra_kwargs: Optional[dict[str, Any]] = None
