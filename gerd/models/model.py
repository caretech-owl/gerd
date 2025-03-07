"""Model configuration for supported model classes."""

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
"""Currently supported chat roles."""
EndpointType = Literal["llama.cpp", "openai"]
"""Endpoint for remote llm services."""


class ChatMessage(TypedDict):
    """Data structure for chat messages."""

    role: ChatRole
    """The role or source of the chat message."""
    content: str
    """The content of the chat message."""


class PromptConfigBase(BaseModel):
    """Parameters for a prompt configuration."""

    text: str = "{message}"
    """The text of the prompt. Can contain placeholders."""
    path: Optional[str] = None
    """The path to an external prompt file.

    This will overload the values of text and/or template."""
    is_template: bool = False
    """Whether the config uses jinja2 templates."""


class PromptConfig(PromptConfigBase):
    """Configuration for prompts."""

    template: Optional[Template] = Field(
        exclude=True,
        default=None,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def as_base(self) -> PromptConfigBase:
        """Returns parameter of the prompt configurations.

        Returns:
            Basic prompt configuration parameters
        """
        return PromptConfigBase(
            text=self.text, path=self.path, is_template=self.is_template
        )

    def format(
        self, parameters: Mapping[str, str | list[ChatMessage]] | None = None
    ) -> str:
        """Format the prompt with the given parameters.

        Parameters:
            parameters: The parameters to format the prompt with.

        Returns:
            The formatted prompt
        """
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
        """Post-initialization hook for pyandic.

        When path is set, the text or template is read from the file and
        the template is created.
        Path ending with '.jinja2' will be treated as a template.
        If no path is set, the text parameter is used to initialize the template
        if is_template is set to True.
        Parameters:
            __context: The context of the model (not used)
        """
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
    def parameters(self) -> list[str]:
        """Retrieves and returns the parameters of the prompt.

        This happens on-the-fly and is not stored in the model.

        Returns:
            The parameters of the prompt.
        """
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
    """Configuration for model endpoints where models are hosted remotely."""

    url: str
    """URL of the remote service"""
    type: EndpointType
    """Type of the remote service"""
    key: Optional[SecretStr] = None
    """Token to use the remote service"""


class ModelConfig(BaseModel):
    """Configuration for large language models.

    Most llm libraries and/or services share common parameters for configuration.
    Explaining each parameter is out of scope for this documentation.
    The most essential parameters are explained for instance
    [here](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values).
    Default values have been chosen according to
    [ctransformers](https://github.com/marella/ctransformers#config) library.
    """

    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    """The name of the model. Can be a path to a local model or a huggingface handle."""
    prompt_setup: List[Tuple[Literal["system", "user", "assistant"], PromptConfig]] = []
    """A list of predefined prompts for the model.

    When a model context is inialized or reset,
    this will be used to set up the context."""
    prompt_config: PromptConfig = PromptConfig()
    """The prompt configuration.

    This is used to process the input passed to the services."""
    endpoint: Optional[ModelEndpoint] = None
    """The endpoint of the model when hosted remotely."""
    file: Optional[str] = None
    """The path to the model file. For local models only."""
    top_k: int = 40
    """The number of tokens to consider for the top-k sampling."""
    top_p: float = 0.95
    """The cumulative probability for the top-p sampling."""
    temperature: float = 0.8
    """The temperature for the sampling."""
    repetition_penalty: float = 1.1
    """The repetition penalty."""
    last_n_tokens: int = 64
    """The number of tokens to consider for the repetition penalty."""
    seed: int = -1
    """The seed for the random number generator."""
    max_new_tokens: int = 256
    """The maximum number of new tokens to generate."""
    stop: Optional[List[str]] = None
    """The stop tokens for the generation."""
    stream: bool = False
    """Whether to stream the output."""
    batch_size: int = 8
    """The batch size for the generation."""
    threads: Optional[int] = None
    """The number of threads to use for the generation."""
    context_length: int = 0
    """The context length for the model. Currently only LLaMA, MPT and Falcon"""
    gpu_layers: int = 0
    """The number of layers to run on the GPU.

    The actual number is only used llama.cpp. The other model libraries will determine
    whether to run on the GPU just by checking of this value is larger than 0.
    """
    torch_dtype: Optional[str] = None
    """The torch data type for the model."""
    loras: set[Path] = set()
    """The list of additional LoRAs files to load."""
    extra_kwargs: Optional[dict[str, Any]] = None
    """Additional keyword arguments for the model library.

    The accepted keys and values depend on the model library used.
    """
