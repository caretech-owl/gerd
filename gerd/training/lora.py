"""Configuration dataclasses for training LoRA models."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Type

import torch
import transformers

# from peft.utils.other import (
#     TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as model_to_lora_modules,
# )
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from yaml import safe_load

from gerd.config import CONFIG, PROJECT_DIR
from gerd.models.model import ModelConfig

MODEL_CLASSES: dict[str, str] = {
    v[1]: v[0] for v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()
}


class LLMModelProto(Protocol):
    """Protocol for the LoRA model.

    A model model needs to implement the named_modules method for
    it to be used in LoRA Training.
    """

    def named_modules(self) -> list[tuple[str, torch.nn.Module]]:
        """Get the named modules of the model.

        Returns:
            The named modules.
        """
        pass


class LoraModules(BaseModel):
    """Configuration for the modules to be trained in LoRA models."""

    q: Optional[bool] = None
    v: Optional[bool] = None
    k: Optional[bool] = None
    o: Optional[bool] = None
    gate: Optional[bool] = None
    down: Optional[bool] = None
    up: Optional[bool] = None
    default: bool = True

    def target_modules(self, model: LLMModelProto) -> List[str]:
        """Get the target modules for the given model.

        Parameters:
            model: The model to be trained.

        Returns:
            The list of target modules
        """
        avail = _find_target_modules(model)
        return [
            f"{name}_proj"
            for name, enabled in self.model_dump().items()
            if (enabled is True or (enabled is None and self.default is True))
            and f"{name}_proj" in avail
        ]


class TrainingFlags(BaseModel):
    """Training flags for LoRA models."""

    use_cpu: bool = not torch.cuda.is_available()
    use_bf16: bool = False
    use_ipex: bool = False
    use_4bit: bool = False
    use_8bit: bool = False


class LoraTrainingConfig(BaseSettings):
    """Configuration for training LoRA models."""

    model_config = SettingsConfigDict(
        env_prefix="gerd_lora_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # custom things
    model: ModelConfig = ModelConfig()
    mode: Literal["unstructured", "instructions"] = "unstructured"
    output_dir: Path = Path("loras/lora")
    input_glob: str = ""
    override_existing: bool = False
    zip_output: bool = False
    pad_token_id: int = 0
    padding_side: Literal["right", "left"] = "right"

    # training parameters
    cutoff_len: int = 256
    overlap_len: int = 128
    train_only_after: str = ""
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    save_steps: int = 500  # Save every n steps
    warmup_steps: int = 100
    epochs: int = 3
    r: int = 8
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"

    # learning_rate
    # 3e-4 is a good starting base point.
    # 1e-2 is extremely high, 1e-6 is extremely low.
    learning_rate: float = 3e-4
    batch_size: int = 32
    micro_batch_size: int = 1
    stop_at_loss: float = 0  # (reasonable numbers are 1.5-1.8)

    # optimizer = [
    #     'adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla',
    #     'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision',
    #     'sgd', 'adagrad']
    # Different optimizer implementation options, for advanced users.
    # Effects of different options are not well documented yet.',
    optimizer: str = "adamw_torch"

    # lr_scheduler = [
    #    'linear', 'constant', 'constant_with_warmup', 'cosine',
    #    'cosine_with_restarts', 'polynomial', 'inverse_sqrt']
    # Learning rate scheduler - defines how the learning rate changes over time.
    # "Constant" means never change, "linear" means to go in a straight line from the
    # learning rate down to 0, cosine follows a curve, etc.'
    lr_scheduler: str = "linear"

    modules: LoraModules = Field(default_factory=LoraModules)
    flags: TrainingFlags = Field(default_factory=TrainingFlags)

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Get the tokenizer for the model."""
        if self._tokenizer is None:
            self.reset_tokenizer()
        return self._tokenizer

    def reset_tokenizer(self) -> None:
        """Resets the tokenizer.

        When a tokenizer has been used it needs to be reset
        before changig parameters to avoid issues with parallelism.
        """
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model.name, trust_remote_code=False, use_fast=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self.pad_token_id
        if self.padding_side:
            self._tokenizer.padding_side = self.padding_side

    def model_post_init(self, _: Any) -> None:  # noqa: ANN401
        """Post-initialization hook for the model.

        This method currently checks whether cutoff is larger than overlap.
        """
        if self.cutoff_len <= self.overlap_len:
            msg = (
                "Overlap must be smaller than cutoff"
                f"({self.cutoff_len}) but is {self.overlap_len}"
            )
            raise ValueError(msg)
        self._tokenizer = None

    @classmethod
    def settings_customise_sources(
        cls,
        _: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize the settings sources used by pydantic-settings.

        The order of the sources is important.
        The first source has the highest priority.

        Parameters:
            cls: The class of the settings.
            init_settings: The settings from the initialization.
            env_settings: The settings from the environment.
            dotenv_settings: The settings from the dotenv file.
            file_secret_settings: The settings from the secret file.

        Returns:
            The customized settings sources.
        """
        return (
            file_secret_settings,
            env_settings,
            dotenv_settings,
            init_settings,
        )


def _find_target_modules(model: LLMModelProto) -> List[str]:
    """Find the modules to be trained for the given model.

    Parameters:
        model: The model to be trained.

    Returns:
        The list of modules to be trained.
    """
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split(".")[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


def load_training_config(config: str) -> LoraTrainingConfig:
    """Load the LLM model configuration.

    Parameters:
        config: The name of the configuration.

    Returns:
        The model configuration.
    """
    config_path = (
        Path(config)
        if config.endswith("yml")
        else Path(PROJECT_DIR, "config", f"{config}.yml")
    )
    with config_path.open("r", encoding="utf-8") as f:
        conf = LoraTrainingConfig.model_validate(safe_load(f))
    return conf
