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
    def named_modules(self) -> list[tuple[str, torch.nn.Module]]:
        pass


class LoraModules(BaseModel):
    q: Optional[bool] = None
    v: Optional[bool] = None
    k: Optional[bool] = None
    o: Optional[bool] = None
    gate: Optional[bool] = None
    down: Optional[bool] = None
    up: Optional[bool] = None
    default: bool = True

    def target_modules(self, model: LLMModelProto) -> List[str]:
        avail = find_target_modules(model)
        return [
            f"{name}_proj"
            for name, enabled in self.model_dump().items()
            if (enabled is True or (enabled is None and self.default is True))
            and f"{name}_proj" in avail
        ]


class TrainingFlags(BaseModel):
    use_cpu: bool = not torch.cuda.is_available()
    use_bf16: bool = False
    use_ipex: bool = False


class LoraTrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="gerd_lora_", env_nested_delimiter="__"
    )

    # custom things
    model: ModelConfig
    output_dir: Path
    input_glob: str = ""
    override_existing: bool = False
    pad_token_id: int = 0
    padding_side: str = ""

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
    batch_size: int = 128
    micro_batch_size: int = 4
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
        return self._tokenizer

    def reset_tokenizer(self) -> None:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model.name, trust_remote_code=False, use_fast=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self.pad_token_id
        if self.padding_side:
            self._tokenizer.padding_side = self.padding_side

    def model_post_init(self, _: Any) -> None:  # noqa: ANN401
        if self.cutoff_len <= self.overlap_len:
            msg = (
                "Overlap must be smaller than cutoff"
                f"({self.cutoff_len}) but is {self.overlap_len}"
            )
            raise ValueError(msg)
        self.reset_tokenizer()

    @classmethod
    def settings_customise_sources(
        cls,
        _: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            file_secret_settings,
            env_settings,
            dotenv_settings,
            init_settings,
        )


def find_target_modules(model: LLMModelProto) -> List[str]:
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

    :param config: The name of the configuration.
    :return: The model configuration.
    """
    config_path = (
        Path(config)
        if config.endswith("yml")
        else Path(PROJECT_DIR, "config", f"{config}.yml")
    )
    with config_path.open("r", encoding="utf-8") as f:
        conf = LoraTrainingConfig.model_validate(safe_load(f))
    return conf
