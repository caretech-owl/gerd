"""Training module for instruction text sets.

In contrast to the [`unstructured`](gerd.training.unstructured) training module,
data must be prepared in a specific format to train LoRA models.
Make sure to provide the training data in the correct format.
"""

import json
import logging
import os
import shutil
from pathlib import Path

import yaml
from datasets import Dataset
from pydantic import BaseModel

from gerd.models.model import ChatMessage
from gerd.training.lora import LoraTrainingConfig, load_training_config
from gerd.training.trainer import Trainer

_LOGGER = logging.getLogger(__name__)


class InstructTrainingSample(BaseModel):
    """Dataclass to hold a training sample for instruction text sets.

    A training sample consists of a list of chat messages.
    """

    messages: list[ChatMessage]
    """The list of chat messages."""


class InstructTrainingData(BaseModel):
    """Dataclass to hold training data for instruction text sets.

    A training data object consists of a list of training samples.
    """

    samples: list[InstructTrainingSample] = []
    """The list of training samples."""


def train_lora(
    config: str | LoraTrainingConfig, data: InstructTrainingData | None = None
) -> Trainer:
    """Train a LoRA model on instruction text sets.

    Parameters:
        config: The configuration name or the configuration itself
        data: The training data to train on, if None,
            the input_glob from the config is used

    Returns:
        The trainer instance that is used for training
    """
    # Disable parallelism to avoid issues with transformers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    lora_config = load_training_config(config) if isinstance(config, str) else config

    if Path(lora_config.output_dir).joinpath("adapter_model.safetensors").exists():
        if lora_config.override_existing:
            # check that we do not delete anything vital
            if lora_config.output_dir == Path("/"):
                msg = "Cannot delete root directory."
                raise RuntimeError(msg)

            _LOGGER.warning(
                "Overriding existing LoRA adapter in %s ...", lora_config.output_dir
            )
            shutil.rmtree(lora_config.output_dir)
        else:
            msg = (
                f"LoRA target directory {lora_config.output_dir}"
                " must not contain another lora adapter."
            )
            raise AssertionError(msg)

    _LOGGER.info("Tokenizing training data ...")

    if data is None:
        data = InstructTrainingData()
        # currently, we expect the input glob to point to a local directory
        # in the future, we might want to support other sources
        glob_pattern = lora_config.input_glob.replace("file://", "")
        for file in Path().glob(glob_pattern):
            if not file.is_file():
                msg = "Can only read files for now."
                raise NotImplementedError(msg)
            if file.suffix == ".json":
                with open(file, "r") as f:
                    data.samples.extend(
                        InstructTrainingData.model_validate_json(f.read()).samples
                    )
            elif file.suffix == ".yml":
                with open(file, "r") as f:
                    obj = yaml.safe_load(f)
                    data.samples.extend(
                        InstructTrainingData.model_validate(obj).samples
                    )
            else:
                msg = f"Unsupported file format: {file.suffix}"
                raise NotImplementedError(msg)

    train_data = Dataset.from_list(
        [
            lora_config.tokenizer(
                lora_config.tokenizer.apply_chat_template(
                    sample.messages, tokenize=False
                )
            )
            for sample in data.samples
        ]
    )
    _LOGGER.info("Decoding sample data ...")
    decoded_entries = []
    for i in range(min(10, len(train_data))):
        decoded_text = lora_config.tokenizer.decode(train_data[i]["input_ids"])
        decoded_entries.append({"value": decoded_text})

    log_dir = lora_config.output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    _LOGGER.info("Writing sample to %s ...", log_dir)
    with open(Path(f"{log_dir}/train_dataset_sample.json"), "w") as json_file:
        json.dump(decoded_entries, json_file, indent=4)

    trainer = Trainer(config=lora_config)
    trainer.setup_training(
        train_data=train_data, train_template={"template_type": "dataset"}
    )

    trainer.train()
    return trainer
