"""Training of LoRA models on unstructured text data.

This module provides functions to train LoRA models to 'imitate' the style
of a given text corpus.
"""

import json
import logging
import os
import shutil
from pathlib import Path

from datasets import Dataset

from gerd.training.data import split_chunks, tokenize
from gerd.training.lora import LoraTrainingConfig, load_training_config
from gerd.training.trainer import Trainer

_LOGGER = logging.getLogger(__name__)


def train_lora(
    config: str | LoraTrainingConfig, texts: list[str] | None = None
) -> Trainer:
    """Train a LoRA model on unstructured text data.

    Parameters:
        config: The configuration name or the configuration itself
        texts: The list of texts to train on, if None,
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

    if not texts:
        texts = []
        # currently, we expect the input glob to point to a local directory
        # in the future, we might want to support other sources
        glob_pattern = lora_config.input_glob.replace("file://", "")
        for file in Path().glob(glob_pattern):
            with open(file, "r") as f:
                texts.append(f.read())

    training_tokens: list[list[int]] = []
    for text in texts:
        if len(text) == 0:
            continue
        training_tokens.extend(
            split_chunks(
                lora_config.tokenizer.encode(text),
                lora_config.cutoff_len,
                lora_config.cutoff_len - lora_config.overlap_len,
            )
        )
    text_chunks = [lora_config.tokenizer.decode(x) for x in training_tokens]
    train_data = Dataset.from_list(
        [
            tokenize(x, lora_config.tokenizer, lora_config.cutoff_len)
            for x in text_chunks
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
        train_data=train_data, train_template={"template_type": "raw_text"}
    )

    trainer.train()
    return trainer
