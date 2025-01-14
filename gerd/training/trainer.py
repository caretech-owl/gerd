import json
import logging
import math
import shutil
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from gerd.training.lora import MODEL_CLASSES, LoraTrainingConfig

_LOGGER = logging.getLogger(__name__)


@dataclass
class Tracked:
    lora_model: PeftModel
    config: LoraTrainingConfig
    train_log: Dict = field(default_factory=dict)
    current_steps: int = 0
    interrupted: bool = False
    max_steps: int = 0
    did_save: bool = False


class Callbacks(transformers.TrainerCallback):
    def __init__(self, tracked: Tracked) -> None:
        super().__init__()
        self.tracked = tracked
        self.gradient_accumulation_steps = (
            tracked.config.batch_size // tracked.config.micro_batch_size
        )
        self.actual_save_steps = math.ceil(
            tracked.config.save_steps / self.gradient_accumulation_steps
        )

    def on_save(
        self,
        args: transformers.TrainingArguments,  # noqa: ARG002
        state: transformers.TrainerState,  # noqa: ARG002
        control: transformers.TrainerControl,  # noqa: ARG002
        **kwargs: int,  # noqa: ARG002
    ) -> None:
        # Save log
        with open(
            f"{self.tracked.config.output_dir}/{self.tracked.current_steps}-training_log.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.tracked.train_log, file, indent=2)

    def on_step_begin(
        self,
        args: transformers.TrainingArguments,  # noqa: ARG002
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs: int,  # noqa: ARG002
    ) -> None:
        self.tracked.current_steps = (
            state.global_step * self.gradient_accumulation_steps
        )
        self.tracked.max_steps = state.max_steps * self.gradient_accumulation_steps
        if self.tracked.interrupted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_substep_end(
        self,
        args: transformers.TrainingArguments,  # noqa: ARG002
        state: transformers.TrainerState,  # noqa: ARG002
        control: transformers.TrainerControl,
        **kwargs: int,  # noqa: ARG002
    ) -> None:
        self.tracked.current_steps += 1
        if self.tracked.interrupted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_log(
        self,
        args: transformers.TrainingArguments,  # noqa: ARG002
        state: transformers.TrainerState,  # noqa: ARG002
        control: transformers.TrainerControl,
        logs: Dict,
        **kwargs: int,  # noqa: ARG002
    ) -> None:
        self.tracked.train_log.update(logs)
        self.tracked.train_log.update({"current_steps": self.tracked.current_steps})
        if self.tracked.interrupted:
            _LOGGER.info("Interrupted by user")

        # print(f"Step: {self.tracked.current_steps}", end="")
        if "loss" in logs:
            loss = float(logs["loss"])
            if loss <= self.tracked.config.stop_at_loss:
                control.should_epoch_stop = True
                control.should_training_stop = True
                _LOGGER.info("Stop Loss %f reached.", self.tracked.config.stop_at_loss)


class Trainer:
    def __init__(
        self,
        config: LoraTrainingConfig,
        callback_cls: Optional[List[Type[transformers.TrainerCallback]]] = None,
    ) -> None:
        self.config = config
        self.base_model: transformers.PreTrainedModel = (
            transformers.AutoModelForCausalLM.from_pretrained(config.model.name)
        )

        training_config = LoraConfig(
            lora_alpha=config.lora_alpha,
            target_modules=config.modules.target_modules(self.base_model),
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=config.task_type,
            r=config.r,
        )

        for param in self.base_model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        if (
            not hasattr(self.base_model, "lm_head")
            or hasattr(self.base_model.lm_head, "weight")
        ) and "quantization_config" in self.base_model.config.to_dict():
            self.base_model = prepare_model_for_kbit_training(self.base_model)
            _LOGGER.info("Quantization detected!")

        self.lora_model = get_peft_model(self.base_model, training_config)
        if not hasattr(self.lora_model.config, "use_cache"):
            msg = "LoRA model config not have 'use_cache' attribute"
            raise AssertionError(msg)
        self.lora_model.config.use_cache = False

        self.tracked = Tracked(self.lora_model, self.config)
        self.trainer = None
        self.callbacks = (
            [cls(self.tracked) for cls in callback_cls]
            if callback_cls is not None
            else [Callbacks(self.tracked)]
        )

        gradient_accumulation_steps = config.batch_size // config.micro_batch_size
        actual_lr = float(config.learning_rate)

        self.args = transformers.TrainingArguments(
            per_device_train_batch_size=self.config.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=math.ceil(config.warmup_steps / gradient_accumulation_steps),
            num_train_epochs=config.epochs,
            learning_rate=actual_lr,
            fp16=not (config.flags.use_cpu or config.flags.use_bf16),
            bf16=config.flags.use_bf16,
            optim=config.optimizer,
            logging_steps=2 if config.stop_at_loss > 0 else 5,
            eval_strategy="no",
            eval_steps=None,
            save_strategy="no",
            output_dir=config.output_dir,
            lr_scheduler_type=config.lr_scheduler,
            load_best_model_at_end=False,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None,
            use_ipex=config.flags.use_ipex,
            save_steps=config.save_steps,
            # any of these two will set `torch_compile=True`
            # torch_compile_backend="inductor",
            # torch_compile_mode="reduce-overhead",
        )

    def train(self) -> threading.Thread:
        model_type = type(self.base_model).__name__
        model_id = MODEL_CLASSES[model_type]

        projections_string = ", ".join(
            [
                projection.replace("_proj", "")
                for projection in self.config.modules.target_modules(self.base_model)
            ]
        )

        _LOGGER.info(
            "Training '%s' model using (%s) projections", model_id, projections_string
        )
        # == Main run and monitor loop ==

        log = self.tracked.train_log
        log.update({"base_model_config": self.config.model.name})
        log.update({"base_model_class": self.base_model.__class__.__name__})
        log.update(
            {
                "base_loaded_in_4bit": getattr(
                    self.lora_model, "is_loaded_in_4bit", False
                )
            }
        )
        log.update(
            {
                "base_loaded_in_8bit": getattr(
                    self.lora_model, "is_loaded_in_8bit", False
                )
            }
        )
        log.update({"projections": projections_string})

        if self.config.stop_at_loss > 0:
            _LOGGER.info(
                "Monitoring loss \033[1;31;1m(Auto-Stop at: %f)\033[0;37;0m",
                self.config.stop_at_loss,
            )

        def threaded_run() -> None:
            if self.trainer is None:
                msg = "Trainer not initialized"
                raise RuntimeError(msg)
            self.trainer.train()
            # Note: save in the thread in case the gradio thread breaks
            # (eg browser closed)
            self.save()
            self.tracked.did_save = True
            # Save log
            with open(
                f"{self.config.output_dir}/training_log.json", "w", encoding="utf-8"
            ) as file:
                json.dump(self.tracked.train_log, file, indent=2)

        self.thread = threading.Thread(target=threaded_run)
        self.thread.start()
        return self.thread

    def setup_training(
        self,
        train_data: Dataset,
        train_template: Dict,
        torch_compile: bool = False,
    ) -> None:
        self.trainer = transformers.Trainer(
            model=self.lora_model,
            train_dataset=train_data,
            eval_dataset=None,
            args=self.args,
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.config.tokenizer, mlm=False
            ),
            callbacks=self.callbacks,
        )

        # This must be done after Trainer init because otherwise
        # the trainer cannot identify the relevant (as in 'trainable') parameters
        # and will remove required information from the training data set.
        # Whether it is useful to compile after reassignment.
        if torch_compile and torch.__version__ >= "2" and sys.platform != "win32":
            self.lora_model = torch.compile(self.lora_model)  # type: ignore[assignment]

        # == Save parameters for reuse ==
        with open(
            f"{self.config.output_dir}/training_parameters.json", "w", encoding="utf-8"
        ) as file:
            file.write(self.config.model_dump_json(indent=2))

        # == Save training prompt ==
        with open(
            f"{self.config.output_dir}/training_prompt.json", "w", encoding="utf-8"
        ) as file:
            json.dump(train_template, file, indent=2)

    def save(self) -> None:
        if self.trainer is not None:
            self.trainer.save_model(self.config.output_dir)
            if self.config.zip_output:
                shutil.make_archive(
                    base_name=self.config.output_dir.as_posix(),
                    root_dir=self.config.output_dir,
                    # base_dir=self.config.output_dir,
                    format="zip",
                )
        else:
            _LOGGER.warning("Trainer not initialized")

    def interrupt(self) -> None:
        self.tracked.interrupted = True
