"""Module for loading language models.

Depending on the configuration, different language models are loaded and
different libraries are used. The main goal is to provide a unified interface
to the different models and libraries.
"""

import abc
import logging
import os
from pathlib import Path
from typing import Iterator

from typing_extensions import override

from gerd.models.model import ChatMessage, ChatRole, ModelConfig, ModelEndpoint

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class LLM:
    """The abstract base class for large language models.

    Should be implemented by all language model backends.
    """

    @abc.abstractmethod
    def __init__(self, config: ModelConfig) -> None:
        """A language model is initialized with a configuration.

        Parameters:
            config: The configuration for the language model
        """
        pass

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text based on a prompt.

        Parameters:
            prompt: The prompt to generate text from

        Returns:
            The generated text
        """
        pass

    @abc.abstractmethod
    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> tuple[ChatRole, str]:
        """Create a chat completion based on a list of messages.

        Parameters:
            messages: The list of messages in the chat history

        Returns:
            The role of the generated message and the content
        """
        pass


class MockLLM(LLM):
    """A mock language model for testing purposes."""

    @override
    def __init__(self, _: ModelConfig) -> None:
        self.ret_value = "MockLLM"
        pass

    @override
    def generate(self, _: str) -> str:
        return self.ret_value

    @override
    def create_chat_completion(self, _: list[ChatMessage]) -> tuple[ChatRole, str]:
        return ("assistant", self.ret_value)


class LlamaCppLLM(LLM):
    """A language model using the Llama.cpp library."""

    @override
    def __init__(self, config: ModelConfig) -> None:
        from llama_cpp import Llama

        self.config = config
        self._model = Llama.from_pretrained(
            repo_id=config.name,
            filename=config.file,
            n_ctx=config.context_length,
            n_gpu_layers=config.gpu_layers,
            n_threads=config.threads,
            **config.extra_kwargs or {},
        )

    @override
    def generate(self, prompt: str) -> str:
        res = self._model(
            prompt,
            stop=self.config.stop,
            max_tokens=self.config.max_new_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
            repeat_penalty=self.config.repetition_penalty,
        )
        output = next(res) if isinstance(res, Iterator) else res
        return output["choices"][0]["text"]

    @override
    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> tuple[ChatRole, str]:
        res = self._model.create_chat_completion(
            # mypy cannot resolve the role parameter even though
            # is is defined on compatible literals
            [{"role": m["role"], "content": m["content"]} for m in messages],  # type: ignore[misc]
            stop=self.config.stop,
            max_tokens=self.config.max_new_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            temperature=self.config.temperature,
            repeat_penalty=self.config.repetition_penalty,
        )
        if not isinstance(res, Iterator):
            msg = res["choices"][0]["message"]
            if msg["role"] == "function":
                error_msg = "function role not expected"
                raise NotImplementedError(error_msg)
            return (msg["role"], msg["content"].strip() if msg["content"] else "")

        error_msg = "Cannot process stream responses for now"
        raise NotImplementedError(error_msg)


class TransformerLLM(LLM):
    """A language model using the transformers library."""

    @override
    def __init__(self, config: ModelConfig) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedModel,
            pipeline,
        )

        # use_fast=False is ignored by transformers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.config = config
        torch_dtypes: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

        model_kwargs = config.extra_kwargs or {}
        if config.torch_dtype in torch_dtypes:
            model_kwargs["torch_dtype"] = torch_dtypes[config.torch_dtype]

        tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-any-unimported]
            config.name, **model_kwargs
        )

        loaded_loras = set()
        for lora in config.loras:
            _LOGGER.info("Loading adapter %s", lora)
            if not Path(lora / "adapter_model.safetensors").exists():
                _LOGGER.warning("Adapter %s does not exist", lora)
                continue
            model.load_adapter(lora)
            loaded_loras.add(lora)
            train_params = Path(lora) / "training_parameters.json"
            if train_params.exists() and tokenizer.pad_token_id is None:
                from gerd.training.lora import LoraTrainingConfig

                with open(train_params, "r") as f:
                    lora_config = LoraTrainingConfig.model_validate_json(f.read())
                    tokenizer.pad_token_id = lora_config.pad_token_id
                    # https://github.com/huggingface/transformers/issues/34842#issuecomment-2490994584
                    tokenizer.padding_side = (
                        "left" if lora_config.padding_side == "right" else "right"
                    )
                    # tokenizer.padding_side = lora_config.padding_side

        if loaded_loras:
            model.enable_adapters()

        self._pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            # device_map="auto",  # https://github.com/huggingface/transformers/issues/31922
            device=(
                "cuda"
                if config.gpu_layers > 0
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            ),
            framework="pt",
            use_fast=False,
        )

    @override
    def generate(self, prompt: str) -> str:
        res = self._pipe(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            do_sample=True,
        )
        output: str = res[0]["generated_text"]
        return output

    @override
    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> tuple[ChatRole, str]:
        msg = self._pipe(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            do_sample=True,
        )[0]["generated_text"][-1]
        return (msg["role"], msg["content"].strip())


class RemoteLLM(LLM):
    """A language model using a remote endpoint.

    The endpoint can be any service that are compatible with llama.cpp and openai API.
    For further information, please refer to the llama.cpp
    [server API](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).
    """

    @override
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        if self.config.endpoint is None:
            msg = "Endpoint is required for remote LLM"
            raise ValueError(msg)

        self._ep: ModelEndpoint = self.config.endpoint
        self._prompt_field = "prompt"
        self._header = {"Content-Type": "application/json"}
        if self.config.endpoint.key:
            self._header["Authorization"] = (
                f"Bearer {self.config.endpoint.key.get_secret_value()}"
            )

        if self.config.endpoint.type == "openai":
            self.msg_template = {
                "model": self.config.name,
                "temperature": self.config.temperature,
                "frequency_penalty": self.config.repetition_penalty,
                "max_completion_tokens": self.config.max_new_tokens,
                "n": 1,
                "stop": self.config.stop,
                "top_p": self.config.top_p,
            }
        elif self.config.endpoint.type == "llama.cpp":
            self.msg_template = {
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "repeat_penalty": self.config.repetition_penalty,
                "n_predict": self.config.max_new_tokens,
                "stop": self.config.stop or [],
            }
        else:
            msg = f"Unknown endpoint type: {self.config.endpoint.type}"
            raise ValueError(msg)

    @override
    def generate(self, prompt: str) -> str:
        import json

        import requests

        if self.config.endpoint and self.config.endpoint.type != "llama.cpp":
            msg = (
                "Only llama.cpp supports simple completion yet. "
                "Use chat completion instead."
            )
            raise NotImplementedError(msg)

        self.msg_template[self._prompt_field] = prompt
        res = requests.post(
            self._ep.url + "/completion",
            headers=self._header,
            data=json.dumps(self.msg_template),
            timeout=300,
        )
        if res.status_code == 200:
            return str(res.json()["content"])
        else:
            _LOGGER.warning("Server returned error code %d", res.status_code)
        return ""

    @override
    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> tuple[ChatRole, str]:
        import json

        import requests

        self.msg_template["messages"] = messages
        res = requests.post(
            self._ep.url + "/v1/chat/completions",
            headers=self._header,
            data=json.dumps(self.msg_template),
            timeout=300,
        )
        if res.status_code == 200:
            msg = res.json()["choices"][0]["message"]
            return (msg["role"], msg["content"].strip())
        else:
            _LOGGER.warning("Server returned error code %d", res.status_code)
        return ("assistant", "")


def load_model_from_config(config: ModelConfig) -> LLM:
    """Loads a language model based on the configuration.

    Which language model is loaded depends on the configuration.
    For instance, if an endpoint is provided, a remote language model is loaded.
    If a file is provided, Llama.cpp is used.
    Otherwise, transformers is used.

    Parameters:
        config: The configuration for the language model

    Returns:
        The loaded language model

    """
    if config.endpoint:
        _LOGGER.info("Using remote endpoint %s", config.endpoint.url)
        return RemoteLLM(config)
    if config.file:
        _LOGGER.info(
            "Using Llama.cpp with model %s and file %s", config.name, config.file
        )
        return LlamaCppLLM(config)
    _LOGGER.info("Using transformers with model %s", config.name)
    return TransformerLLM(config)
