import abc
import logging
from typing import Iterator

from gerd.models.model import ChatMessage, ChatRole, ModelConfig, ModelEndpoint

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class LLM:
    @abc.abstractmethod
    def __init__(self, config: ModelConfig) -> None:
        pass

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abc.abstractmethod
    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> tuple[ChatRole, str]:
        pass


class MockLLM(LLM):
    def __init__(self, _: ModelConfig) -> None:
        self.ret_value = "MockLLM"
        pass

    def generate(self, _: str) -> str:
        return self.ret_value

    def create_chat_completion(self, _: list[ChatMessage]) -> tuple[ChatRole, str]:
        return ("assistant", self.ret_value)


class LlamaCppLLM(LLM):
    def __init__(self, config: ModelConfig) -> None:
        from llama_cpp import Llama

        self._config = config
        self._model = Llama.from_pretrained(
            repo_id=config.name,
            filename=config.file,
            n_ctx=config.context_length,
            n_gpu_layers=config.gpu_layers,
            n_threads=config.threads,
        )

    def generate(self, prompt: str) -> str:
        res = self._model(
            prompt,
            stop=self._config.stop,
            max_tokens=self._config.max_new_tokens,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            temperature=self._config.temperature,
            repeat_penalty=self._config.repetition_penalty,
        )
        output = next(res) if isinstance(res, Iterator) else res
        return output["choices"][0]["text"]

    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> tuple[ChatRole, str]:
        res = self._model.create_chat_completion(
            # mypy cannot resolve the role parameter even though
            # is is defined on compatible literals
            [{"role": m["role"], "content": m["content"]} for m in messages],  # type: ignore[misc]
            stop=self._config.stop,
            max_tokens=self._config.max_new_tokens,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            temperature=self._config.temperature,
            repeat_penalty=self._config.repetition_penalty,
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
    def __init__(self, config: ModelConfig) -> None:
        import torch
        from transformers import pipeline

        self._config = config
        torch_dtypes: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

        model_kwargs = {}
        if config.torch_dtype in torch_dtypes:
            model_kwargs["torch_dtype"] = torch_dtypes[config.torch_dtype]

        self._pipe = pipeline(
            task="text-generation",
            model=config.name,
            # device_map="auto",  # https://github.com/huggingface/transformers/issues/31922
            device="cuda"
            if config.gpu_layers > 0
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu",
            framework="pt",
            model_kwargs=model_kwargs,
        )

    def generate(self, prompt: str) -> str:
        res = self._pipe(
            prompt,
            max_new_tokens=self._config.max_new_tokens,
            repetition_penalty=self._config.repetition_penalty,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            temperature=self._config.temperature,
            do_sample=True,
        )
        output: str = res[0]["generated_text"]
        return output

    def create_chat_completion(
        self, messages: list[ChatMessage]
    ) -> tuple[ChatRole, str]:
        msg = self._pipe(
            messages,
            max_new_tokens=self._config.max_new_tokens,
            repetition_penalty=self._config.repetition_penalty,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            temperature=self._config.temperature,
            do_sample=True,
        )[0]["generated_text"][-1]
        return (msg["role"], msg["content"].strip())


class RemoteLLM(LLM):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        if self._config.endpoint is None:
            msg = "Endpoint is required for remote LLM"
            raise ValueError(msg)

        self._ep: ModelEndpoint = self._config.endpoint
        self._prompt_field = "prompt"
        self._header = {"Content-Type": "application/json"}
        if self._config.endpoint.key:
            self._header["Authorization"] = (
                f"Bearer {self._config.endpoint.key.get_secret_value()}"
            )

        if self._config.endpoint.type == "openai":
            self.msg_template = {
                "model": self._config.name,
                "temperature": self._config.temperature,
                "frequency_penalty": self._config.repetition_penalty,
                "max_completion_tokens": self._config.max_new_tokens,
                "n": 1,
                "stop": self._config.stop,
                "top_p": self._config.top_p,
            }
        elif self._config.endpoint.type == "llama.cpp":
            self.msg_template = {
                "temperature": self._config.temperature,
                "top_k": self._config.top_k,
                "top_p": self._config.top_p,
                "repeat_penalty": self._config.repetition_penalty,
                "n_predict": self._config.max_new_tokens,
                "stop": self._config.stop or [],
            }
        else:
            msg = f"Unknown endpoint type: {self._config.endpoint.type}"
            raise ValueError(msg)

    def generate(self, prompt: str) -> str:
        import json

        import requests

        if self._config.endpoint and self._config.endpoint.type != "llama.cpp":
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
    if config.endpoint:
        return RemoteLLM(config)
    if config.file:
        return LlamaCppLLM(config)
    return TransformerLLM(config)
