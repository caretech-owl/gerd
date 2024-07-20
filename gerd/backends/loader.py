import abc
import logging

from gerd.models.model import ModelConfig

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class LLM:
    @abc.abstractmethod
    def __init__(self, config: ModelConfig) -> None:
        pass

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class MockLLM(LLM):
    def __init__(self, config: ModelConfig) -> None:
        self.ret_value = "MockLLM"
        pass

    def generate(self, prompt: str) -> str:
        return self.ret_value


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
        output = self._model(
            prompt,
            stop=self._config.stop,
            max_tokens=self._config.max_new_tokens,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            temperature=self._config.temperature,
            repeat_penalty=self._config.repetition_penalty,
        )["choices"][0]["text"]

        return output


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
            device="cuda" if config.gpu_layers > 0 else None,
            framework="pt",
            model_kwargs=model_kwargs,
        )

    def generate(self, prompt: str) -> str:
        output = self._pipe(
            prompt,
            max_new_tokens=self._config.max_new_tokens,
            repetition_penalty=self._config.repetition_penalty,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            temperature=self._config.temperature,
            do_sample=True,
        )[0]["generated_text"]

        return output


class RemoteLLM(LLM):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._prompt_field = "prompt"
        self.msg_template = {
            "temperature": self._config.temperature,
            "top_k": self._config.top_k,
            "top_p": self._config.top_p,
            "repeat_penalty": self._config.repetition_penalty,
            "n_predict": self._config.max_new_tokens,
            "stop": self._config.stop or [],
        }

    def generate(self, prompt: str) -> str:
        import json

        import requests

        self.msg_template[self._prompt_field] = prompt
        res = requests.post(
            self._config.endpoint.url + "/completion",
            headers={"Content-Type": "application/json"},
            data=json.dumps(self.msg_template),
            timeout=300,
        )
        if res.status_code == 200:
            return res.json()["content"]
        else:
            _LOGGER.warning("Server returned error code %d", res.status_code)
        return ""


def load_model_from_config(config: ModelConfig) -> LLM:
    if config.endpoint:
        return RemoteLLM(config)
    if config.file:
        return LlamaCppLLM(config)
    return TransformerLLM(config)