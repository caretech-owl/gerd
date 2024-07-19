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
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        torch_dtypes: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }


        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config.name)
        self._model = AutoModelForCausalLM.from_pretrained(config.name)

        model_kwargs = {}
        if config.torch_dtype in torch_dtypes:
            model_kwargs["torch_dtype"] = torch_dtypes[config.torch_dtype]


        self._pipe = pipeline(
            task="text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
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


def load_model_from_config(config: ModelConfig) -> LLM:
    if config.file:
        return LlamaCppLLM(config)
    return TransformerLLM(config)
