import pytest

from gerd.backends.loader import LlamaCppLLM, TransformerLLM
from gerd.config import CONFIG
from gerd.gen.generation_service import GenerationService


@pytest.mark.skip("Should only be run manually")
def test_fail() -> None:
    pytest.fail("This test should not be executed.")

@pytest.mark.skip("Should only be run manually")
def test_generation_transformers() -> None:
    config = CONFIG.model_copy(deep=True)
    config.gen.model.name = "distilbert/distilgpt2"
    config.gen.model.file = None
    config.gen.model.prompt.text = "Say {the_word}!"
    config.gen.model.temperature = 0.01
    config.gen.model.max_new_tokens = 10
    gen = GenerationService(config.gen)
    assert isinstance(gen._model, TransformerLLM)  # noqa: SLF001
    res = gen.generate({'the_word': 'please'}, add_prompt=True)
    assert res.status == 200
    assert res.prompt == "Say please!"
    assert res.text

@pytest.mark.skip("Should only be run manually")
def test_generation_llama_cpp() -> None:
    config = CONFIG.model_copy(deep=True)
    config.gen.model.name = "bartowski/Phi-3.1-mini-4k-instruct-GGUF"
    config.gen.model.file = "Phi-3.1-mini-4k-instruct-IQ2_XS.gguf"
    config.gen.model.prompt.text = "Say {the_word}!"
    config.gen.model.max_new_tokens = 10
    gen = GenerationService(config.gen)
    assert isinstance(gen._model, LlamaCppLLM)  # noqa: SLF001
    res = gen.generate({'the_word': 'please'}, add_prompt=True)
    assert res.status == 200
    assert res.prompt == "Say please!"
    assert res.text
