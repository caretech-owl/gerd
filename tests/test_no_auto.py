import pytest

from gerd.backends.loader import LlamaCppLLM, TransformerLLM
from gerd.config import load_gen_config
from gerd.gen.generation_service import GenerationService


@pytest.mark.skip("Should only be run manually")
def test_fail() -> None:
    pytest.fail("This test should not be executed.")


@pytest.mark.skip("Should only be run manually")
def test_manual_generation_transformers() -> None:
    config = load_gen_config()
    config.model.name = "distilbert/distilgpt2"
    config.model.file = None
    config.model.prompt_config.text = "Say {the_word}!"
    config.model.temperature = 0.01
    config.model.max_new_tokens = 10
    gen = GenerationService(config)
    assert isinstance(gen._model, TransformerLLM)  # noqa: SLF001
    res = gen.generate({"the_word": "please"}, add_prompt=True)
    assert res.status == 200
    assert res.prompt == "Say please!"
    assert res.text


@pytest.mark.skip("Should only be run manually")
def test_manual_generation_llama_cpp() -> None:
    config = load_gen_config()
    config.model.name = "bartowski/Phi-3.1-mini-4k-instruct-GGUF"
    config.model.file = "Phi-3.1-mini-4k-instruct-IQ2_M.gguf"
    config.model.prompt_config.text = "Say {the_word}!"
    config.model.max_new_tokens = 10
    gen = GenerationService(config)
    assert isinstance(gen._model, LlamaCppLLM)  # noqa: SLF001
    res = gen.generate({"the_word": "please"}, add_prompt=True)
    assert res.status == 200
    assert res.prompt == "Say please!"
    assert res.text
