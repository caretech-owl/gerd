import pytest

from gerd.config import CONFIG
from gerd.gen.generation_service import GenerationService
from gerd.transport import PromptConfig


@pytest.fixture()
def gen_service() -> GenerationService:
    return GenerationService(CONFIG.gen)


def test_init() -> None:
    gen = GenerationService(CONFIG.gen)


def test_get_prompt(gen_service: GenerationService) -> None:
    prompt = gen_service.get_prompt()
    assert prompt
    assert prompt.text == CONFIG.gen.model.prompt.text


def test_set_prompt(gen_service: GenerationService) -> None:
    tmp = "A {value} prompt {test}."
    prompt = gen_service.set_prompt(PromptConfig(text=tmp))
    assert "value" in prompt.parameters
    assert "test" in prompt.parameters
    assert gen_service.get_prompt().text == tmp


def test_prompt_duplicate(gen_service: GenerationService) -> None:
    tmp = """A {value} prompt {test} and another duplicate {value} and {foo}
    in between as well as {test} and {value} again."""
    prompt = gen_service.set_prompt(PromptConfig(text=tmp))
    assert len(prompt.parameters) == len(set(prompt.parameters))


def test_generate(gen_service: GenerationService) -> None:
    gen_service.set_prompt(PromptConfig(text="Schreibe einen Brief an Herrn {name}."))
    response = gen_service.generate({"name": "Cajal"})
    assert response.status == 200
    assert response.error_msg == ""
    assert response.text


def test_jinja_prompt(gen_service: GenerationService) -> None:
    _ = gen_service.set_prompt(PromptConfig(path="tests/data/simple_prompt.jinja"))
    response = gen_service.generate(
        {
            "task": "Halte dich so kurz wie möglich und sag "
            "das Wort 'hallo'! Das Wort 'hallo' muss in "
            "der Ausgabe erscheinen."
        }
    )
    assert response.status == 200
    assert response.error_msg == ""
    assert "hallo" in response.text.lower()
