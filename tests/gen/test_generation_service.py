import pytest
from pytest_mock import MockerFixture

from gerd.backends.loader import MockLLM
from gerd.gen.generation_service import GenerationService
from gerd.models.gen import GenerationConfig
from gerd.transport import PromptConfig


@pytest.fixture
def gen_service(
    mocker: MockerFixture, generation_config: GenerationConfig
) -> GenerationService:
    _ = mocker.patch(
        "gerd.backends.loader.load_model_from_config",
        return_value=MockLLM(generation_config.model),
    )
    return GenerationService(generation_config)


def test_init(mocker: MockerFixture, generation_config: GenerationConfig) -> None:
    loader = mocker.patch(
        "gerd.backends.loader.load_model_from_config",
        return_value=MockLLM(generation_config.model),
    )
    gen = GenerationService(generation_config)
    assert loader.called


def test_get_prompt(
    gen_service: GenerationService, generation_config: GenerationConfig
) -> None:
    prompt = gen_service.get_prompt_config()
    assert prompt
    assert prompt.text == generation_config.model.prompt_config.text


def test_set_prompt(gen_service: GenerationService) -> None:
    tmp = "A {value} prompt {test}."
    prompt = gen_service.set_prompt_config(PromptConfig(text=tmp))
    assert "value" in prompt.parameters
    assert "test" in prompt.parameters
    assert gen_service.get_prompt_config().text == tmp


def test_prompt_duplicate(gen_service: GenerationService) -> None:
    tmp = """A {value} prompt {test} and another duplicate {value} and {foo}
    in between as well as {test} and {value} again."""
    prompt = gen_service.set_prompt_config(PromptConfig(text=tmp))
    assert len(prompt.parameters) == len(set(prompt.parameters))


def test_generate(gen_service: GenerationService) -> None:
    gen_service.set_prompt_config(
        PromptConfig(text="Schreibe einen Brief an Herrn {name}.")
    )
    response = gen_service.generate({"name": "Cajal"})
    assert response.status == 200
    assert response.error_msg == ""
    assert response.text


def test_jinja_prompt(gen_service: GenerationService) -> None:
    _ = gen_service.set_prompt_config(
        PromptConfig(path="tests/data/simple_prompt.jinja2")
    )
    response = gen_service.generate(
        {"task": "Halte dich so kurz wie möglich."}, add_prompt=True
    )
    assert response.status == 200
    assert response.error_msg == ""
    assert "Halte dich so kurz wie möglich." in response.prompt
