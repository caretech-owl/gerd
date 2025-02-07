"""Tests for the GenerationService class."""

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
    """A fixture that returns a GenerationService instance.

    Parameters:
        mocker: The mocker fixture
        generation_config: The generation configuration fixture

    Returns:
        A GenerationService instance
    """
    _ = mocker.patch(
        "gerd.backends.loader.load_model_from_config",
        return_value=MockLLM(generation_config.model),
    )
    return GenerationService(generation_config)


def test_init(mocker: MockerFixture, generation_config: GenerationConfig) -> None:
    """Test the initialization of the GenerationService class.

    Parameters:
        mocker: The mocker fixture
        generation_config: The generation configuration fixture
    """
    loader = mocker.patch(
        "gerd.backends.loader.load_model_from_config",
        return_value=MockLLM(generation_config.model),
    )
    _ = GenerationService(generation_config)
    assert loader.called


def test_get_prompt(
    gen_service: GenerationService, generation_config: GenerationConfig
) -> None:
    """Test the get_prompt_config method of the GenerationService class.

    Parameters:
        gen_service: The GenerationService fixture
        generation_config: The generation configuration fixture
    """
    prompt = gen_service.get_prompt_config()
    assert prompt
    assert prompt.text == generation_config.model.prompt_config.text


def test_set_prompt(gen_service: GenerationService) -> None:
    """Test the set_prompt_config method of the GenerationService class.

    Parameters:
        gen_service: The GenerationService fixture
    """
    tmp = "A {value} prompt {test}."
    prompt = gen_service.set_prompt_config(PromptConfig(text=tmp))
    assert "value" in prompt.parameters
    assert "test" in prompt.parameters
    assert gen_service.get_prompt_config().text == tmp


def test_generate(gen_service: GenerationService) -> None:
    """Test the generate method of the GenerationService class.

    Parameters:
        gen_service: The GenerationService fixture
    """
    gen_service.set_prompt_config(
        PromptConfig(text="Schreibe einen Brief an Herrn {name}.")
    )
    response = gen_service.generate({"name": "Cajal"})
    assert response.status == 200
    assert response.error_msg == ""
    assert response.text
