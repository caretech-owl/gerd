from pathlib import Path

from gerd.models.model import PromptConfig


def test_text() -> None:
    config = PromptConfig(text="Hello, World!")
    assert config.text == "Hello, World!"


def test_path() -> None:
    config = PromptConfig(path="tests/data/prompt.txt")
    assert config.text == Path("tests/data/prompt.txt").read_text()
    assert config.template is None


def test_jinja() -> None:
    config = PromptConfig(path="tests/data/simple_prompt.jinja")
    assert "Filled Text" in config.template.render(task="Filled Text")


def test_text_parameters() -> None:
    config = PromptConfig(text="A {value} prompt {test}.")
    assert config.parameters == ["test", "value"]


def test_jinja_parameters() -> None:
    config = PromptConfig(text="A {{ value }} prompt {{ test }}", is_template=True)
    assert config.parameters == ["test", "value"]
