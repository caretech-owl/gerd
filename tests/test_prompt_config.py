"""Tests for the PromptConfig class."""

from pathlib import Path

from gerd.models.model import PromptConfig


def test_text() -> None:
    """Tests that a prompt config is correctly loaded from a text string."""
    config = PromptConfig(text="Hello, World!")
    assert config.text == "Hello, World!"


def test_path() -> None:
    """Tests that a prompt config is correctly loaded from a text file."""
    config = PromptConfig(path="tests/data/prompt.txt")
    assert config.text == Path("tests/data/prompt.txt").read_text(encoding="utf-8")
    assert config.template is None


def test_jinja() -> None:
    """Tests that a Jinja template is loaded correctly."""
    config = PromptConfig(path="tests/data/simple_prompt.jinja2")
    assert "Filled Text" in config.template.render(task="Filled Text")


def test_text_parameters() -> None:
    """Tests that prompt parameters are extracted from a text template."""
    config = PromptConfig(text="A {value} prompt {test}.")
    assert config.parameters == ["test", "value"]


def test_jinja_parameters() -> None:
    """Tests that prompt parameters are extracted from a Jinja template."""
    config = PromptConfig(text="A {{ value }} prompt {{ test }}", is_template=True)
    assert config.parameters == ["test", "value"]


def test_prompt_duplicate() -> None:
    """Tests whether prompt parameters are only returned once."""
    tmp = """A {value} prompt {test} and another duplicate {value} and {foo}
    in between as well as {test} and {value} again."""
    prompt = PromptConfig(text=tmp)
    assert prompt.parameters == {"test", "value", "foo"}
