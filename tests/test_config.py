"""Tests for the config module."""

from gerd.config import load_gen_config


def test_config() -> None:
    """Tests that a generation config is correctly loaded from a YAML file.

    Especially unicode characters may cause problems depending on the encoding
    used by the operating system.
    """
    config = load_gen_config("tests/data/gen_test.yml")

    assert config.model.prompt_config.path == "tests/data/prompt.txt"
    assert (
        config.model.prompt_config.text
        == """Erste Zeile
Zweite Zeile
Dritte Zeile
Umlaute: äöü
Sonderzeichen: |$&?(())
{variableA}{variableB}
"""
    )
    assert config.model.prompt_config.parameters == ["variableA", "variableB"]
