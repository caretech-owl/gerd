from gerd.config import Settings


def test_config() -> None:
    config = Settings(_env_file="tests/data/env.test")

    assert config.gen.model.prompt.path == "tests/data/prompt.txt"
    assert (
        config.gen.model.prompt.text
        == """Erste Zeile
Zweite Zeile
Dritte Zeile
Umlaute: äöü
Sonderzeichen: |$&?(())
{variableA}{variableB}
"""
    )
    assert config.gen.model.prompt.parameters == ["variableA", "variableB"]
