from gerd.config import Settings, load_gen_config


def test_config() -> None:
    config = load_gen_config("tests/data/gen_test.yml")

    assert config.model.prompt["format"].path == "tests/data/prompt.txt"
    assert (
        config.model.prompt["format"].text
        == """Erste Zeile
Zweite Zeile
Dritte Zeile
Umlaute: äöü
Sonderzeichen: |$&?(())
{variableA}{variableB}
"""
    )
    assert config.model.prompt["format"].parameters == ["variableA", "variableB"]
