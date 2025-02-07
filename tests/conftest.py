"""Pytest configuration parameters and fixtures used for all tests."""

from pathlib import Path
from typing import Any, List

import pytest
from typing_extensions import Final
from yaml import safe_load

from gerd.config import PROJECT_DIR
from gerd.models.gen import GenerationConfig
from gerd.models.qa import QAConfig

TEST_PATH = Path(__file__).resolve().parent
DATA_PATH = Path(TEST_PATH, "data")
GRASCCO_PATH = Path(DATA_PATH, "grascco", "raw")

NO_SKIP_OPTION: Final[str] = "--no-skip"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add a command line option to run skipped tests.

    Will be called by pytest when parsing command line options.
    """
    parser.addoption(
        NO_SKIP_OPTION,
        action="store_true",
        default=False,
        help="Also run skipped tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: List[Any]) -> None:
    """Remove the skip marker from tests if the --no-skip option is set.

    Will be called by pytest when collecting tests.
    """
    if config.getoption(NO_SKIP_OPTION):
        for test in items:
            test.own_markers = [
                marker
                for marker in test.own_markers
                if marker.name not in ("skip", "skipif")
            ]


@pytest.fixture(scope="session")
def cajal_txt() -> bytes:
    """Fixture to load the Cajal.txt file from the GRASCCO corpus.

    Returns:
        The content of the Cajal.txt
    """
    p = Path(GRASCCO_PATH, "Cajal.txt")
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        data = f.read()
    return data


test_files: List[str] = ["Cajal.txt", "Boeck.txt", "Baastrup.txt"]
"""A list of test files to be used in the tests."""


@pytest.fixture(params=test_files)
def test_file(test_file_name: pytest.FixtureRequest) -> tuple[str, bytes]:
    """Fixture to load a test file from the GRASCCO corpus.

    A test using this fixture will be run for each file in the test_files list.

    Parameters:
        test_file_name: The name of the test file

    Returns:
        A tuple containing the name of the test file and its content
    """
    p = Path(GRASCCO_PATH, test_file_name.param)
    assert p.exists()
    with p.open("r", encoding="utf-8-sig") as f:
        data = f.read()
    return test_file_name, data


@pytest.fixture
def qa_config() -> QAConfig:
    """Fixture to load a QA configuration from the test YAML file.

    Returns:
        The QA configuration
    """
    p = Path(PROJECT_DIR, "tests", "data", "qa_test.yml")
    assert p.exists()
    config = QAConfig.model_validate(safe_load(p.read_text()))
    config.embedding.db_path = ""
    return config


@pytest.fixture
def generation_config() -> GenerationConfig:
    """Fixture to load the default configuration from the config folder.

    Returns:
        The generation configuration
    """
    p = Path(PROJECT_DIR, "config", "gen_default.yml")
    assert p.exists()
    return GenerationConfig.model_validate(safe_load(p.read_text()))
