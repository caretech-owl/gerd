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


def pytest_addoption(parser):
    parser.addoption(
        NO_SKIP_OPTION,
        action="store_true",
        default=False,
        help="also run skipped tests",
    )


def pytest_collection_modifyitems(config, items: List[Any]):
    if config.getoption(NO_SKIP_OPTION):
        for test in items:
            test.own_markers = [
                marker
                for marker in test.own_markers
                if marker.name not in ("skip", "skipif")
            ]


@pytest.fixture(scope="session")
def cajal_txt() -> bytes:
    p = Path(GRASCCO_PATH, "Cajal.txt")
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        data = f.read()
    return data


@pytest.fixture
def files_txt(test_file: str) -> bytes:
    p = Path(GRASCCO_PATH, test_file)
    assert p.exists()
    with p.open("r", encoding="utf-8-sig") as f:
        data = f.read()
    return data


@pytest.fixture
def qa_config() -> QAConfig:
    p = Path(PROJECT_DIR, "config", "qa_default.yml")
    assert p.exists()
    config = QAConfig.model_validate(safe_load(p.read_text()))
    config.embedding.db_path = ""
    return config


@pytest.fixture
def generation_config() -> GenerationConfig:
    p = Path(PROJECT_DIR, "config", "gen_default.yml")
    assert p.exists()
    return GenerationConfig.model_validate(safe_load(p.read_text()))
