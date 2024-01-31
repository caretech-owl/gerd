from pathlib import Path

import pytest

from team_red.config import CONFIG

TEST_PATH = Path(__file__).resolve().parent
DATA_PATH = Path(TEST_PATH, "data")
GRASCCO_PATH = Path(DATA_PATH, "grascco", "raw")


@pytest.fixture(scope="session", autouse=True)
def _set_config() -> None:
    CONFIG.qa.embedding.db_path = ""


@pytest.fixture(scope="session")
def cajal_txt() -> bytes:
    p = Path(GRASCCO_PATH, "Cajal.txt")
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        data = f.read()
    return data
