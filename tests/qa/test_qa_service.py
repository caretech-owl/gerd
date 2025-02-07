"""Test the QAService class."""

from datetime import datetime
from pathlib import Path
from typing import List

import pytest
from pytest_mock import MockerFixture

from gerd.loader import MockLLM
from gerd.models.model import PromptConfig
from gerd.models.qa import QAConfig
from gerd.qa import QAService
from gerd.transport import QAAnswer, QAFileUpload, QAModesEnum, QAQuestion

QA_PATH = Path(__file__).resolve().parent


@pytest.fixture
def qa_service(mocker: MockerFixture, qa_config: QAConfig) -> QAService:
    """A fixture for the QAService class.

    Patches the load_model_from_config function to return a MockLLM instance.
    Using actual LLM is a bit too much for unit tests.

    Parameters:
        mocker: The mocker fixture
        qa_config: The QA configuration

    Returns:
        The (mocked) QAService instance
    """
    _ = mocker.patch(
        "gerd.loader.load_model_from_config",
        return_value=MockLLM(qa_config.model),
    )
    return QAService(qa_config)


@pytest.fixture
def qa_service_cajal(qa_service: QAService, cajal_txt: bytes) -> QAService:
    """A fixture for the QAService class with a GRASCCO document preloaded.

    Parameters:
        qa_service: The QAService instance
        cajal_txt: The fixture that loads the Cajal.txt file

    Returns:
        A QAService instance with the Cajal.txt file preloaded
    """
    request = QAFileUpload(data=cajal_txt, name="Cajal.txt")
    qa_service.add_file(request)
    return qa_service


test_questions: List[str] = [
    "Wie heißt der Patient?",
    "Wann hat der Patient Geburtstag?",
    "Wie heißt der behandelnde Arzt?",
    "Wann wurde der Patient bei uns entlassen?",
    "Wann wurde der Patient bei uns aufgenommen?",
    "Wo wohnt der Patient?",
    "Wie heißt der Hausarzt des Patienten?",
    "Wie heißt die behandelnde Einrichtung?",
    "Welche Medikamente bekommt der Patient?",
    "Welche Diagnose wurde gestellt?",
]


@pytest.fixture(params=test_questions)
def query_questions(request: pytest.FixtureRequest) -> str:
    """A fixture to execute a test for each question in the list.

    Parameters:
        request: The request object passed by the fixture decorator

    Returns:
        The question to be tested
    """
    return request.param


@pytest.fixture
def qa_service_file(qa_service: QAService, test_files: tuple[str, bytes]) -> QAService:
    """A fixture for the QAService class with a file preloaded.

    Parameters:
        qa_service: QAService fixture
        test_files: The fixture that loads the test files

    Returns:
        The QAService instance with the file preloaded
    """
    request = QAFileUpload(data=test_files[1], name=test_files[0])
    qa_service.add_file(request)
    return qa_service


def test_init(mocker: MockerFixture, qa_config: QAConfig) -> None:
    """Test the initialization of the QAService class.

    Parameters:
        mocker: The mocker fixture
        qa_config: The QA configuration fixture
    """
    loader = mocker.patch(
        "gerd.loader.load_model_from_config",
        return_value=MockLLM(qa_config.model),
    )
    qa = QAService(qa_config)
    assert loader.called


def test_query_without_document(qa_service: QAService) -> None:
    """Test the query method without any document loaded.

    Parameters:
        qa_service: The QAService fixture
    """
    assert qa_service.config.embedding.db_path == ""  # noqa: SLF001
    res = qa_service.query(QAQuestion(question="This should return a 404"))
    assert res.status == 404


def test_load(qa_service: QAService, cajal_txt: bytes) -> None:
    """Test the load method of the QAService class.

    Parameters:
        qa_service: The QAService fixture
        cajal_txt: The fixture that loads the Cajal.txt file
    """
    request = QAFileUpload(data=cajal_txt, name="Cajal.txt")
    qa_service.add_file(request)


def test_query(qa_service_cajal: QAService) -> None:
    """Test the query method of the QAService class.

    Parameters:
        qa_service_cajal: The QAService fixture with a document loaded
    """
    res = qa_service_cajal.query(QAQuestion(question="Wer ist der Patient?"))
    assert res.status == 200
    assert res.response


def test_db_query(qa_service_cajal: QAService, qa_config: QAConfig) -> None:
    """Test the db_query method of the QAService class.

    Parameters:
        qa_service_cajal: The QAService fixture with a document loaded
        qa_config: The QA configuration fixture
    """
    q = QAQuestion(question="Wer ist der Patient?", max_sources=3)
    res = qa_service_cajal.db_query(q)
    assert len(res) == q.max_sources
    assert res[0].name == "Cajal.txt"
    assert len(res[0].content) <= qa_config.embedding.chunk_size
    assert res[0] != res[1]  # return values should not be the same
    q = QAQuestion(question="Wie heißt das Krankenhaus", max_sources=1)
    res = qa_service_cajal.db_query(q)
    assert len(res) == q.max_sources
    assert "Diakonissenkrankenhaus Berlin" in res[0].content


def test_set_qa_prompt(qa_service: QAService) -> None:
    """Test the set_qa_prompt method of the QAService class.

    Parameters:
        qa_service: The QAService fixture
    """
    res: QAAnswer = qa_service.set_prompt_config(
        PromptConfig(text="This is a test prompt."), qa_mode=QAModesEnum.SEARCH
    )
    assert res.status == 200
    assert "{context}" in res.error_msg
    assert "{question}" in res.error_msg
    res = qa_service.set_prompt_config(
        PromptConfig(text="This is a test prompt with {context}."),
        qa_mode=QAModesEnum.SEARCH,
    )
    assert res.status == 200
    assert "{context}" not in res.error_msg
    assert "{question}" in res.error_msg
    res = qa_service.set_prompt_config(
        PromptConfig(text="This is a test prompt with a {question}."),
        qa_mode=QAModesEnum.SEARCH,
    )
    assert res.status == 200
    assert "{context}" in res.error_msg
    assert "{question}" not in res.error_msg
    res = qa_service.set_prompt_config(
        PromptConfig(text="This is a test prompt with {question} and {context}."),
        qa_mode=QAModesEnum.SEARCH,
    )
    assert res.status == 200
    assert res.error_msg == ""
