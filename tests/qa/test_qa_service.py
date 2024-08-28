from datetime import datetime
from pathlib import Path
from typing import List

import pytest
from pytest_mock import MockerFixture

from gerd.backends.loader import MockLLM
from gerd.models.qa import QAConfig
from gerd.qa import QAService
from gerd.transport import QAFileUpload, QAQuestion

QA_PATH = Path(__file__).resolve().parent


@pytest.fixture
def qa_service(mocker: MockerFixture, qa_config: QAConfig) -> QAService:
    _ = mocker.patch(
        "gerd.backends.loader.load_model_from_config",
        return_value=MockLLM(qa_config.model),
    )
    return QAService(qa_config)


@pytest.fixture
def qa_service_cajal(qa_service: QAService, cajal_txt: bytes) -> QAService:
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
    return request.param


test_files: List[str] = ["Cajal.txt", "Boeck.txt", "Baastrup.txt"]


@pytest.fixture(params=test_files)
def test_file(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def qa_service_file(
    qa_service: QAService, files_txt: bytes, test_file: str
) -> QAService:
    request = QAFileUpload(data=files_txt, name=test_file)
    qa_service.add_file(request)
    return qa_service


def test_init(mocker: MockerFixture, qa_config: QAConfig) -> None:
    loader = mocker.patch(
        "gerd.backends.loader.load_model_from_config",
        return_value=MockLLM(qa_config.model),
    )
    qa = QAService(qa_config)
    assert loader.called


def test_query_without_document(qa_service: QAService) -> None:
    assert qa_service._config.embedding.db_path == ""  # noqa: SLF001
    res = qa_service.query(QAQuestion(question="This should return a 404"))
    assert res.status == 404


def test_load(qa_service: QAService, cajal_txt: bytes) -> None:
    request = QAFileUpload(data=cajal_txt, name="Cajal.txt")
    qa_service.add_file(request)


def test_query(qa_service_cajal: QAService) -> None:
    res = qa_service_cajal.query(QAQuestion(question="Wer ist der Patient?"))
    assert res.status == 200
    assert res.answer


def test_db_query(qa_service_cajal: QAService, qa_config: QAConfig) -> None:
    """
    Test the db search mode
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


def test_queries(qa_service_file: QAService, query_questions: str) -> None:
    """
    Test the search mode
    """
    res = qa_service_file.query(QAQuestion(question=query_questions))

    txt_out = (
        "=" * 10
        + "\n"
        + "["
        + str(datetime.today())
        + "]\n"
        + "Question: "
        + query_questions
        + "\n"
        + "Answer: "
        + res.answer
        + "\n"
        + "Model response: "
        + res.response
        + "\n"
        + "Sources: "
        + (" ".join(doc.content for doc in res.sources))
        + "\n"
        + "=" * 10
        + "\n"
    )
    assert res.status == 200


def test_analyze_queries(qa_service_file: QAService) -> None:
    """
    Test the analyze mode
    """
    res = qa_service_file.analyze_query()

    # remove unwanted fields from answer
    qa_res_dic = {
        key: value
        for key, value in vars(res).items()
        if value is not None
        and value != ""
        and key not in res.__class__.__dict__
        and key != "sources"
        and key != "status"
    }
    qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())

    txt_out = (
        "=" * 10
        + "\n"
        + "["
        + str(datetime.today())
        + "]\n"
        + "Answer: "
        + qa_res_str
        + "\n"
        + "Model response: "
        + res.response
        + "\n"
        + "Prompt: "
        + res.prompt
        + "\n"
        + "Sources: "
        + ("; ".join(f"{doc.query}={doc.content}\n\n" for doc in res.sources))
        + "\n"
        + "=" * 10
        + "\n"
    )
    assert res.status == 200


def test_analyze_mult_prompts_queries(qa_service_file: QAService) -> None:
    """
    Test the analyze mult prompts mode
    """
    res = qa_service_file.analyze_mult_prompts_query()

    # remove unwanted fields from answer
    qa_res_dic = {
        key: value
        for key, value in vars(res).items()
        if value is not None
        and value != ""
        and key not in res.__class__.__dict__
        and key != "sources"
        and key != "status"
    }
    qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())

    # write result to file
    txt_out = (
        "=" * 10
        + "\n"
        + "["
        + str(datetime.today())
        + "]\n"
        + "Answer: "
        + qa_res_str
        + "\n"
        + "Model response: "
        + res.response
        + "\n"
        + "Sources: "
        + ("; ".join(f"{doc.query}={doc.content}\n\n" for doc in res.sources))
        + "\n"
        + "=" * 10
        + "\n"
    )

    assert res.status == 200
