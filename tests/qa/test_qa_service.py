import pytest

from team_red.config import CONFIG
from team_red.qa import QAService
from team_red.transport import FileTypes, QAFileUpload, QAQuestion


@pytest.fixture()
def qa_service() -> QAService:
    return QAService(CONFIG.qa)


@pytest.fixture()
def qa_service_cajal(qa_service: QAService, cajal_txt: bytes) -> QAService:
    request = QAFileUpload(data=cajal_txt, name="Cajal.txt")
    qa_service.add_file(request)
    return qa_service


def test_init() -> None:
    qa = QAService(CONFIG.qa)


def test_query_without_document(qa_service: QAService) -> None:
    assert CONFIG.qa.embedding.db_path == ""
    res = qa_service.query(QAQuestion(question="This should return a 404"))
    assert res.status == 404


def test_load(qa_service: QAService, cajal_txt: bytes) -> None:
    request = QAFileUpload(data=cajal_txt, name="Cajal.txt")
    qa_service.add_file(request)


def test_query(qa_service_cajal: QAService) -> None:
    res = qa_service_cajal.query(QAQuestion(question="Wer ist der Patient?"))
    assert res.status == 200
    assert res.answer


def test_db_query(qa_service_cajal: QAService) -> None:
    q = QAQuestion(question="Wer ist der Patient?")
    res = qa_service_cajal.db_query(q)
    assert len(res) == q.max_sources
    assert res[0].name == "Cajal.txt"
    assert len(res[0].content) <= CONFIG.qa.embedding.chunk_size
    q = QAQuestion(question="Wie heißt das Krankenhaus", max_sources=1)
    res = qa_service_cajal.db_query(q)
    assert len(res) == q.max_sources
    assert "Diakonissenkrankenhaus Berlin" in res[0].content
