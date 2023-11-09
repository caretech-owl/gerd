import pytest

from team_red.config import CONFIG
from team_red.qa import QAService
from team_red.transport import FileTypes, QAFileUpload, QAQuestion


@pytest.fixture()
def qa_service() -> QAService:
    return QAService()


@pytest.fixture()
def qa_service_cajal(qa_service: QAService, cajal_txt: bytes) -> QAService:
    request = QAFileUpload(data=cajal_txt, type=FileTypes.TEXT)
    qa_service.add_file(request)
    return qa_service


def test_init() -> None:
    qa = QAService()


def test_query_without_document(qa_service: QAService) -> None:
    assert CONFIG.data.embedding.db_path == ""
    res = qa_service.query(QAQuestion(question="This should return a 404"))
    assert res.status == 404


def test_load(qa_service: QAService, cajal_txt: bytes) -> None:
    request = QAFileUpload(data=cajal_txt, type=FileTypes.TEXT)
    qa_service.add_file(request)


def test_query_post_load(qa_service_cajal: QAService) -> None:
    res = qa_service_cajal.query(QAQuestion(question="Wer ist der Patient?"))
    assert res.status == 200
    assert "Cajal" in res.answer
