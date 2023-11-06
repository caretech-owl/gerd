from typing import Protocol

from pydantic import BaseModel


class QAQuestion(BaseModel):
    question: str


class QAAnswer(BaseModel):
    status: int = 200
    error_msg: str = ""
    answer: str = ""


class Interface(Protocol):
    def qa_query(query: QAQuestion) -> QAAnswer:
        pass
