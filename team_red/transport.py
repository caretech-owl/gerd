from enum import Enum
from typing import Dict, List, Optional, Protocol

from pydantic import BaseModel


class PromptConfig(BaseModel):
    text: str
    parameters: Optional[Dict[str, str]] = None


class GenResponse(BaseModel):
    status: int = 200
    text: str = ""
    error_msg: str = ""


class QAQuestion(BaseModel):
    question: str


class DocumentSource(BaseModel):
    content: str
    name: str
    page: int


class QAAnswer(BaseModel):
    status: int = 200
    error_msg: str = ""
    answer: str = ""
    sources: List[DocumentSource] = []


class FileTypes(Enum):
    TEXT = "txt"
    PDF = "pdf"


class QAFileUpload(BaseModel):
    data: bytes
    type: FileTypes


class Transport(Protocol):
    def qa_query(self, query: QAQuestion) -> QAAnswer:
        pass

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        pass

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        pass

    def get_gen_rompt(self) -> PromptConfig:
        pass

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        pass
