from enum import Enum
from typing import Dict, List, Optional, Protocol

from pydantic import BaseModel


class PromptParameters(BaseModel):
    parameters: Dict[str, str]


class PromptConfig(BaseModel):
    text: str
    parameters: Optional[PromptParameters] = None


class GenResponse(BaseModel):
    text: str


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

    def generate(self, parameters: PromptParameters) -> GenResponse:
        pass
