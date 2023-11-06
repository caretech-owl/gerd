from typing import Dict, Optional, Protocol

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


class QAAnswer(BaseModel):
    status: int = 200
    error_msg: str = ""
    answer: str = ""


class Interface(Protocol):
    def qa_query(self, query: QAQuestion) -> QAAnswer:
        pass

    def set_prompt(self, config: PromptConfig) -> PromptConfig:
        pass

    def get_prompt(self) -> PromptConfig:
        pass

    def generate(self, parameters: PromptParameters) -> GenResponse:
        pass
