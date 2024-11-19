from enum import Enum
from typing import Dict, List, Protocol

from pydantic import BaseModel

from gerd.models.model import PromptConfig


class GenResponse(BaseModel):
    status: int = 200
    text: str = ""
    error_msg: str = ""
    prompt: str | None = None


# Dataclass to hold a QAQuestion
class QAQuestion(BaseModel):
    question: str
    search_strategy: str = "similarity"
    max_sources: int = 3


# Dataclass to hold a docsource
class DocumentSource(BaseModel):
    query: str
    content: str
    name: str
    page: int


# Dataclass to hold a QAAnswer
class QAAnswer(BaseModel):
    status: int = 200
    error_msg: str = ""
    answer: str = ""
    sources: List[DocumentSource] = []
    response: str = ""


# Dataclass to hold a QAAnalyzeAnswer
class QAAnalyzeAnswer(BaseModel):
    status: int = 200
    error_msg: str = ""
    # Section
    conclusion: str = ""
    anamnesis: str = ""
    salutation: str = ""
    admission_diagnosis: str = ""
    admission_medication: str = ""
    findings: str = ""
    echo_findings: str = ""
    discharge_diagnosis: str = ""
    discharge_medication: str = ""
    physical_examination_findings: str = ""
    laboratory: str = ""
    mix: str = ""
    risk_factor_allergy: str = ""
    recommendations: str = ""
    summary: str = ""
    # Context
    recording_date: str = ""
    recording_duration: str = ""
    attending_doctors: List[str] = []
    release_date: str = ""
    family_doctor: str = ""
    institution: str = ""
    department: str = ""
    patient_name: str = ""
    patient_date_of_birth: str = ""
    sources: List[DocumentSource] = []
    # Mediction Information
    active_ingredient: str = ""
    dosage: str = ""
    drug: str = ""
    duration: str = ""
    form: str = ""
    frequency: str = ""
    reason: str = ""
    route: str = ""
    strength: str = ""
    response: str = ""
    prompt: str = ""


# QAModes
class QAModesEnum(Enum):
    NONE = 0
    SEARCH = 1
    ANALYZE = 2
    ANALYZE_MULT_PROMPTS = 3


# All supported filetypes
class FileTypes(Enum):
    TEXT = "txt"
    PDF = "pdf"


# Dataclass to hold a fileupload
class QAFileUpload(BaseModel):
    data: bytes
    name: str


class QAPromptConfig(BaseModel):
    config: PromptConfig
    mode: QAModesEnum


class Transport(Protocol):
    def qa_query(self, query: QAQuestion) -> QAAnswer:
        pass

    def analyze_query(self) -> QAAnalyzeAnswer:
        pass

    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        pass

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        pass

    def db_embedding(self, question: QAQuestion) -> List[float]:
        pass

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        pass

    def remove_file(self, file_name: str) -> QAAnswer:
        pass

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        pass

    def get_gen_prompt(self) -> PromptConfig:
        pass

    def set_qa_prompt(self, config: PromptConfig, qa_mode: QAModesEnum) -> QAAnswer:
        pass

    def get_qa_prompt(self, qa_mode: QAModesEnum) -> PromptConfig:
        pass

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        pass

    def gen_continue(self, parameters: Dict[str, str]) -> GenResponse:
        pass
