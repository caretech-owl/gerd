"""Module to define the transport protocol.

The transport protocol is used to connect the backend and frontend services.
Implemetations of the transport protocol can be found in the
[`gerd.backends`][gerd.backends] module.
"""

from enum import Enum
from typing import Dict, List, Protocol

from pydantic import BaseModel

from gerd.models.model import PromptConfig


class GenResponse(BaseModel):
    """Dataclass to hold a response from the generation service."""

    status: int = 200
    """The status code of the response."""
    text: str = ""
    """The generated text if the status code is 200."""
    error_msg: str = ""
    """The error message if the status code is not 200."""
    prompt: str | None = None
    """The custom prompt that was used to generate the text."""


class QAQuestion(BaseModel):
    """Dataclass to hold a question for the QA service."""

    question: str
    """The question to ask the QA service."""
    search_strategy: str = "similarity"
    """The search strategy to use."""
    max_sources: int = 3
    """The maximum number of sources to return."""


# Dataclass to hold a docsource
class DocumentSource(BaseModel):
    """Dataclass to hold a document source."""

    query: str
    """The query that was used to find the document."""
    content: str
    """The content of the document."""
    name: str
    """The name of the document."""
    page: int
    """The page of the document."""


# Dataclass to hold a QAAnswer
class QAAnswer(BaseModel):
    """Dataclass to hold an answer from the QA service."""

    status: int = 200
    """The status code of the answer."""
    error_msg: str = ""
    """The error message of the answer if the status code is not 200."""
    sources: List[DocumentSource] = []
    """The sources of the answer."""
    response: str = ""
    """The response of the answer."""


class QAAnalyzeAnswer(BaseModel):
    """Dataclass to hold an answer from the predefined queries to the QA service."""

    status: int = 200
    """The status code of the answer."""
    error_msg: str = ""
    """The error message of the answer if the status code is not 200."""
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
    """Enum to hold all supported QA modes."""

    NONE = 0
    """No mode."""
    SEARCH = 1
    """Search mode."""
    ANALYZE = 2
    """Analyze mode."""
    ANALYZE_MULT_PROMPTS = 3
    """Analyze multiple prompts mode."""


class FileTypes(Enum):
    """Enum to hold all supported file types."""

    TEXT = "txt"
    """Text file type."""
    PDF = "pdf"
    """PDF file type."""


class QAFileUpload(BaseModel):
    """Dataclass to hold a file upload."""

    data: bytes
    """The file data."""
    name: str
    """The name of the file."""


class QAPromptConfig(BaseModel):
    """Prompt configuration for the QA service."""

    config: PromptConfig
    """The prompt configuration."""
    mode: QAModesEnum
    """The mode to set the prompt configuration for."""


class Transport(Protocol):
    """Transport protocol to connect backend and frontend services.

    Transport should be implemented by a class that provides the necessary methods to
    interact with the backend.
    """

    def qa_query(self, query: QAQuestion) -> QAAnswer:
        """Query the QA service with a question.

        Parameters:
            query: The question to query the QA service with.

        Returns:
           The answer from the QA service.
        """
        pass

    def analyze_query(self) -> QAAnalyzeAnswer:
        """Queries the vector store with a predefined query.

        The query should return vital information gathered
        from letters of discharge.

        Returns:
            The answer from the QA service.
        """
        pass

    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        """Queries the vector store with a set of predefined queries.

        In contrast to [`analyze_query`][gerd.transport.Transport.analyze_query],
        this method queries the vector store with multiple prompts.

        Returns:
            The answer from the QA service.
        """
        pass

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        """Queries the vector store with a question.

        Parameters:
            question: The question to query the vector store with.

        Returns:
            A list of document sources
        """
        pass

    def db_embedding(self, question: QAQuestion) -> List[float]:
        """Converts a question to an embedding.

        The embedding is defined by the vector store.

        Parameters:
            question: The question to convert to an embedding.

        Returns:
            The embedding of the question
        """
        pass

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        """Add a file to the vector store.

        The returned answer has a status code of 200 if the file was added successfully.
        Parameters:
            file: The file to add to the vector store.

        Returns:
            The answer from the QA service

        """
        pass

    def remove_file(self, file_name: str) -> QAAnswer:
        """Remove a file from the vector store.

        The returned answer has a status code of 200
        if the file was removed successfully.
        Parameters:
            file_name: The name of the file to remove from the vector store.

        Returns:
            The answer from the QA service
        """
        pass

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        """Sets the prompt configuration for the generation service.

        The prompt configuration that is returned should in most cases
        be the same as the one that was set.
        Parameters:
            config: The prompt configuration to set

        Returns:
            The prompt configuration that was set
        """
        pass

    def get_gen_prompt(self) -> PromptConfig:
        """Gets the prompt configuration for the generation service.

        Returns:
            The current prompt configuration
        """
        pass

    def set_qa_prompt(self, config: PromptConfig, qa_mode: QAModesEnum) -> QAAnswer:
        """Sets the prompt configuration for the QA service.

        Since the QA service uses multiple prompt configurations,
        the mode should be specified. For more details, see the documentation
        of [`QAService.set_prompt_config`][gerd.qa.QAService.set_prompt_config].

        Parameters:
            config: The prompt configuration to set
            qa_mode: The mode to set the prompt configuration for

        Returns:
            The answer from the QA service
        """
        pass

    def get_qa_prompt(self, qa_mode: QAModesEnum) -> PromptConfig:
        """Gets the prompt configuration for a mode of the QA service.

        Parameters:
            qa_mode: The mode to get the prompt configuration for

        Returns:
            The prompt configuration for the QA service
        """
        pass

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        """Generates text with the generation service.

        Parameters:
            parameters: The parameters to generate text with

        Returns:
            The generation result
        """
        pass
