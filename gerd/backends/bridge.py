"""The Bridge connects backend and frontend services directly for local use."""

import logging
from typing import Dict, List, Optional

from typing_extensions import override

from gerd.config import load_gen_config, load_qa_config
from gerd.gen import GenerationService
from gerd.qa import QAService
from gerd.transport import QAQuestion

from ..transport import (
    DocumentSource,
    GenResponse,
    PromptConfig,
    QAAnalyzeAnswer,
    QAAnswer,
    QAFileUpload,
    QAModesEnum,
    QAQuestion,
    Transport,
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Bridge(Transport):
    """Direct connection between backend services and frontend.

    Frontends that make use of the [`Transport`][gerd.transport.Transport] abstraction
    can use `Bridge` to get accessto generation and QA services directly. This is useful
    for local use cases where the frontend and backend are running in the same process.
    """

    def __init__(self) -> None:
        """The services associated with the bridge are initialized lazily."""
        super().__init__()
        self._qa: Optional[QAService] = None
        self._gen: Optional[GenerationService] = None

    @property
    def qa(self) -> QAService:
        """Get the QA service instance. It will be created if it does not exist."""
        if self._qa is None:
            self._qa = QAService(load_qa_config())
        return self._qa

    @property
    def gen(self) -> GenerationService:
        """Get the generation service instance.

        It will be created if it does not exist.
        """
        if self._gen is None:
            self._gen = GenerationService(load_gen_config())
        return self._gen

    @override
    def qa_query(self, question: QAQuestion) -> QAAnswer:
        return self.qa.query(question)

    @override
    def analyze_query(self) -> QAAnalyzeAnswer:
        return self.qa.analyze_query()

    @override
    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        return self.qa.analyze_mult_prompts_query()

    @override
    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        return self.qa.db_query(question)

    @override
    def db_embedding(self, question: QAQuestion) -> List[float]:
        return self.qa.db_embedding(question)

    @override
    def add_file(self, file: QAFileUpload) -> QAAnswer:
        return self.qa.add_file(file)

    @override
    def remove_file(self, file_name: str) -> QAAnswer:
        return self.qa.remove_file(file_name)

    @override
    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        return self.gen.set_prompt_config(config)

    @override
    def get_gen_prompt(self) -> PromptConfig:
        return self.gen.get_prompt_config()

    @override
    def set_qa_prompt(self, config: PromptConfig, qa_mode: QAModesEnum) -> QAAnswer:
        return self.qa.set_prompt_config(config, qa_mode)

    @override
    def get_qa_prompt(self, qa_mode: QAModesEnum) -> PromptConfig:
        return self.qa.get_prompt_config(qa_mode)

    @override
    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        return self.gen.generate(parameters)
