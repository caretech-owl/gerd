import logging
from typing import Dict, List, Optional

from team_red.config import CONFIG
from team_red.gen import GenerationService
from team_red.qa import QAService

from ..transport import (
    DocumentSource,
    GenResponse,
    PromptConfig,
    QAAnswer,
    QAFileUpload,
    QAQuestion,
    Transport,
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Bridge(Transport):
    def __init__(self) -> None:
        super().__init__()
        self._qa: Optional[QAService] = None
        self._gen: Optional[GenerationService] = None

    @property
    def qa(self) -> QAService:
        if self._qa is None:
            self._qa = QAService(CONFIG.qa)
        return self._qa

    @property
    def gen(self) -> GenerationService:
        if self._gen is None:
            self._gen = GenerationService(CONFIG.gen)
        return self._gen

    def qa_query(self, question: QAQuestion) -> QAAnswer:
        return self.qa.query(question)

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        return self.qa.db_query(question)

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        return self.qa.add_file(file)

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        return self.gen.set_prompt(config)

    def get_gen_prompt(self) -> PromptConfig:
        return self.gen.get_prompt()

    def set_qa_prompt(self, config: PromptConfig) -> PromptConfig:
        return self.qa.set_prompt(config)

    def get_qa_prompt(self) -> PromptConfig:
        return self.qa.get_prompt()

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        return self.gen.generate(parameters)
