import logging
from typing import Dict, Optional

from team_red.gen import GenerationService
from team_red.qa import QAService

from ..transport import (
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

    def qa_query(self, question: QAQuestion) -> QAAnswer:
        if not self._qa:
            self._qa = QAService()
        return self._qa.query(question)

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        if not self._qa:
            self._qa = QAService()
        return self._qa.add_file(file)

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        if not self._gen:
            self._gen = GenerationService()
        return self._gen.set_prompt(config)

    def get_gen_prompt(self) -> PromptConfig:
        if self._qa:
            return self._gen.get_prompt()
        return PromptConfig(text="")

    def set_qa_prompt(self, config: PromptConfig) -> PromptConfig:
        if not self._qa:
            self._gen = QAService()
        return self._gen.set_prompt(config)

    def get_qa_prompt(self) -> PromptConfig:
        if self._qa:
            return self._qa.get_prompt()
        return PromptConfig(text="")

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        if not self._gen:
            self._gen = GenerationService()
        return self._gen.generate(parameters)
