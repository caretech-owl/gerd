import logging
from typing import Optional

from team_red.gen import GenerationService
from team_red.qa import QAService

from ..transport import (
    GenResponse,
    PromptConfig,
    PromptParameters,
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

    def get_gen_rompt(self) -> PromptConfig:
        if self._gen:
            self._gen.get_prompt()
        return PromptConfig(text="")

    def generate(self, parameters: PromptParameters) -> GenResponse:
        if not self._gen:
            self._gen = GenerationService()
        return self._gen.generate(parameters)
