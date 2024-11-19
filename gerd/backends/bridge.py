import logging
from typing import Dict, List, Optional

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
    def __init__(self) -> None:
        super().__init__()
        self._qa: Optional[QAService] = None
        self._gen: Optional[GenerationService] = None

    @property
    def qa(self) -> QAService:
        if self._qa is None:
            self._qa = QAService(load_qa_config())
        return self._qa

    @property
    def gen(self) -> GenerationService:
        if self._gen is None:
            self._gen = GenerationService(load_gen_config())
        return self._gen

    def qa_query(self, question: QAQuestion) -> QAAnswer:
        return self.qa.query(question)

    def analyze_query(self) -> QAAnalyzeAnswer:
        return self.qa.analyze_query()

    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        return self.qa.analyze_mult_prompts_query()

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        return self.qa.db_query(question)

    def db_embedding(self, question: QAQuestion) -> List[float]:
        return self.qa.db_embedding(question)

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        return self.qa.add_file(file)

    def remove_file(self, file_name: str) -> QAAnswer:
        return self.qa.remove_file(file_name)

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        return self.gen.set_prompt_config(config)

    def get_gen_prompt(self) -> PromptConfig:
        return self.gen.get_prompt_config()

    def set_qa_prompt(self, config: PromptConfig, qa_mode: QAModesEnum) -> PromptConfig:
        return self.qa.set_prompt_config(config, qa_mode)

    def get_qa_prompt(self, qa_mode: QAModesEnum) -> PromptConfig:
        return self.qa.get_prompt_config(qa_mode)

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        return self.gen.generate(parameters)

    def gen_continue(self, parameters: Dict[str, str]) -> GenResponse:
        return self.gen.gen_continue(parameters)
