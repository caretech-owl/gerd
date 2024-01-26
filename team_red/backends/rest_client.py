import logging
from typing import Dict, List

import requests
from pydantic import TypeAdapter

from team_red.config import CONFIG
from team_red.transport import QAQuestion

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


class RestClient(Transport):
    def __init__(self) -> None:
        super().__init__()
        self._url = f"http://{CONFIG.server.host}:{CONFIG.server.port}{CONFIG.server.api_prefix}"
        self.timeout = 10

    def qa_query(self, question: QAQuestion) -> QAAnswer:
        return QAAnswer.model_validate(
            requests.post(
                f"{self._url}/qa/query",
                data=question.model_dump_json(),
                timeout=self.timeout,
            ).json()
        )

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        request = question.model_dump_json()
        _LOGGER.debug("db_query - request: %s", request)
        response = requests.post(
            f"{self._url}/qa/db_query",
            data=question.model_dump_json(),
            timeout=self.timeout,
        )
        _LOGGER.debug("db_query - response: %s", response.json())
        return TypeAdapter(List[DocumentSource]).validate_python(response.json())

    def db_embedding(self, question: QAQuestion) -> List[float]:
        request = question.model_dump_json()
        _LOGGER.debug("db_embedding - request: %s", request)
        response = requests.post(
            f"{self._url}/qa/db_embedding",
            data=question.model_dump_json(),
            timeout=self.timeout,
        )
        _LOGGER.debug("db_embedding - response: %s", response.json())
        return TypeAdapter(List[float]).validate_python(response.json())

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        return QAAnswer.model_validate(
            requests.post(
                f"{self._url}/qa/file",
                data=file.model_dump_json(),
                timeout=self.timeout,
            ).json()
        )

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        return PromptConfig.model_validate(
            requests.post(
                f"{self._url}/gen/prompt",
                data=config.model_dump_json(),
                timeout=self.timeout,
            ).json()
        )

    def get_gen_prompt(self) -> PromptConfig:
        return PromptConfig.model_validate(
            requests.get(f"{self._url}/gen/prompt", timeout=self.timeout).json()
        )

    def set_qa_prompt(self, config: PromptConfig) -> PromptConfig:
        return PromptConfig.model_validate(
            requests.post(
                f"{self._url}/qa/prompt",
                data=config.model_dump_json(),
                timeout=self.timeout,
            ).json()
        )

    def get_qa_prompt(self) -> PromptConfig:
        return PromptConfig.model_validate(
            requests.get(f"{self._url}/qa/prompt", timeout=self.timeout).json()
        )

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        return GenResponse.model_validate(
            requests.post(
                f"{self._url}/gen/generate",
                json=parameters,
                timeout=self.timeout,
            ).json()
        )
