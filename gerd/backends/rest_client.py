"""REST client for the GERD server."""

import logging
from typing import Dict, List

import requests
from pydantic import TypeAdapter
from typing_extensions import override

from gerd.config import CONFIG
from gerd.transport import QAQuestion

from ..transport import (
    DocumentSource,
    GenResponse,
    PromptConfig,
    QAAnalyzeAnswer,
    QAAnswer,
    QAFileUpload,
    QAModesEnum,
    QAPromptConfig,
    QAQuestion,
    Transport,
)

_LOGGER = logging.getLogger(__name__)


class RestClient(Transport):
    """REST client for the GERD server."""

    def __init__(self) -> None:
        """The client initalizes the server URL.

        It is retrieved from the global [CONFIG][gerd.config.CONFIG].
        Other (timeout) settings are also set here but not configurable as of now.
        """
        super().__init__()
        self._url = f"http://{CONFIG.server.host}:{CONFIG.server.port}{CONFIG.server.api_prefix}"
        self.timeout = 10
        self.longtimeout = 10000

    @override
    def qa_query(self, question: QAQuestion) -> QAAnswer:
        return QAAnswer.model_validate(
            requests.post(
                f"{self._url}/qa/query",
                data=question.model_dump_json().encode("utf-8"),
                timeout=self.longtimeout,
            ).json()
        )

    @override
    def analyze_query(self) -> QAAnalyzeAnswer:
        return QAAnalyzeAnswer.model_validate(
            requests.post(
                f"{self._url}/qa/query_analyze",
                timeout=self.longtimeout,
            ).json()
        )

    @override
    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        return QAAnalyzeAnswer.model_validate(
            requests.post(
                f"{self._url}/qa/query_analyze_mult_prompt",
                timeout=self.longtimeout,
            ).json()
        )

    @override
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

    @override
    def db_embedding(self, question: QAQuestion) -> List[float]:
        request = question.model_dump_json()
        _LOGGER.debug("db_embedding - request: %s", request)
        response = requests.post(
            f"{self._url}/qa/db_embedding",
            data=question.model_dump_json().encode("utf-8"),
            timeout=self.timeout,
        )
        _LOGGER.debug("db_embedding - response: %s", response.json())
        return TypeAdapter(List[float]).validate_python(response.json())

    @override
    def add_file(self, file: QAFileUpload) -> QAAnswer:
        t = file.model_dump_json().encode("utf-8")
        return QAAnswer.model_validate(
            requests.post(
                f"{self._url}/qa/file",
                data=file.model_dump_json().encode("utf-8"),
                timeout=self.timeout,
            ).json()
        )

    @override
    def remove_file(self, file_name: str) -> QAAnswer:
        return QAAnswer.model_validate(
            requests.delete(
                f"{self._url}/qa/file",
                data=file_name.encode("utf-8"),
                timeout=self.timeout,
            ).json()
        )

    @override
    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        return PromptConfig.model_validate(
            requests.post(
                f"{self._url}/gen/prompt",
                data=config.model_dump_json(),
                timeout=self.timeout,
            ).json()
        )

    @override
    def get_gen_prompt(self) -> PromptConfig:
        return PromptConfig.model_validate(
            requests.get(f"{self._url}/gen/prompt", timeout=self.timeout).json()
        )

    @override
    def set_qa_prompt(self, config: PromptConfig, qa_mode: QAModesEnum) -> QAAnswer:
        return QAAnswer.model_validate(
            requests.post(
                f"{self._url}/qa/prompt",
                data=QAPromptConfig(config=config, mode=qa_mode).model_dump_json(),
                timeout=self.timeout,
            ).json()
        )

    @override
    def get_qa_prompt(self, qa_mode: QAModesEnum) -> PromptConfig:
        return PromptConfig.model_validate(
            requests.get(
                f"{self._url}/qa/prompt",
                timeout=self.timeout,
                params={"qa_mode": qa_mode.value},
            ).json()
        )

    @override
    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        return GenResponse.model_validate(
            requests.post(
                f"{self._url}/gen/generate",
                json=parameters,
                timeout=self.timeout,
            ).json()
        )
