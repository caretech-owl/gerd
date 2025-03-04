"""REST server as a GERD backend."""

import logging
from typing import Dict, List

import uvicorn
from fastapi import APIRouter, FastAPI
from typing_extensions import override

from gerd.backends.bridge import Bridge
from gerd.config import CONFIG
from gerd.transport import (
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


class RestServer(Transport):
    """REST server as a GERD backend."""

    def __init__(self) -> None:
        """The REST server initializes a private bridge and an API router.

        The API router is used to define the endpoints for the REST server.
        """
        super().__init__()

        prefix = CONFIG.server.api_prefix
        self._bridge = Bridge()
        self.router = APIRouter()
        self.router.add_api_route(f"{prefix}/qa/query", self.qa_query, methods=["POST"])
        self.router.add_api_route(
            f"{prefix}/qa/query_analyze", self.analyze_query, methods=["POST"]
        )
        self.router.add_api_route(
            f"{prefix}/qa/query_analyze_mult_prompt",
            self.analyze_mult_prompts_query,
            methods=["POST"],
        )
        self.router.add_api_route(
            f"{prefix}/qa/db_query", self.db_query, methods=["POST"]
        )
        self.router.add_api_route(
            f"{prefix}/qa/db_embedding", self.db_embedding, methods=["POST"]
        )
        self.router.add_api_route(f"{prefix}/qa/file", self.add_file, methods=["POST"])
        self.router.add_api_route(
            f"{prefix}/qa/file", self.remove_file, methods=["DELETE"]
        )
        self.router.add_api_route(
            f"{prefix}/gen/prompt", self.set_gen_prompt, methods=["POST"]
        )
        self.router.add_api_route(
            f"{prefix}/gen/prompt", self.get_gen_prompt, methods=["GET"]
        )
        self.router.add_api_route(
            f"{prefix}/qa/prompt", self.set_qa_prompt_rest, methods=["POST"]
        )
        self.router.add_api_route(
            f"{prefix}/qa/prompt", self.get_qa_prompt_rest, methods=["GET"]
        )
        self.router.add_api_route(
            f"{prefix}/gen/generate", self.generate, methods=["POST"]
        )

    @override
    def qa_query(self, question: QAQuestion) -> QAAnswer:
        return self._bridge.qa_query(question)

    @override
    def analyze_query(self) -> QAAnalyzeAnswer:
        return self._bridge.qa.analyze_query()

    @override
    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        return self._bridge.qa.analyze_mult_prompts_query()

    @override
    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        # _LOGGER.debug("dq_query - request: %s", question)
        response = self._bridge.db_query(question)
        # _LOGGER.debug("dq_query - response: %s", response)
        return response

    @override
    def db_embedding(self, question: QAQuestion) -> List[float]:
        return self._bridge.db_embedding(question)

    @override
    def add_file(self, file: QAFileUpload) -> QAAnswer:
        return self._bridge.add_file(file)

    @override
    def remove_file(self, file_name: str) -> QAAnswer:
        return self._bridge.remove_file(file_name)

    @override
    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        return self._bridge.set_gen_prompt(config)

    @override
    def get_gen_prompt(self) -> PromptConfig:
        return self._bridge.get_gen_prompt()

    def set_qa_prompt_rest(self, config: QAPromptConfig) -> QAAnswer:
        """Set the QA prompt configuration.

        The call is forwarded to the bridge.
        Parameters:
            config: The QA prompt configuration
        Returns:
            The QA prompt configuration; Should be the same as the input in most cases
        """
        return self._bridge.set_qa_prompt(config.config, config.mode)

    def get_qa_prompt_rest(self, qa_mode: int) -> PromptConfig:
        """Get the QA prompt configuration.

        The call is forwarded to the bridge.
        Parameters:
            qa_mode: The QA mode
        Returns:
            The QA prompt configuration
        """
        return self._bridge.get_qa_prompt(QAModesEnum(qa_mode))

    @override
    def set_qa_prompt(self, config: PromptConfig, qa_mode: QAModesEnum) -> QAAnswer:
        return self._bridge.set_qa_prompt(config, qa_mode)

    @override
    def get_qa_prompt(self, qa_mode: QAModesEnum) -> PromptConfig:
        return self._bridge.get_qa_prompt(qa_mode)

    @override
    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        return self._bridge.generate(parameters)


app = FastAPI()
rest_server = RestServer()
app.include_router(rest_server.router)


if __name__ == "__main__":
    uvicorn.run(
        "gerd.backends.rest_server:app",
        host=CONFIG.server.host,
        port=CONFIG.server.port,
        reload=True,
    )
