import logging
from typing import Dict, List

import uvicorn
from fastapi import APIRouter, FastAPI

from team_red.backends.bridge import Bridge
from team_red.config import CONFIG

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


class RestServer(Transport):
    def __init__(self) -> None:
        super().__init__()

        prefix = CONFIG.server.api_prefix
        self._bridge = Bridge()
        self.router = APIRouter()
        self.router.add_api_route(f"{prefix}/qa/query", self.qa_query, methods=["POST"])
        self.router.add_api_route(
            f"{prefix}/qa/db_query", self.db_query, methods=["POST"]
        )
        self.router.add_api_route(f"{prefix}/qa/file", self.add_file, methods=["POST"])
        self.router.add_api_route(
            f"{prefix}/gen/prompt", self.set_gen_prompt, methods=["POST"]
        )
        self.router.add_api_route(
            f"{prefix}/gen/prompt", self.get_gen_prompt, methods=["GET"]
        )
        self.router.add_api_route(
            f"{prefix}/qa/prompt", self.set_qa_prompt, methods=["POST"]
        )
        self.router.add_api_route(
            f"{prefix}/qa/prompt", self.get_qa_prompt, methods=["GET"]
        )
        self.router.add_api_route(
            f"{prefix}/gen/generate", self.generate, methods=["POST"]
        )

    def qa_query(self, question: QAQuestion) -> QAAnswer:
        return self._bridge.qa_query(question)

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        _LOGGER.debug("dq_query - request: %s", question)
        response = self._bridge.db_query(question)
        _LOGGER.debug("dq_query - response: %s", response)
        return response

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        return self._bridge.add_file(file)

    def set_gen_prompt(self, config: PromptConfig) -> PromptConfig:
        return self._bridge.set_gen_prompt(config)

    def get_gen_prompt(self) -> PromptConfig:
        return self._bridge.get_gen_prompt()

    def set_qa_prompt(self, config: PromptConfig) -> PromptConfig:
        return self._bridge.set_qa_prompt(config)

    def get_qa_prompt(self) -> PromptConfig:
        return self._bridge.get_qa_prompt()

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        return self._bridge.generate(parameters)


app = FastAPI()
rest_server = RestServer()
app.include_router(rest_server.router)


if __name__ == "__main__":
    uvicorn.run(
        "team_red.backends.rest_server:app",
        host=CONFIG.server.host,
        port=CONFIG.server.port,
        reload=True,
    )
