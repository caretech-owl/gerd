import logging
from pathlib import Path
from typing import Iterable, Protocol

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from gerd.backends.loader import LLM
from gerd.models.model import ModelConfig, PromptConfig
from gerd.transport import DocumentSource, QAAnswer, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def create_faiss(documents: list[Document], model_name: str, device: str) -> FAISS:
    return FAISS.from_documents(
        documents,
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        ),
    )


def load_faiss(dp_path: Path, model_name: str, device: str) -> FAISS:
    return FAISS.load_local(
        dp_path.as_posix(),
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        ),
    )


class Rag:
    def __init__(
        self,
        model: LLM,
        model_config: ModelConfig,
        prompt: PromptConfig,
        store: FAISS,
        return_source: bool,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.prompt = prompt
        self.store = store
        self.return_source = return_source

        if "context" not in prompt.parameters:
            _LOGGER.warning(
                "Prompt does not include '{context}' variable! "
                "No context will be added to the query."
            )

    def query(self, question: QAQuestion) -> QAAnswer:
        docs = self.store.search(
            question.question,
            search_type=question.search_strategy,
            k=question.max_sources,
        )
        context = "\n".join(doc.page_content for doc in docs)
        resolved = self.prompt.text.format(context=context, question=question.question)
        role, response = self.model.create_chat_completion(
            [{"role": "user", "content": resolved}]
        )
        answer = QAAnswer(answer=response)
        if self.return_source:
            for doc in docs:
                answer.sources.append(
                    DocumentSource(
                        query=question.question,
                        content=doc.page_content,
                        name=doc.metadata.get("source", "unknown"),
                        page=doc.metadata.get("page", 1),
                    )
                )
        return answer
