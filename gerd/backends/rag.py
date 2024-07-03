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

class VectorEmbeddings(Protocol):
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        pass


class VectorStore(Protocol):
    def merge_from(self, vector_store: "VectorStore") -> None:
        pass

    def save_local(self, path: str) -> None:
        pass

    def add_texts(
        self,
        texts: Iterable[str],
    ) -> list[str]:
        pass

    def search(self, query: str, search_type: str, k: int) -> list[Document]:
        pass

    embeddings: VectorEmbeddings


def create_faiss(
    documents: list[Document], model_name: str, device: str
) -> VectorStore:
    return FAISS.from_documents(
        documents,
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        ),
    )


def load_faiss(dp_path: Path, model_name: str, device: str) -> VectorStore:
    return FAISS.load_local(
        dp_path,
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
        store: VectorStore | None,
        return_source: bool,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.prompt = prompt
        self.store = store
        self.return_source = return_source

        if "context" not in prompt.parameters:
            _LOGGER.warning(
                "Prompt does not include '{context}' variable."
                "It will be appened to the prompt."
            )
            prompt.text += "\n\n{context}"

    def query(self, question: QAQuestion) -> QAAnswer:
        docs = self.store.search(
            question.question,
            search_type=question.search_strategy,
            k=question.max_sources,
        )
        context = "\n".join(doc.page_content for doc in docs)
        resolved = self.prompt.text.format(context=context, question=question.question)
        response = self.model.generate(resolved)
        answer = QAAnswer(answer=response)
        if self.return_source:
            for doc in docs:
                answer.sources.append(
                    DocumentSource(
                        content=doc.page_content,
                        name=doc.metadata.get("source", "unknown"),
                        page=doc.metadata.get("page", 1),
                    )
                )
        return answer
