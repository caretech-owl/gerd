"""Retrieval-Augmented Generation (RAG) backend.

This module provides the RAG backend for the GERD system which is currently
based on FAISS.
"""

import logging
from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from gerd.loader import LLM
from gerd.models.model import ModelConfig, PromptConfig
from gerd.transport import DocumentSource, QAAnswer, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def create_faiss(documents: list[Document], model_name: str, device: str) -> FAISS:
    """Create a new FAISS store from a list of documents.

    Parameters:
        documents: The list of documents to index
        model_name: The name of the Hugging Face model to for the embeddings
        device: The device to use for the model

    Returns:
        The newly created FAISS store
    """
    return FAISS.from_documents(
        documents,
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        ),
    )


def load_faiss(dp_path: Path, model_name: str, device: str) -> FAISS:
    """Load a FAISS store from a disk path.

    Parameters:
        dp_path: The path to the disk path
        model_name: The name of the Hugging Face model to for the embeddings
        device: The device to use for the model

    Returns:
        The loaded FAISS store
    """
    return FAISS.load_local(
        dp_path.as_posix(),
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        ),
    )


class Rag:
    """The RAG backend for GERD."""

    def __init__(
        self,
        model: LLM,
        model_config: ModelConfig,
        prompt: PromptConfig,
        store: FAISS,
        return_source: bool,
    ) -> None:
        """The RAG backend will check for a context parameter in the prompt.

        If the context parameter is not included, a warning will be logged.
        Without the context parameter, no context will be added to the query.

        Parameters:
            model: The LLM model to use
            model_config: The model configuration
            prompt: The prompt configuration
            store: The FAISS store to use
            return_source: Whether to return the source documents
        """
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
        """Query the RAG backend with a question.

        Parameters:
            question: The question to ask

        Returns:
            The answer to the question including the sources
        """
        docs = self.store.search(
            question.question,
            search_type=question.search_strategy,
            k=question.max_sources,
        )
        context = "\n".join(doc.page_content for doc in docs)
        resolved = self.prompt.text.format(context=context, question=question.question)
        _, response = self.model.create_chat_completion(
            [{"role": "user", "content": resolved}]
        )
        answer = QAAnswer(response=response)
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
