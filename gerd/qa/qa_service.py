import logging
from os import unlink
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import gerd.backends.loader as gerd_loader
from gerd.backends.rag import Rag, VectorStore, create_faiss, load_faiss
from gerd.models.qa import QAConfig
from gerd.transport import (
    DocumentSource,
    FileTypes,
    PromptConfig,
    QAAnswer,
    QAFileUpload,
    QAQuestion,
)

if TYPE_CHECKING:
    from langchain.docstore.document import Document
    from langchain.document_loaders.base import BaseLoader


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class QAService:
    def __init__(self, config: QAConfig) -> None:
        self._config = config
        self._llm = gerd_loader.load_model_from_config(config.model)
        self._vectorstore: Optional[VectorStore] = None
        self._database: Optional[Rag] = None
        if (
            config.embedding.db_path
            and Path(config.embedding.db_path, "index.faiss").exists()
        ):
            _LOGGER.info(
                "Load existing vector store from '%s'.", config.embedding.db_path
            )
            self._vectorstore = load_faiss(
                Path(config.embedding.db_path, "index.faiss")
            )

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        if not self._vectorstore:
            return []
        return [
            DocumentSource(
                content=doc.page_content,
                name=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 1),
            )
            for doc in self._vectorstore.search(
                question.question,
                search_type=question.search_strategy,
                k=question.max_sources,
            )
        ]

    def db_embedding(self, question: QAQuestion) -> List[float]:
        if not self._vectorstore:
            return []
        return self._vectorstore.embeddings.embed_documents([question.question])[0]

    def query(self, question: QAQuestion) -> QAAnswer:
        if not self._database:
            if not self._vectorstore:
                return QAAnswer(error_msg="No database available!", status=404)
            self._database = Rag(
                self._llm,
                self._config.model,
                self._config.model.prompt,
                self._vectorstore,
                self._config.features.return_source,
            )
        return self._database.query(question)

    def set_prompt(self, config: PromptConfig) -> PromptConfig:
        self._config.model.prompt = config
        self._database = Rag(
            self._llm,
            self._config.model,
            self._config.model.prompt,
            self._vectorstore,
            self._config.features.return_source,
        )
        return self._config.model.prompt

    def get_prompt(self) -> PromptConfig:
        return self._config.model.prompt

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        documents: Optional[List[Document]] = None
        file_path = Path(file.name)
        try:
            f = NamedTemporaryFile(dir=".", suffix=file_path.suffix, delete=False)
            f.write(file.data)
            f.flush()
            f.close()
            file_type = FileTypes(file_path.suffix[1:])
            loader: BaseLoader
            if file_type == FileTypes.TEXT:
                loader = TextLoader(f.name)
            elif file_type == FileTypes.PDF:
                loader = PyPDFLoader(f.name)
            documents = loader.load()
            # source must be overriden to not leak upload information
            # about the temp file which are rather useless anyway
            for doc in documents:
                doc.metadata["source"] = file_path.name
        except BaseException as err:
            _LOGGER.error(err)
        finally:
            if f:
                unlink(f.name)
        if not documents:
            _LOGGER.warning("No document was loaded!")
            return QAAnswer(error_msg="No document was loaded!", status=500)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.embedding.chunk_size,
            chunk_overlap=self._config.embedding.chunk_overlap,
        )
        texts = text_splitter.split_documents(documents)
        if self._vectorstore is None:
            _LOGGER.info("Create new vector store from document.")
            self._vectorstore = create_faiss(
                texts, self._config.embedding.model.name, self._config.device
            )
        else:
            _LOGGER.info("Adding document to existing vector store.")
            tmp = create_faiss(
                texts, self._config.embedding.model.name, self._config.device
            )
            self._vectorstore.merge_from(tmp)
        if self._config.embedding.db_path:
            self._vectorstore.save_local(self._config.embedding.db_path)
        return QAAnswer()
