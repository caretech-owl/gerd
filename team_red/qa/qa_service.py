import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Iterable, List, Optional, Protocol

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from team_red.config import CONFIG
from team_red.llm import build_llm
from team_red.transport import (
    DocumentSource,
    FileTypes,
    QAAnswer,
    QAFileUpload,
    QAQuestion,
)
from team_red.utils import build_retrieval_qa

from .prompts import fact_checking_template, qa_template

if TYPE_CHECKING:
    from langchain.document_loaders.base import BaseLoader


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class VectorStore(Protocol):
    def merge_from(self, vector_store: "VectorStore") -> None:
        pass

    def save_local(self, path: str) -> None:
        pass

    def add_texts(
        self,
        texts: Iterable[str],
    ) -> List[str]:
        pass


class QAService:
    def __init__(self) -> None:
        self._llm = build_llm()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG.data.embedding.model,
            model_kwargs={"device": CONFIG.device},
        )
        self._vectorstore: Optional[VectorStore] = None
        self._database: Optional[BaseRetrievalQA] = None
        self._fact_checker_db: Optional[BaseRetrievalQA] = None
        if Path(CONFIG.data.embedding.db_path, "index.faiss").exists():
            self._vectorstore = FAISS.load_local(
                CONFIG.data.embedding.db_path, self._embeddings
            )

    def query(self, question: QAQuestion) -> QAAnswer:
        if not self._database:
            if not self._vectorstore:
                msg = "No vector store initialized! Upload documents first."
                _LOGGER.error(msg)
                return QAAnswer(status=404, error_msg=msg)
            self._database = self._setup_dbqa()

        response = self._database({"query": question.question})
        answer = QAAnswer(answer=response["result"])
        if CONFIG.features.return_source:
            for doc in response["source_documents"]:
                answer.sources.append(
                    DocumentSource(
                        content=doc.page_content,
                        name=doc.metadata.get("source", "unknown"),
                        page=doc.metadata.get("page", 1),
                    )
                )
        if CONFIG.features.fact_checking:
            if not self._fact_checker_db:
                self._fact_checker_db = self._setup_dbqa_fact_checking()
            response = self._fact_checker_db({"query": response["result"]})
            answer.answer = response["result"]
        return answer

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        with NamedTemporaryFile(dir=".", suffix=file.type.value) as f:
            f.write(file.data)
            f.flush()
            loader: BaseLoader
            if file.type == FileTypes.TEXT:
                loader = TextLoader(f.name)
            elif file.type == FileTypes.PDF:
                loader = PyPDFLoader(f.name)
            documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.data.chunk_size,
            chunk_overlap=CONFIG.data.chunk_overlap,
        )
        texts = text_splitter.split_documents(documents)
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(texts, self._embeddings)
        else:
            tmp = FAISS.from_documents(texts, self._embeddings)
            self._vectorstore.merge_from(tmp)
        self._vectorstore.save_local(CONFIG.data.embedding.db_path)
        return QAAnswer()

    def _setup_dbqa(self) -> BaseRetrievalQA:
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"],
        )
        dbqa = build_retrieval_qa(self._llm, qa_prompt, self._vectorstore)

        return dbqa

    def _setup_dbqa_fact_checking(self) -> BaseRetrievalQA:
        fact_checking_prompt = PromptTemplate(
            template=fact_checking_template,
            input_variables=["context", "question"],
        )
        dbqa = build_retrieval_qa(self._llm, fact_checking_prompt, self._vectorstore)

        return dbqa


# # Process source documents
# source_docs = response["source_documents"]
# for i, doc in enumerate(source_docs):
#     _LOGGER.debug(f"\nSource Document {i+1}\n")
#     _LOGGER.debug(f"Source Text: {doc.page_content}")
#     _LOGGER.debug(f'Document Name: {doc.metadata["source"]}')
#     _LOGGER.debug(f'Page Number: {doc.metadata.get("page", 1)}\n')
#     _LOGGER.debug("=" * 60)

# _LOGGER.debug(f"Time to retrieve response: {endQA - startQA}")

# if CONFIG.features.fact_checking:
#     startFactCheck = timeit.default_timer()
#     dbqafact = setup_dbqa_fact_checking()
#     response_fact = dbqafact({"query": response["result"]})
#     endFactCheck = timeit.default_timer()
#     _LOGGER.debug("Factcheck:")
#     _LOGGER.debug(f'\nAnswer: {response_fact["result"]}')
#     _LOGGER.debug("=" * 50)

#     # Process source documents
#     source_docs = response_fact["source_documents"]
#     for i, doc in enumerate(source_docs):
#         _LOGGER.debug(f"\nSource Document {i+1}\n")
#         _LOGGER.debug(f"Source Text: {doc.page_content}")
#         _LOGGER.debug(f'Document Name: {doc.metadata["source"]}')
#         _LOGGER.debug(f'Page Number: {doc.metadata.get("page", 1)}\n')
#         _LOGGER.debug("=" * 60)

#     _LOGGER.debug(f"Time to retrieve fact check: {endFactCheck - startFactCheck}")
