import json
import logging
from os import unlink
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Protocol, Tuple

from ctransformers import AutoModelForCausalLM
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from team_red.llm import build_llm
from team_red.models.model import ModelConfig
from team_red.models.qa import QAConfig
from team_red.transport import (
    DocumentSource,
    FileTypes,
    PromptConfig,
    QAAnalyzeAnswer,
    QAAnswer,
    QAFileUpload,
    QAModesEnum,
    QAQuestion,
)
from team_red.utils import build_retrieval_qa

if TYPE_CHECKING:
    from langchain.document_loaders.base import BaseLoader


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class VectorEmbeddings(Protocol):
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        pass


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

    def search(self, query: str, search_type: str, k: int) -> List[Document]:
        pass

    embeddings: VectorEmbeddings

class QAService:
    def __init__(self, config: QAConfig) -> None:
        self._config = config
        self._llm = build_llm(config.model)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._config.embedding.model.name,
            model_kwargs={"device": self._config.device},
        )
        self._vectorstore: Optional[VectorStore] = None
        self._fact_checker_db: Optional[BaseRetrievalQA] = None
        if (
            config.embedding.db_path
            and Path(config.embedding.db_path, "index.faiss").exists()
        ):
            _LOGGER.info(
                "Load existing vector store from '%s'.", config.embedding.db_path
            )
            self._vectorstore = FAISS.load_local(
                config.embedding.db_path, self._embeddings
            )

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        if not self._vectorstore:
            return []
        return [
            DocumentSource(
                question=question.question,
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
        if not self._vectorstore:
            msg = "No vector store initialized! Upload documents first."
            _LOGGER.error(msg)
            return QAAnswer(status=404, error_msg=msg)

        self._model = self._init_model(self._config.model)

        parameters: Dict[str, str] = {}
        context_list: List[Document]
        qa_query_prompt = self._config.model.prompt

        context_list = list(self._vectorstore.search(
            question.question,
            search_type=question.search_strategy,
            k=question.max_sources
        ))

        parameters["context"] = " ".join(doc.page_content for doc in context_list)
        parameters["question"] = question.question

        if ("context" not in qa_query_prompt.parameters
            or "question"  not in qa_query_prompt.parameters):
                msg = "Prompt does not include '{context}' or '{question}' variable."
                _LOGGER.error(msg)
                return QAAnalyzeAnswer(status=404, error_msg=msg)

        formatted_prompt = qa_query_prompt.text.format(**parameters)

        response = self._query_model(self._config.model, formatted_prompt)

        _LOGGER.info(
            "\n===== Modelresult ====\n\n%s\n\n====================",
            response
        )

        if response is not None:
            response = response.replace('"""', '"')
            if ("["  in response or "]"  in response):
                response = response.replace('[', '').replace(']', '')

        try:
            answer_json = json.loads(response)["answer"]

            if answer_json is None:
                answer_json = ""
        except BaseException:
            answer_json = ""

        answer = QAAnswer(answer=answer_json)

        if self._config.features.return_source:
            answer.sources = self._collect_source_docs(question.question, context_list)
            answer.model_response = response

        if self._config.features.fact_checking.enabled is True:
            if not self._fact_checker_db:
                self._fact_checker_db = self._setup_dbqa_fact_checking(
                    self._config.features.fact_checking.model.prompt
                )
            response = self._fact_checker_db({"query": response["result"]})

            answer.sources.append(
                self._collect_source_docs(
                    response.get("source_documents", [])))

        _LOGGER.debug("\n==== Answer ====\n\n%s\n===============", answer)
        return answer

    def analyze_query(self) -> QAAnalyzeAnswer:
        if not self._vectorstore:
            msg = "No vector store initialized! Upload documents first."
            _LOGGER.error(msg)
            return QAAnalyzeAnswer(status=404, error_msg=msg)

        config = self._config.features.analyze
        self._model = self._init_model(config.model)

        questions_model_dict : Dict[str, str] = {
            "Wie heißt der Patient?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,", # noqa E501
            "Wann hat der Patient Geburstag?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,", # noqa E501
            "Wie heißt der Arzt?":
            "Mit freundlichen kollegialen Grüßen, Prof, Dr",
            "Wann wurde der Patient bei uns aufgenommen?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren",
            "Wann wurde der Patient bei uns entlassen?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren"
        }

        fields: Dict[str, str] = {
            "Wie heißt der Patient?": "patient_name",
            "Wann hat der Patient Geburstag?": "patient_date_of_birth",
            "Wie heißt der Arzt?": "attending_doctors",
            "Wann wurde der Patient bei uns aufgenommen?": "recording_date",
            "Wann wurde der Patient bei uns entlassen?": "release_date",
        }

        questions_dict : Dict[str, str] = {}
        parameters: Dict[str, str] = {}
        question_counter : int = 0

        qa_analyze_prompt = config.model.prompt
        for i in range(0, len(questions_model_dict)):
            if ("context" + str(i) not in qa_analyze_prompt.parameters
                or "question" + str(i)
                not in qa_analyze_prompt.parameters):
                msg = ("Prompt does not include '{context"
                + str(i) + "}' or '{question"
                + str(i) + "}' variable.")
                _LOGGER.error(msg)
                return QAAnalyzeAnswer(status=404, error_msg=msg)

        for question_m, question_v in questions_model_dict.items():
            questions_dict[question_m] = list(self._vectorstore.search(
                question_v,
                search_type="similarity",
                k=3
                ))

            parameters["context" + str(question_counter)] = " ".join(
                doc.page_content for doc in questions_dict[question_m]
            )
            parameters["question" + str(question_counter)] = question_m
            parameters["field" + str(question_counter)] = fields[question_m]

            question_counter = question_counter + 1

        formatted_prompt = qa_analyze_prompt.text.format(**parameters)

        response = self._query_model(config.model, formatted_prompt)

        if response is not None:
            response = self._clean_response(response)

        _LOGGER.info(
            "\n===== Modelresult ====\n\n%s\n\n====================",
            response
        )

        # convert json to QAAnalyzerAnswerclass
        answer = self._format_response_analyze(response)

        if self._config.features.return_source:
            answer.model_response = response
            answer.prompt = formatted_prompt
            for question in questions_dict:
                answer.sources = answer.sources + self._collect_source_docs(
                    question, questions_dict[question])
            _LOGGER.info(
           "\n===== Sources ====\n\n%s\n\n====================",
           answer.sources
        )

        _LOGGER.warning("\n==== Answer ====\n\n%s\n===============", answer)
        return answer

    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        if not self._vectorstore:
            msg = "No vector store initialized! Upload documents first."
            _LOGGER.error(msg)
            return QAAnalyzeAnswer(status=404, error_msg=msg)

        config = self._config.features.analyze_mult_prompts

        qa_analyze_mult_prompts = config.model.prompt
        if ("context" not in qa_analyze_mult_prompts.parameters
                or "question"  not in qa_analyze_mult_prompts.parameters):
                msg = "Prompt does not include '{context}' or '{question}' variable."
                _LOGGER.error(msg)
                return QAAnalyzeAnswer(status=404, error_msg=msg)

        self._model = self._init_model(config.model)

        questions_model_dict : Dict[str, str] = {
            "Wie heißt der Patient?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,", # noqa: E501
            "Wann hat der Patient Geburstag?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,", # noqa: E501
            "Wie heißt der Arzt?" :
            "Mit freundlichen kollegialen Grüßen, Prof, Dr",
            "Wann wurde der Patient bei uns aufgenommen?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren",
            "Wann wurde der Patient bei uns entlassen?" :
            "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren"}
        fields: Dict[str, str]= {
            "Wie heißt der Patient?" : "patient_name",
            "Wann hat der Patient Geburstag?" : "patient_date_of_birth",
            "Wie heißt der Arzt?" : "attending_doctors",
            "Wann wurde der Patient bei uns aufgenommen?" : "recording_date",
            "Wann wurde der Patient bei uns entlassen?" : "release_date"}
        questions_dict : Dict[str, str] = {}
        answer_dict: Dict[str, str] = {}
        responses: str = ""

        for question_m, question_v in questions_model_dict.items():
            questions_dict, formatted_prompt = self._create_analyze_mult_prompt(
                question_m, question_v, qa_analyze_mult_prompts.text)

            response = self._query_model(config.model, formatted_prompt)

            if response is not None:
                response = self._clean_response(response)

                if self._config.features.return_source:
                    responses = responses + "; " + question_m + ": " + response

            answer_dict[fields[question_m]] = self._format_response_analyze_mult_prompt(
                response, fields[question_m])

            _LOGGER.info(
                "\n===== Modelresult ====\n\n%s\n\n====================",
                response
            )

        answer = QAAnalyzeAnswer(**answer_dict)

        if self._config.features.return_source:
            answer.model_response = responses
            for question in questions_dict:
                answer.sources = answer.sources + self._collect_source_docs(
                    question, questions_dict[question]
                )

        _LOGGER.warning("\n==== Answer ====\n\n%s\n===============", answer)
        return answer

    def set_prompt(self, config: PromptConfig, qa_mode: QAModesEnum) -> PromptConfig:
        if qa_mode == QAModesEnum.SEARCH:
            self._config.model.prompt = config
            return self._config.model.prompt
        elif qa_mode == QAModesEnum.ANALYZE:
            self._config.features.analyze.model.prompt = config
            return self._config.features.analyze.model.prompt
        elif qa_mode == QAModesEnum.ANALYZE_MULT_PROMPTS:
            self._config.features.analyze_mult_prompts.model.prompt = config
            return self._config.features.analyze_mult_prompts.model.prompt
        return PromptConfig()

    def get_prompt(self, qa_mode: QAModesEnum) -> PromptConfig:
        if qa_mode == QAModesEnum.SEARCH:
            return self._config.model.prompt
        elif qa_mode == QAModesEnum.ANALYZE:
            return self._config.features.analyze.model.prompt
        elif qa_mode == QAModesEnum.ANALYZE_MULT_PROMPTS:
            return self._config.features.analyze_mult_prompts.model.prompt
        return PromptConfig()

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
            self._vectorstore = FAISS.from_documents(texts, self._embeddings)
        else:
            _LOGGER.info("Adding document to existing vector store.")
            tmp = FAISS.from_documents(texts, self._embeddings)
            self._vectorstore.merge_from(tmp)
        if self._config.embedding.db_path:
            self._vectorstore.save_local(self._config.embedding.db_path)
        return QAAnswer()

    def _init_model(self, model_config: ModelConfig) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
                model_path_or_repo_id = model_config.name,
                model_file = model_config.file,
                model_type = model_config.type,
                context_length = model_config.context_length
                )

    def _query_model(self, model_config: ModelConfig, prompt: str) -> str:
        if  not self._model:
            return ""

        return self._model(
                prompt,
                stop = model_config.stop,
                max_new_tokens = model_config.max_new_tokens,
                top_p = model_config.top_p,
                top_k = model_config.top_k,
                temperature = model_config.temperature,
                repetition_penalty = model_config.repetition_penalty,
            )

    def _create_analyze_mult_prompt(
            self, model_q: str, vector_q: str, prompt: str
            ) -> Tuple[Dict[str, str], str]:
        parameters: Dict[str, str] = {}
        questions_dict : Dict[str, str] = {}

        questions_dict[model_q] = list(self._vectorstore.search(
                vector_q,
                search_type="similarity",
                k=3
                ))

        parameters["context"] = " ".join(
            doc.page_content for doc in questions_dict[model_q]
        )
        parameters["question"] = model_q

        return [questions_dict, prompt.format(**parameters)]

    def _format_response_analyze_mult_prompt(
            self, response: str, field: str) -> str | List[str]:
        try:
            answer = json.loads(response)["answer"]

            if field == "attending_doctors":
                return self._format_attending_doctors(answer)
            elif answer is not None:
                return answer
            else:
                return ""
        except BaseException:
            if field == "attending_doctors":
                return []
            else:
                return ""

    def _format_response_analyze(self, response: str) -> QAAnalyzeAnswer:
        try:
            answer_dict = json.loads(response)

            if answer_dict is not None:
                answer_dict["attending_doctors"] = self._format_attending_doctors(
                    answer_dict["attending_doctors"])

                return QAAnalyzeAnswer(**answer_dict)
            else:
                return QAAnalyzeAnswer()
        except BaseException as err:
            _LOGGER.error(err)
            return QAAnalyzeAnswer()

    def _collect_source_docs(
            self, question_str: str, docs: List[Document]
            ) -> List[DocumentSource]:
        answer_sources : List[DocumentSource] = []

        answer_sources = [DocumentSource(
                question=question_str,
                content=doc.page_content,
                name=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 1),
                ) for doc in docs]

        return answer_sources

    def _clean_response(self, response : str) -> str:
        response = response.replace(
            '"""', '"').replace(
            "\t", "").replace(
            "\n", "")
        return response

    def _format_attending_doctors(self, attending_doctors: str) -> List[str]:
        if (attending_doctors is not None
                and "[" not in attending_doctors
                and isinstance(attending_doctors), str):
            return [attending_doctors]
        elif attending_doctors is not None and attending_doctors != "":
            return attending_doctors
        else:
            return []


    def _setup_dbqa_fact_checking(self, prompt: PromptConfig) -> BaseRetrievalQA:
        _LOGGER.info("Setup fact checking...")
        if "context" not in prompt.parameters:
            _LOGGER.warning(
                "Prompt does not include '{context}' variable."
                "It will be appened to the prompt."
            )
            prompt.text += "\n\n{context}"
        fact_checking_prompt = PromptTemplate(
            template=prompt.text,
            input_variables=prompt.parameters,
        )
        dbqa = build_retrieval_qa(
            self._llm,
            fact_checking_prompt,
            self._vectorstore,
            self._config.embedding.vector_count,
            self._config.features.return_source,
        )

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
