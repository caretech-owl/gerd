"""Implements the QAService class.

The question and answer service is used to query a language model with questions related
to a specific context. The context is usually a set of documents that are loaded into
a vector store.
"""

import json
import logging
import re
from os import unlink
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import gerd.loader as gerd_loader
from gerd.models.qa import QAConfig
from gerd.rag import FAISS, Rag, create_faiss, load_faiss
from gerd.transport import (
    DocumentSource,
    FileTypes,
    PromptConfig,
    QAAnalyzeAnswer,
    QAAnswer,
    QAFileUpload,
    QAModesEnum,
    QAQuestion,
)

if TYPE_CHECKING:
    from langchain.docstore.document import Document
    from langchain.document_loaders.base import BaseLoader


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class QAService:
    """The question and answer service class."""

    def __init__(self, config: QAConfig) -> None:
        """The service is initialized with a configuration.

        Depending on the configuration, the service will create a new in-memory
        vector store or load an existing one from a file.

        Parameters:
            config: The configuration for the QA service
        """
        self.config = config
        self._llm = gerd_loader.load_model_from_config(config.model)
        self._vectorstore: Optional[FAISS] = None
        self._database: Optional[Rag] = None
        if (
            config.embedding.db_path
            and Path(config.embedding.db_path, "index.faiss").exists()
        ):
            _LOGGER.info(
                "Load existing vector store from '%s'.", config.embedding.db_path
            )
            self._vectorstore = load_faiss(
                Path(config.embedding.db_path, "index.faiss"),
                config.embedding.model.name,
                config.device,
            )

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        """Queries the vector store with a question.

        The number of sources that are returned is defined by the max_sources parameter
        of the service's configuration.

        Parameters:
            question: The question to query the vector store with.

        Returns:
            A list of document sources
        """
        if not self._vectorstore:
            return []
        return [
            DocumentSource(
                query=question.question,
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
        """Converts a question to an embedding.

        The embedding to be used is defined by the vector store or more specifically
        by the configured parameters passed to initialize the vector store.

        Parameters:
            question: The question to convert to an embedding.

        Returns:
            The embedding of the question
        """
        if self._vectorstore is None or self._vectorstore.embeddings is None:
            return []
        return self._vectorstore.embeddings.embed_documents([question.question])[0]

    def query(self, question: QAQuestion) -> QAAnswer:
        """Pass a question to the language model.

        The language model will generate an answer based on the question and
        the context derived from the vector store.

        Parameters:
            question: The question to be answered

        Returns:
            The answer from the language model
        """
        if not self._database:
            if not self._vectorstore:
                return QAAnswer(error_msg="No database available!", status=404)
            self._database = Rag(
                self._llm,
                self.config.model,
                self.config.model.prompt_config,
                self._vectorstore,
                self.config.features.return_source,
            )
        return self._database.query(question)

    def analyze_query(self) -> QAAnalyzeAnswer:
        """Read a set of data from a set of documents.

        Loads the data via single prompt.

        *Data*
            - patient_name
            - patient_date_of_birth
            - attending_doctors
            - recording_date
            - release_date

        Returns:
            The answer from the language model
        """
        if not self._vectorstore:
            msg = "No vector store initialized! Upload documents first."
            _LOGGER.error(msg)
            return QAAnalyzeAnswer(status=404, error_msg=msg)

        config = self.config.features.analyze

        # questions to search model and vectorstore
        questions_model_dict: Dict[str, str] = {
            "Wie heißt der Patient?": "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa E501
            "Wann hat der Patient Geburstag?": "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa E501
            "Wie heißt der Arzt?": "Mit freundlichen kollegialen Grüßen, Prof, Dr",
            "Wann wurde der Patient bei uns aufgenommen?": (
                "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren"
            ),
            "Wann wurde der Patient bei uns entlassen?": (
                "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren"
            ),
        }

        # map model questions to jsonfields
        fields: Dict[str, str] = {
            "Wie heißt der Patient?": "patient_name",
            "Wann hat der Patient Geburstag?": "patient_date_of_birth",
            "Wie heißt der Arzt?": "attending_doctors",
            "Wann wurde der Patient bei uns aufgenommen?": "recording_date",
            "Wann wurde der Patient bei uns entlassen?": "release_date",
        }

        questions_dict: Dict[str, List[DocumentSource]] = {}
        parameters: Dict[str, str] = {}
        question_counter: int = 0

        qa_analyze_prompt = config.model.prompt_config
        # check if prompt contains needed fields
        for i in range(0, len(questions_model_dict)):
            if (
                "context" + str(i) not in qa_analyze_prompt.parameters
                or "question" + str(i) not in qa_analyze_prompt.parameters
            ):
                msg = (
                    "Prompt does not include '{context"
                    + str(i)
                    + "}' or '{question"
                    + str(i)
                    + "}' variable."
                )
                _LOGGER.error(msg)
                return QAAnalyzeAnswer(status=404, error_msg=msg)

        # load context from vectorstore for each question
        for question_m, question_v in questions_model_dict.items():
            questions_dict[question_m] = list(
                self.db_query(QAQuestion(question=question_v, max_sources=3))
            )

            parameters["context" + str(question_counter)] = " ".join(
                doc.content for doc in questions_dict[question_m]
            )
            parameters["question" + str(question_counter)] = question_m
            parameters["field" + str(question_counter)] = fields[question_m]

            question_counter = question_counter + 1

        formatted_prompt = qa_analyze_prompt.text.format(**parameters)

        # query the model
        response = self._llm.generate(formatted_prompt)

        if response is not None:
            response = self._clean_response(response)

        _LOGGER.info("\n===== Modelresult ====\n\n%s\n\n====================", response)

        # convert json to QAAnalyzerAnswerclass
        answer = self._format_response_analyze(response)

        # if enabled, pass source data to answer
        if self.config.features.return_source:
            answer.response = response
            answer.prompt = formatted_prompt
            for question in questions_dict:
                answer.sources = answer.sources + questions_dict[question]
            _LOGGER.info(
                "\n===== Sources ====\n\n%s\n\n====================", answer.sources
            )

        _LOGGER.warning("\n==== Answer ====\n\n%s\n===============", answer)
        return answer

    def analyze_mult_prompts_query(self) -> QAAnalyzeAnswer:
        """Reads a set of data from doc.

        Loads the data via multiple prompts by asking for each data field
        separately.

        *Data*
            - patient_name
            - patient_date_of_birth
            - attending_doctors
            - recording_date
            - release_date

        Returns:
            The answer from the language model
        """
        if not self._vectorstore:
            msg = "No vector store initialized! Upload documents first."
            _LOGGER.error(msg)
            return QAAnalyzeAnswer(status=404, error_msg=msg)

        config = self.config.features.analyze_mult_prompts

        # check if prompt contains needed fields
        qa_analyze_mult_prompts = config.model.prompt_config
        if (
            "context" not in qa_analyze_mult_prompts.parameters
            or "question" not in qa_analyze_mult_prompts.parameters
        ):
            msg = "Prompt does not include '{context}' or '{question}' variable."
            _LOGGER.error(msg)
            return QAAnalyzeAnswer(status=404, error_msg=msg)

        # questions to search model and vectorstore
        questions_model_dict: Dict[str, str] = {
            "Wie heißt der Patient?": "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa: E501
            "Wann hat der Patient Geburstag?": "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren oder  Patient, * 0.00.0000,",  # noqa: E501
            "Wie heißt der Arzt?": "Mit freundlichen kollegialen Grüßen, Prof, Dr",
            "Wann wurde der Patient bei uns aufgenommen?": (
                "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren"
            ),
            "Wann wurde der Patient bei uns entlassen?": (
                "wir berichten über unseren Patient oder Btr. oder Patient, wh, geboren"
            ),
        }
        fields: Dict[str, str] = {
            "Wie heißt der Patient?": "patient_name",
            "Wann hat der Patient Geburstag?": "patient_date_of_birth",
            "Wie heißt der Arzt?": "attending_doctors",
            "Wann wurde der Patient bei uns aufgenommen?": "recording_date",
            "Wann wurde der Patient bei uns entlassen?": "release_date",
        }
        questions_dict: Dict[str, List[DocumentSource]] = {}
        answer_dict: Dict[str, Any] = {}
        responses: str = ""

        # load context from vectorstore for each question
        for question_m, question_v in questions_model_dict.items():
            questions_dict, formatted_prompt = self._create_analyze_mult_prompt(
                question_m, question_v, qa_analyze_mult_prompts.text
            )

            # query the model for each question
            response = self._llm.generate(formatted_prompt)

            # format the response
            if response is not None:
                response = self._clean_response(response)

                # if enabled, collect response
                if self.config.features.return_source:
                    responses = responses + "; " + question_m + ": " + response
            # format the response
            answer_dict[fields[question_m]] = self._format_response_analyze_mult_prompt(
                response, fields[question_m]
            )

            _LOGGER.info(
                "\n===== Modelresult ====\n\n%s\n\n====================", response
            )

        answer = QAAnalyzeAnswer(**answer_dict)

        # if enabled, pass source data to answer
        if self.config.features.return_source:
            answer.response = responses
            for question in questions_dict:
                answer.sources = answer.sources + questions_dict[question]

        _LOGGER.warning("\n==== Answer ====\n\n%s\n===============", answer)
        return answer

    def set_prompt_config(self, config: PromptConfig, qa_mode: QAModesEnum) -> QAAnswer:
        """Sets the prompt config for the given mode.

        Parameters:
            config: The prompt config to set
            qa_mode: The mode to set the prompt config for

        Returns:
            an answer object with status 200 if successful
        """
        answer = QAAnswer()
        if qa_mode == QAModesEnum.SEARCH:
            self.config.model.prompt_config = config
            if "context" not in config.parameters:
                answer.error_msg = (
                    "Prompt does not include '{context}' variable. "
                    "No context will be added. "
                )
            if "question" not in config.parameters:
                answer.error_msg += (
                    "Prompt does not include '{question}' variable. "
                    "Questions will not be passed to the model."
                )
        elif qa_mode == QAModesEnum.ANALYZE:
            self.config.features.analyze.model.prompt_config = config
        elif qa_mode == QAModesEnum.ANALYZE_MULT_PROMPTS:
            self.config.features.analyze_mult_prompts.model.prompt_config = config
        return answer

    def get_prompt_config(self, qa_mode: QAModesEnum) -> PromptConfig:
        """Returns the prompt config for the given mode.

        Parameters:
            qa_mode: The mode to get the prompt config for

        Returns:
            The prompt config for the given mode
        """
        if qa_mode == QAModesEnum.SEARCH:
            return self.config.model.prompt_config
        elif qa_mode == QAModesEnum.ANALYZE:
            return self.config.features.analyze.model.prompt_config
        elif qa_mode == QAModesEnum.ANALYZE_MULT_PROMPTS:
            return self.config.features.analyze_mult_prompts.model.prompt_config
        return PromptConfig()

    def remove_file(self, file_name: str) -> QAAnswer:
        """Removes a document from the vectorstore.

        Parameters:
            file_name: The name of the file to remove

        Returns:
            an answer object with status 200 if successful
        """
        if not self._vectorstore:
            return QAAnswer(error_msg="No vector store initialized!", status=404)
        self._vectorstore.delete(
            [
                id
                for id in self._vectorstore.index_to_docstore_id.values()
                if id.startswith(file_name)
            ]
        )
        return QAAnswer(status=200)

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        """Add a document to the vectorstore.

        Parameters:
            file: The file to add to the vectorstore

        Returns:
            an answer object with status 200 if successful
        """
        documents: Optional[List[Document]] = None
        file_path = Path(file.name)
        try:
            with NamedTemporaryFile(
                dir=".", suffix=file_path.suffix, delete=False
            ) as f:
                f.write(file.data)
                f.flush()
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
            chunk_size=self.config.embedding.chunk_size,
            chunk_overlap=self.config.embedding.chunk_overlap,
        )
        texts = text_splitter.split_documents(documents)
        for i, text in enumerate(texts):
            text.id = f"{file_path.name}_{i}"
        if self._vectorstore is None:
            _LOGGER.info("Create new vector store from document.")
            self._vectorstore = create_faiss(
                texts, self.config.embedding.model.name, self.config.device
            )
        else:
            _LOGGER.info("Adding document to existing vector store.")
            tmp = create_faiss(
                texts, self.config.embedding.model.name, self.config.device
            )
            self._vectorstore.merge_from(tmp)
        if self.config.embedding.db_path:
            self._vectorstore.save_local(self.config.embedding.db_path)
        return QAAnswer()

    def _create_analyze_mult_prompt(
        self, model_q: str, vector_q: str, prompt: str
    ) -> Tuple[Dict[str, List[DocumentSource]], str]:
        if not self._vectorstore:
            msg = "No vector store initialized! Upload documents first."
            _LOGGER.error(msg)
            empty_dict: Dict[str, List[DocumentSource]] = {}
            return (empty_dict, "")

        parameters: Dict[str, str] = {}
        questions_dict: Dict[str, List[DocumentSource]] = {}

        questions_dict[model_q] = self.db_query(
            QAQuestion(question=vector_q, max_sources=3)
        )

        parameters["context"] = " ".join(doc.content for doc in questions_dict[model_q])
        parameters["question"] = model_q

        return (questions_dict, prompt.format(**parameters))

    def _format_response_query(self, response: str) -> str:
        response = response.replace('"""', '"')
        response = re.sub(r'""\n*(?=(\,|\\n}|}))', '" ', response)
        response = re.sub(r'\:\s*""(?=.)', ': "', response)  # noqa W605
        response = re.sub(r'\{\s*""(?=.)', "{ ", response)  # noqa W605

        split = response.split("{")
        if len(split) > 1:
            response = split[1]

        split = response.split("}")
        if len(split) > 1:
            response = split[0]

        if "{" not in response:
            response = "{" + response
        if "}" not in response:
            response = response + "}"

        if "[" in response or "]" in response:
            response = response.replace("[", "").replace("]", "")
        return response

    def _format_response_analyze_mult_prompt(
        self, response: str, field: str
    ) -> str | List[str]:
        try:
            # load in json structur
            answer = json.loads(response)["answer"]

            # format the answer
            if field == "attending_doctors":
                return self._format_attending_doctors(answer)
            elif answer is not None:
                return str(answer)
            else:
                return ""
        except BaseException:
            if field == "attending_doctors":
                empty_list: List[str] = []
                return empty_list
            else:
                return ""

    def _format_response_analyze(self, response: str) -> QAAnalyzeAnswer:
        try:
            # load in json structure
            answer_dict = json.loads(response)

            # format the attending_doctors field
            if answer_dict is not None:
                answer_dict["attending_doctors"] = self._format_attending_doctors(
                    answer_dict["attending_doctors"]
                )

                return QAAnalyzeAnswer(**answer_dict)
            else:
                return QAAnalyzeAnswer()
        except BaseException as err:
            _LOGGER.error(err)
            return QAAnalyzeAnswer()

    def _clean_response(self, response: str) -> str:
        response = response.replace('"""', '"').replace("\t", "").replace("\n", "")
        return response

    def _format_attending_doctors(self, attending_doctors: str) -> List[str] | str:
        if (
            attending_doctors is not None
            and "[" not in attending_doctors
            and isinstance(attending_doctors, str)
        ):
            return [attending_doctors]
        elif attending_doctors is not None and attending_doctors != "":
            return attending_doctors
        else:
            return []
