import logging
from string import Formatter
from typing import Optional

from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

from team_red.config import CONFIG
from team_red.utils import setup_dbqa, setup_dbqa_fact_checking

from .interface import (
    GenResponse,
    Interface,
    PromptConfig,
    PromptParameters,
    QAAnswer,
    QAQuestion,
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Direct(Interface):
    def __init__(self) -> None:
        super().__init__()
        self._database = setup_dbqa()
        self._prompt_config: Optional[PromptConfig] = None
        if CONFIG.features.fact_checking:
            self._fact_checker_db = setup_dbqa_fact_checking()

    def qa_query(self, question: QAQuestion) -> QAAnswer:
        response = self._database({"query": question.question})
        if CONFIG.features.fact_checking:
            response = self._fact_checker_db({"query": response["result"]})
        return QAAnswer(answer=response["result"])

    def set_prompt(self, config: PromptConfig) -> PromptConfig:
        names = {
            fn: fn for _, fn, _, _ in Formatter().parse(config.text) if fn is not None
        }
        config.parameters = PromptParameters(parameters=names)
        self._prompt_config = config
        return self._prompt_config

    def get_prompt(self) -> PromptConfig:
        return self._prompt_config or PromptConfig(text="")

    def generate(self, parameters: PromptParameters) -> GenResponse:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=CONFIG.model.name,
            model_file=CONFIG.model.file,
            model_type=CONFIG.model.type,
            hf=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )
        if self._prompt_config is None:
            self._prompt_config = PromptConfig(
                text="Bitte antworte, dass du keinen sinnvollen prompt erhalten hast."
            )
        prompt = self._prompt_config.text.format(**parameters.parameters)
        _LOGGER.debug("Resolved prompt: %s", prompt)

        response = pipe(
            prompt,
            do_sample=True,
            top_p=0.95,
            max_new_tokens=CONFIG.model.max_new_tokens,
        )
        response_content = response[0]
        letter_raw = response_content["generated_text"]
        # Cut string after matching keyword
        before1, match1, after1 = letter_raw.partition(
            "Generiere daraus das Dokument:",
        )
        # Cut string before matching keyword
        before2, match2, after2 = after1.partition("assistant")
        # Output relevant model answer
        generated_cover_letter = before2
        return GenResponse(text=generated_cover_letter)


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
