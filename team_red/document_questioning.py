import argparse
import logging
import timeit

import box
import yaml
from .utils import setup_dbqa, setup_dbqa_fact_checking

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


# Import config vars
with open("config/config.yml", "r", encoding="utf8") as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def document_questioning() -> None:
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="Wie heißt der Patient?",
        help="Enter the query to pass into the LLM",
    )
    args = parser.parse_args()

    # Setup DBQA
    startQA = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({"query": args.input})
    endQA = timeit.default_timer()

    _LOGGER.debug(f'\nAnswer: {response["result"]}')
    _LOGGER.debug("=" * 50)

    # Process source documents
    source_docs = response["source_documents"]
    for i, doc in enumerate(source_docs):
        _LOGGER.debug(f"\nSource Document {i+1}\n")
        _LOGGER.debug(f"Source Text: {doc.page_content}")
        _LOGGER.debug(f'Document Name: {doc.metadata["source"]}')
        _LOGGER.debug(f'Page Number: {doc.metadata.get("page", 1)}\n')
        _LOGGER.debug("=" * 60)

    _LOGGER.debug(f"Time to retrieve response: {endQA - startQA}")

    if cfg.FACTCHECKING == True:
        startFactCheck = timeit.default_timer()
        dbqafact = setup_dbqa_fact_checking()
        response_fact = dbqafact({"query": response["result"]})
        endFactCheck = timeit.default_timer()
        _LOGGER.debug("Factcheck:")
        _LOGGER.debug(f'\nAnswer: {response_fact["result"]}')
        _LOGGER.debug("=" * 50)

        # Process source documents
        source_docs = response_fact["source_documents"]
        for i, doc in enumerate(source_docs):
            _LOGGER.debug(f"\nSource Document {i+1}\n")
            _LOGGER.debug(f"Source Text: {doc.page_content}")
            _LOGGER.debug(f'Document Name: {doc.metadata["source"]}')
            _LOGGER.debug(f'Page Number: {doc.metadata.get("page", 1)}\n')
            _LOGGER.debug("=" * 60)

        _LOGGER.debug(f"Time to retrieve fact check: {endFactCheck - startFactCheck}")

    logging.shutdown()
