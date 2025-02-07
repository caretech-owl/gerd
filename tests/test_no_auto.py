"""Tests that should not be run automatically.

These tests are processing-intensive and should not be run on the test runners
"""

import logging
from datetime import datetime

import pytest

from gerd.config import load_gen_config
from gerd.gen.generation_service import GenerationService
from gerd.loader import LlamaCppLLM, TransformerLLM
from gerd.qa.qa_service import QAService
from gerd.transport import QAQuestion

_LOGGER = logging.getLogger(__name__)


@pytest.mark.skip("Should only be run manually")
def test_fail() -> None:
    """Checks whether manual tests are correctly skipped."""
    pytest.fail("This test should not be executed.")


@pytest.mark.skip("Should only be run manually")
def test_manual_generation_transformers() -> None:
    """Tests text generation with a transformer model."""
    config = load_gen_config()
    config.model.name = "distilbert/distilgpt2"
    config.model.file = None
    config.model.prompt_config.text = "Say {the_word}!"
    config.model.temperature = 0.01
    config.model.max_new_tokens = 10
    gen = GenerationService(config)
    assert isinstance(gen._model, TransformerLLM)  # noqa: SLF001
    res = gen.generate({"the_word": "please"}, add_prompt=True)
    assert res.status == 200
    assert res.prompt == "Say please!"
    assert res.text


@pytest.mark.skip("Should only be run manually")
def test_manual_generation_llama_cpp() -> None:
    """Tests text generation with a (quantized) LlamaCpp model."""
    config = load_gen_config()
    config.model.name = "bartowski/Phi-3.1-mini-4k-instruct-GGUF"
    config.model.file = "Phi-3.1-mini-4k-instruct-IQ2_M.gguf"
    config.model.prompt_config.text = "Say {the_word}!"
    config.model.max_new_tokens = 10
    gen = GenerationService(config)
    assert isinstance(gen._model, LlamaCppLLM)  # noqa: SLF001
    res = gen.generate({"the_word": "please"}, add_prompt=True)
    assert res.status == 200
    assert res.prompt == "Say please!"
    assert res.text


@pytest.mark.skip("Should only be run manually")
def test_qa_queries(qa_service_file: QAService, query_questions: str) -> None:
    """Test the query method with multiple questions.

    Parameters:
        qa_service_file: The QAService fixture with a file loaded
        query_questions: The question to be tested
    """
    res = qa_service_file.query(QAQuestion(question=query_questions))

    txt_out = (
        "=" * 10
        + "\n"
        + "["
        + str(datetime.today())
        + "]\n"
        + "Question: "
        + query_questions
        + "\n"
        + "Answer: "
        + res.response
        + "\n"
        + "Model response: "
        + res.response
        + "\n"
        + "Sources: "
        + (" ".join(doc.content for doc in res.sources))
        + "\n"
        + "=" * 10
        + "\n"
    )
    _LOGGER.info(txt_out)
    assert res.status == 200


@pytest.mark.skip("Should only be run manually")
def test_qa_predef_multi(qa_service_file: QAService) -> None:
    """Test the analyze_mult_prompts_query method of the QAService class.

    Parameters:
        qa_service_file: The QAService fixture with a document loaded
    """
    res = qa_service_file.analyze_mult_prompts_query()

    # remove unwanted fields from answer
    qa_res_dic = {
        key: value
        for key, value in vars(res).items()
        if value is not None
        and value != ""
        and key not in res.__class__.__dict__
        and key != "sources"
        and key != "status"
    }
    qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())

    # write result to file
    txt_out = (
        "=" * 10
        + "\n"
        + "["
        + str(datetime.today())
        + "]\n"
        + "Answer: "
        + qa_res_str
        + "\n"
        + "Model response: "
        + res.response
        + "\n"
        + "Sources: "
        + ("; ".join(f"{doc.query}={doc.content}\n\n" for doc in res.sources))
        + "\n"
        + "=" * 10
        + "\n"
    )
    _LOGGER.info(txt_out)

    assert res.status == 200


@pytest.mark.skip("Should only be run manually")
def test_qa_predef_single(qa_service_file: QAService) -> None:
    """Test the analyze_query method of the QAService class.

    Parameters:
        qa_service_file: The QAService fixture with a document loaded
    """
    res = qa_service_file.analyze_query()

    # remove unwanted fields from answer
    qa_res_dic = {
        key: value
        for key, value in vars(res).items()
        if value is not None
        and value != ""
        and key not in res.__class__.__dict__
        and key != "sources"
        and key != "status"
    }
    qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())

    txt_out = (
        "=" * 10
        + "\n"
        + "["
        + str(datetime.today())
        + "]\n"
        + "Answer: "
        + qa_res_str
        + "\n"
        + "Model response: "
        + res.response
        + "\n"
        + "Prompt: "
        + res.prompt
        + "\n"
        + "Sources: "
        + ("; ".join(f"{doc.query}={doc.content}\n\n" for doc in res.sources))
        + "\n"
        + "=" * 10
        + "\n"
    )
    _LOGGER.info(txt_out)
    assert res.status == 200
