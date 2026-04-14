"""RAG Frontend for GERD.

This module implements a Gradio-based frontend
for the Retrieval-Augmented Generation (RAG) system in GERD.
It allows users to upload documents, ask questions,
and view relevant sources and answers.
"""

import logging
import pathlib
import threading
from typing import Optional

import gradio as gr
from langchain_community.vectorstores import FAISS

from gerd.backends import TRANSPORTER
from gerd.config import CONFIG, load_qa_config
from gerd.transport import QAFileUpload, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())
_MODEL_SWITCH_LOCK = threading.Lock()
_CURRENT_MODEL = load_qa_config().model.name


# -----------------------------
# Model Selection
# -----------------------------
def change_model(name: str) -> None:
    """Change the LLM model used for QA.

    Updates the QA service with the selected model name.

    Parameter:
        name (str): The name of the model to switch to.

    Returns:
        none
    """
    global _CURRENT_MODEL
    with _MODEL_SWITCH_LOCK:
        if name == _CURRENT_MODEL:
            return
        _LOGGER.info("Changing model to: %s", name)
        qa_config = load_qa_config()
        qa_config.model.name = name
        TRANSPORTER.reinit_qa_service(qa_config)
        _CURRENT_MODEL = name
        _LOGGER.info("Model successfully switched to: %s", name)


store_set: set[str] = set()


# -----------------------------
# Upload Files
# -----------------------------
def files_changed(file_paths: Optional[list[str]]) -> None:
    """Check if the file upload element has changed.

    If so, upload the new files to the vectorstore and delete the one that
    have been removed.

    Parameters:
        file_paths: The file paths to upload
    """
    file_paths = file_paths or []
    progress = gr.Progress()
    new_set = set(file_paths)
    new_files = new_set - store_set
    delete_files = store_set - new_set
    for new_file in new_files:
        store_set.add(new_file)
        with pathlib.Path(new_file).open("rb") as file:
            data = QAFileUpload(
                data=file.read(),
                name=pathlib.Path(new_file).name,
            )
        res = TRANSPORTER.add_file(data)
        if res.status != 200:
            _LOGGER.warning(
                "Data upload failed with error code: %d\nReason: %s",
                res.status,
                res.error_msg,
            )
            msg = (
                f"Datei konnte nicht hochgeladen werden: {res.error_msg}"
                "(Error Code {res.status})"
            )
            raise gr.Error(msg)
    for delete_file in delete_files:
        store_set.remove(delete_file)
        res = TRANSPORTER.remove_file(pathlib.Path(delete_file).name)
    progress(100, desc="Fertig!")


# -----------------------------
# Query LLM
# -----------------------------
def query(
    question: str, k_source: int, strategy: str, no_think: bool, model_name: str
) -> tuple[str, str]:
    """Handle the QA query.

    Parameters:
        question (str): The user's question.
        k_source (int): The number of sources to retrieve.
        strategy (str): The search strategy to use ("similarity" or "mmr").
        no_think (bool): Whether to disable the "thinking" step in the LLM.

    Returns:
        tuple[str, str]: A tuple containing the answer and the relevant sources.
    """
    # Guarantees query and model selection stay in sync even if events race.
    if model_name != _CURRENT_MODEL:
        _LOGGER.warning(
            "Model mismatch detected before query (%s != %s). Applying switch first.",
            model_name,
            _CURRENT_MODEL,
        )
        change_model(model_name)

    gesamt_context = ""
    _LOGGER.info("no_think: %s", no_think)
    q = QAQuestion(
        question=question,
        search_strategy=strategy,
        max_sources=k_source,
        no_think=no_think,
    )

    try:
        context = TRANSPORTER.db_query(q)
    except Exception as e:
        context = f"Source retrieval error: {e}"
    for cnt in context:
        # _LOGGER.info("Retrieved source: %s", cnt.content[:100])
        gesamt_context += cnt.content + "\n" + "******************************" + "\n\n"

    try:
        qa_res = TRANSPORTER.qa_query(q)

        if qa_res.status != 200:
            error_msg = f"Query failed: {qa_res.error_msg} (Code {qa_res.status})"
            raise gr.Error(error_msg) from None
        return qa_res.response, gesamt_context

    except Exception as e:
        error_msg = f"QA Query failed: {str(e)}"
        raise gr.Error(error_msg) from e


# -----------------------------
# Gradio UI
# -----------------------------
demo = gr.Blocks(title="GERD - RAG Frontend")

with demo:
    gr.Markdown("# GERD - RAG Frontend")
    gr.Markdown("Retrieval-Augmented Generation QA System")

    with gr.Row():
        with gr.Column(scale=2):
            file_upload = gr.Files(
                file_types=[".txt", ".pdf"], label="Upload Documents"
            )
        with gr.Column(scale=2):
            think_box = gr.Checkbox(value=False, label="no_think Mode")
            type_radio = gr.Radio(
                choices=["qwen/qwen2.5-0.5B-instruct", "qwen/qwen3-0.6B"],
                value="qwen/qwen3-0.6B",
                label="Model",
            )
            k_slider = gr.Slider(
                minimum=1, maximum=10, step=1, value=3, label="Number of Sources"
            )
            strategy_dropdown = gr.Dropdown(
                choices=["similarity", "mmr"],
                value="similarity",
                label="Search Strategy",
            )
    question_box = gr.Textbox(label="Question", placeholder="Ask a question...")

    with gr.Row():
        source_box = gr.Textbox(label="Relevant Sources", lines=10)
        answer_box = gr.Textbox(label="Answer", lines=10)

    upload_status = gr.Textbox(label="Upload Status")
    submit_btn = gr.Button("Submit", variant="primary")

    # Events
    file_upload.upload(fn=files_changed, inputs=file_upload, outputs=upload_status)
    file_upload.delete(fn=files_changed, inputs=file_upload, outputs=upload_status)
    file_upload.clear(fn=files_changed, inputs=file_upload, outputs=upload_status)
    type_radio.change(fn=change_model, inputs=type_radio)
    submit_btn.click(
        fn=query,
        inputs=[question_box, k_slider, strategy_dropdown, think_box, type_radio],
        outputs=[answer_box, source_box],
    )
    question_box.submit(
        fn=query,
        inputs=[question_box, k_slider, strategy_dropdown, think_box, type_radio],
        outputs=[answer_box, source_box],
    )
# -----------------------------
# Start App
# -----------------------------
if __name__ == "__main__":
    from gerd.config import CONFIG

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gerd").setLevel(CONFIG.logging.level.value.upper())
    demo.launch()
