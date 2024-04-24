import logging
import pathlib
from typing import Optional

import gradio as gr

from team_red.backend import TRANSPORTER
from team_red.transport import PromptConfig, QAFileUpload, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.DEBUG)


def query(question: str, search_type: str, k_source: int, search_strategy: str) -> str:
    q = QAQuestion(
        question=question, search_strategy=search_strategy, max_sources=k_source
    )
    if search_type == "LLM":
        qa_res = TRANSPORTER.qa_query(q)
        if qa_res.status != 200:
            msg = (
                f"Query was unsuccessful: {qa_res.error_msg}"
                f" (Error Code {qa_res.status})"
            )
            raise gr.Error(msg)
        return qa_res.answer
    elif search_type == "Analyze":
        qa_res = TRANSPORTER.analyze_query()
        if qa_res.status != 200:
            msg = (
                f"Query was unsuccessful: {qa_res.error_msg}"
                f" (Error Code {qa_res.status})"
            )
            raise gr.Error(msg)

        qa_res_dic = {key: value for key, value in vars(qa_res).items()
                      if value is not None
                      and value != ""
                      and key not in qa_res.__class__.__dict__
                      and key != "sources"
                      and key != "status"
        }
        qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())
        return qa_res_str
    elif search_type == "Analyze mult.":
        qa_res = TRANSPORTER.analyze_mult_prompts_query()
        if qa_res.status != 200:
            msg = (
                f"Query was unsuccessful: {qa_res.error_msg}"
                f" (Error Code {qa_res.status})"
            )
            raise gr.Error(msg)

        qa_res_dic = {key: value for key, value in vars(qa_res).items()
                      if value is not None
                      and value != ""
                      and key not in qa_res.__class__.__dict__
                      and key != "sources"
                      and key != "status"}
        qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())
        return qa_res_str
    db_res = TRANSPORTER.db_query(q)
    if not db_res:
        msg = f"Database query returned empty!"
        raise gr.Error(msg)
    output = ""
    for doc in db_res:
        output += f"{doc.content}\n"
        output += f"({doc.name} / {doc.page})\n----------\n\n"
    return output


def upload(file_path: str, progress: Optional[gr.Progress] = None) -> None:
    if not file_path:
        return
    if progress is None:
        progress = gr.Progress()
    progress(0, desc="Hochladen...")
    with pathlib.Path(file_path).open("rb") as file:
        data = QAFileUpload(
            data=file.read(),
            name=pathlib.Path(file_path).name,
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
    progress(100, desc="Fertig!")


def set_prompt(prompt: str, progress: Optional[gr.Progress] = None) -> None:
    if progress is None:
        progress = gr.Progress()
    progress(0, "Aktualisiere Prompt...")
    _ = TRANSPORTER.set_qa_prompt(PromptConfig(text=prompt))
    progress(100, "Fertig!")


demo = gr.Blocks(title="Entlassbriefe QA")


with demo:
    gr.Markdown("# Entlassbriefe QA")
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(file_count="single", file_types=[".txt"])
        with gr.Column(scale=1):
            type_radio = gr.Radio(
                choices=["LLM", "Analyze", "Analyze mult.", "VectorDB"],
                value="LLM",
                label="Suchmodus",
                interactive=True,
            )
            k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=3,
                interactive=True,
                label="Quellenanzahl",
            )
            strategy_dropdown = gr.Dropdown(
                choices=["similarity", "mmr"],
                value="similarity",
                interactive=True,
                label="Suchmodus",
            )
    prompt = gr.TextArea(
        value=TRANSPORTER.get_qa_prompt().text, interactive=True, label="Prompt"
    )
    prompt_submit = gr.Button("Aktualisiere Prompt")
    inp = gr.Textbox(
        label="Stellen Sie eine Frage:", placeholder="Wie heißt der Patient?"
    )
    out = gr.Textbox(label="Antwort")
    file_upload.change(fn=upload, inputs=file_upload, outputs=out)
    btn = gr.Button("Frage stellen")
    btn.click(
        fn=query, inputs=[inp, type_radio, k_slider, strategy_dropdown], outputs=out
    )
    inp.submit(
        fn=query, inputs=[inp, type_radio, k_slider, strategy_dropdown], outputs=out
    )
    prompt_submit.click(fn=set_prompt, inputs=prompt, outputs=out)

if __name__ == "__main__":
    demo.launch()
