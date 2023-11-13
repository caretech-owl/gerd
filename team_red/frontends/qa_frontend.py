import logging
import pathlib
from typing import Optional

import gradio as gr

from team_red.backend import TRANSPORTER
from team_red.config import CONFIG
from team_red.transport import FileTypes, PromptConfig, QAFileUpload, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


PROMPT = """Verwende die folgenden Informationen, \
um die Frage des Benutzers zu beantworten. Wenn du die Antwort nicht weißt, \
sag einfach, dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden.

Kontext: {context}
Frage: {question}

Gebe nur die hilfreiche Antwort unten zurück und nichts anderes. Halte dich außerdem \
sehr kurz mit der Antwort.
Hilfreiche Antwort:
"""


def query(question: str) -> str:
    res = TRANSPORTER.qa_query(QAQuestion(question=question))
    if res.status != 200:
        msg = f"Query was unsuccessful: {res.error_msg} (Error Code {res.status})"
        raise gr.Error(msg)
    return res.answer


def upload(file_path: str, progress: Optional[gr.Progress] = None) -> None:
    if not file_path:
        return
    if progress is None:
        progress = gr.Progress()
    progress(0, desc="Hochladen...")
    with pathlib.Path(file_path).open("rb") as file:
        data = QAFileUpload(
            data=file.read(),
            type=FileTypes(pathlib.Path(file_path).suffix[1:]),
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
    _ = TRANSPORTER.set_gen_prompt(PromptConfig(text=prompt))
    progress(100, "Fertig!")


demo = gr.Blocks(title="Entlassbriefe QA")


with demo:
    gr.Markdown("# Entlassbriefe QA")
    with gr.Row():
        file_upload = gr.File(file_count="single", file_types=[".txt"])
        with gr.Column():
            prompt = gr.TextArea(value=PROMPT, interactive=True, label="Prompt")
            prompt_submit = gr.Button("Aktualisiere Prompt")
    inp = gr.Textbox(
        label="Stellen Sie eine Frage:", placeholder="Wie heißt der Patient?"
    )
    out = gr.Textbox(label="Antwort")
    file_upload.change(fn=upload, inputs=file_upload, outputs=out)
    btn = gr.Button("Frage stellen")
    btn.click(fn=query, inputs=inp, outputs=out)
    inp.submit(fn=query, inputs=inp, outputs=out)
    prompt_submit.click(fn=set_prompt, inputs=prompt, outputs=out)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    demo.launch()
