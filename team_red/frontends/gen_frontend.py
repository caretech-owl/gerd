import logging
from typing import List, Tuple

import gradio as gr

from team_red.backend import TRANSPORTER
from team_red.config import CONFIG
from team_red.transport import PromptConfig

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

PROMPT = """Du bist ein hilfreicher Assistant.\
Du wandelst Eckdaten in ein fertiges Dokument um.
Du gibst ausschließlich das fertige Dokument zurück und nichts anderes.\
Die Eckdaten lauten wie folgt:
Der Aufenthaltsverlauf des Patienten: {history}\n
Der Name des Arztes der im Anschreiben angegeben werden soll: {doctor_name}\n
Der Name des Patienten, um den es geht: {patient_name}\n
Das Krankenhaus, bei dem der Patient behandelt wurde: {hospital}\n
Generiere daraus das Dokument:"""

_field_labels = {
    "history": "Patientengeschichte",
    "doctor_name": "Name des behandelnden Hausarztes",
    "patient_name": "Name des Patienten",
    "hospital": "Name des Krankenhauses",
}


def _pairwise(fields: Tuple[gr.Textbox]) -> List[gr.Textbox]:
    a = iter(fields)
    return zip(a, a, a)


def generate(*fields: Tuple[gr.Textbox]) -> str:
    params = {}
    for key, name, value in _pairwise(fields):
        if not value:
            msg = f"Feld '{name}' darf nicht leer sein!"
            raise gr.Error(msg)
        params[key] = value
    response = TRANSPORTER.generate(params)
    return response.text


demo = gr.Blocks()

with demo:
    config = TRANSPORTER.set_gen_prompt(PromptConfig(text=PROMPT))
    if not config.parameters:
        config.parameters = {}
    fields = []
    for key in config.parameters:
        fields.append(gr.Textbox(key, visible=False))
        fields.append(gr.Textbox(_field_labels.get(key, key), visible=False))
        fields.append(gr.Textbox(label=_field_labels.get(key, key)))
    output = gr.TextArea(label="Dokument")
    submit_button = gr.Button("Generiere Dokument")
    submit_button.click(fn=generate, inputs=fields, outputs=output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    demo.launch()
