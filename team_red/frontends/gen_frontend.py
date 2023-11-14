import logging
from typing import Iterable, Tuple

import gradio as gr

from team_red.backend import TRANSPORTER

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

_field_labels = {
    "history": "Patientengeschichte",
    "doctor_name": "Name des behandelnden Hausarztes",
    "patient_name": "Name des Patienten",
    "hospital": "Name des Krankenhauses",
}


def _pairwise(
    fields: Tuple[gr.Textbox, ...]
) -> Iterable[Tuple[gr.Textbox, gr.Textbox, gr.Textbox]]:
    a = iter(fields)
    return zip(a, a, a)


def generate(*fields: gr.Textbox) -> str:
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
    config = TRANSPORTER.get_gen_prompt()

    gr.Markdown("# Entlassbrief generieren")

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
