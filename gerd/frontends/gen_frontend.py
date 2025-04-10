"""A gradio frontend to interact with the generation service.

This frontend is tailored to the letter of discharge generation task.
For a more general frontend see [`gerd.frontend.generate`][gerd.frontends.generate].
"""

import logging
from typing import Dict, Iterable, Tuple

import gradio as gr

from gerd.backends import TRANSPORTER

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

_field_labels = {
    "doctor_name": "Name des behandelnden Hausarztes",
    "attending_physician": "Name des behandelnden Hausarztes",
    "patient_name": "Name des Patienten",
    "hospital": "Name des Krankenhauses",
    "date_of_stay": "Datum des Aufenthalts",
    "diagnosis": "Diagnose",
    "anamnesis": "Anamnese",
    "findings": "Befunde",
    "treatment": "Behandlung",
    "medication": "Medikation",
    "patient_birth_date": "Geburtsdatum des Patienten",
    "patient_address": "Adresse des Patienten",
}

sections = ["Anrede", "Anamnese", "Diagnose", "Behandlung", "Medikation", "Grußformel"]

logging.basicConfig(level=logging.DEBUG)


def _pairwise(
    fields: Tuple[str, ...],
) -> Iterable[Tuple[str, str, str]]:
    a = iter(fields)
    return zip(a, a, a, strict=True)


def generate(*fields: str) -> Tuple[str, str, gr.TextArea, gr.Button]:
    """Generate a letter of discharge based on the provided fields.

    Parameters:
        *fields: The fields to generate the letter of discharge from.

    Returns:
        The generated letter of discharge, a text area to display it,
        and a button state to continue the generation
    """
    params = {}
    for key, name, value in _pairwise(fields):
        if not value:
            msg = f"Feld '{name}' darf nicht leer sein!"
            raise gr.Error(msg)
        params[key] = value
    response = TRANSPORTER.generate(params)
    return (
        response.text,
        response.text,
        gr.TextArea(label="Dokument", interactive=True),
        gr.Button("Kontinuiere Dokument", visible=True),
    )


def compare_paragraphs(src_doc: str, mod_doc: str) -> Dict[str, str]:
    """Compare paragraphs of two documents and return the modified parts.

    Parameters:
        src_doc: The source document
        mod_doc: The modified document

    Returns:
        The modified parts of the document
    """
    mod_parts = {}
    src_doc_split = src_doc.split("\n\n")
    mod_doc_split = mod_doc.split("\n\n")
    for section_order, src_para in zip(sections, src_doc_split, strict=True):
        mod_para = mod_doc_split[sections.index(section_order)]
        if src_para != mod_para:
            mod_parts[section_order] = mod_para
    return mod_parts


def insert_paragraphs(src_doc: str, new_para: Dict[str, str]) -> str:
    """Insert modified paragraphs into the source document.

    Parameters:
        src_doc: The source document
        new_para: The modified paragraphs

    Returns:
        The updated document
    """
    for section_order, mod_para in new_para.items():
        split_doc = src_doc.split("\n\n")[sections.index(section_order)]
        src_doc = src_doc.replace(split_doc, mod_para)
    return src_doc


def response_parser(response: str) -> Dict[str, str]:
    """Parse the response from the generation service.

    Parameters:
        response: The response from the generation service

    Returns:
        The parsed response
    """
    parsed_response = {}
    split_response = response.split("\n\n")
    for paragraph in split_response:
        for section in sections:
            if section in paragraph:
                parsed_response[section] = paragraph
                break
    return parsed_response


demo = gr.Blocks()

with demo:
    config = TRANSPORTER.get_gen_prompt()

    gr.Markdown("# GERD - Entlassbrief generieren")
    temp_output = gr.State(" ")

    fields = []
    for key in config.parameters:
        fields.append(gr.Textbox(key, visible=False))
        fields.append(gr.Textbox(_field_labels.get(key, key), visible=False))
        fields.append(gr.Textbox(label=_field_labels.get(key, key)))
    output = gr.TextArea(label="Dokument", interactive=False)
    submit_button = gr.Button("Generiere Dokument")
    continuation_button = gr.Button("Kontinuiere Dokument", visible=False)
    submit_button.click(
        fn=generate,
        inputs=fields,
        outputs=[output, temp_output, output, continuation_button],
    )

if __name__ == "__main__":
    demo.launch()
