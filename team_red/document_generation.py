import logging
from typing import Dict

import streamlit as st

from team_red.backend import BACKEND
from team_red.backend.interface import PromptConfig, PromptParameters

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


def document_generation() -> None:
    # Define the Streamlit app layout
    st.title("Dokument-Generator mit Llama 2")

    with st.form("Form zur Generierung eines Dokumentes"):
        # User input for letter of dismissal
        st.markdown("### Details")
        if BACKEND is None:
            _LOGGER.error("Backend has not been set!")
            return
        config = BACKEND.set_prompt(PromptConfig(text=PROMPT))
        fields = {}
        if not config.parameters:
            config.parameters = PromptParameters(parameters={})
        for key, value in config.parameters.parameters.items():
            fields[key] = st.text_input(value)

        # Generate LLM repsonse
        generate_cover_letter = st.form_submit_button("Generiere Dokument")

    if generate_cover_letter:
        with st.spinner("Generiere Dokument..."):
            response = BACKEND.generate(PromptParameters(parameters=fields))

        st.success("Fertig!")
        st.subheader("Generiertes Dokument:")
        st.text(response.text)

        # Offering download link for generated cover letter
        st.subheader("Download generiertes Dokument:")
        st.download_button(
            "Download generiertes Dokument als .txt",
            response.text,
            key="cover_letter",
        )
