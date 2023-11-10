import logging

import streamlit as st

from team_red.backend import TRANSPORTER
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


def gen_frontend() -> None:
    # Define the Streamlit app layout
    st.title("Dokument-Generator mit Llama 2")

    with st.form("Form zur Generierung eines Dokumentes"):
        # User input for letter of dismissal
        st.markdown("### Details")
        config = TRANSPORTER.set_gen_prompt(PromptConfig(text=PROMPT))
        fields = {}
        if not config.parameters:
            config.parameters = {}
        for key, value in config.parameters.items():
            fields[key] = st.text_input(_field_labels.get(key, key), value=value)

        # Generate LLM repsonse
        generate_cover_letter = st.form_submit_button("Generiere Dokument")

    if generate_cover_letter:
        with st.spinner("Generiere Dokument..."):
            response = TRANSPORTER.generate(fields)

        st.success("Fertig!")
        st.subheader("Generiertes Dokument:")
        st.markdown(response.text)

        # Offering download link for generated cover letter
        st.subheader("Download generiertes Dokument:")
        st.download_button(
            "Download generiertes Dokument als .txt",
            response.text,
            key="cover_letter",
        )
