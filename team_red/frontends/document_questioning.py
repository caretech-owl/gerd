import logging
import pathlib
from tempfile import NamedTemporaryFile

import streamlit as st

from team_red.backend import TRANSPORTER
from team_red.config import CONFIG
from team_red.transport import QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def document_questioning() -> None:
    st.title("Entlassbriefe QA")

    st.divider()

    uploaded_file = st.file_uploader(
        "Ein Dokument hochladen:", accept_multiple_files=False, type=["pdf", "txt"]
    )

    if uploaded_file is not None:
        filetype = pathlib.Path(uploaded_file.name).suffix
        buffer = uploaded_file.getbuffer()
        # BACKEND.upload_txt(buffer)

    with st.form("user_form", clear_on_submit=False):
        question = st.text_input("Stellen Sie Ihre Frage: ", value="")
        submit_button = st.form_submit_button(label="Frage stellen")

    if submit_button:
        with st.spinner(f"{CONFIG.model.name} generiert Antwort..."):
            answer = TRANSPORTER.qa_query(QAQuestion(question=question))
            st.success(f"Antwort: {answer.answer}")
