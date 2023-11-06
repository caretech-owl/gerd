import logging
import pathlib
import timeit
from tempfile import NamedTemporaryFile

import streamlit as st

from team_red.backend.interface import QAQuestion

from .backend import BACKEND
from .config import CONFIG
from .utils import setup_dbqa, setup_dbqa_fact_checking

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
            answer = BACKEND.qa_query(QAQuestion(question=question))
            st.success(f"Antwort: {answer.answer}")
