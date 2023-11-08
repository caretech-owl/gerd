import logging
import pathlib

import streamlit as st

from team_red.backend import TRANSPORTER
from team_red.config import CONFIG
from team_red.transport import FileTypes, QAFileUpload, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def qa_frontend() -> None:
    st.title("Entlassbriefe QA")

    st.divider()

    uploaded_file = st.file_uploader(
        "Ein Dokument hochladen:", accept_multiple_files=False, type=["pdf", "txt"]
    )

    if uploaded_file is not None:
        res = TRANSPORTER.add_file(
            QAFileUpload(
                data=uploaded_file.getvalue(),
                type=FileTypes(pathlib.Path(uploaded_file.name).suffix[1:]),
            )
        )
        if res.status != 200:
            _LOGGER.warning(
                "Data upload failed with error code: %d\nReason: %s",
                res.status,
                res.error_msg,
            )

    with st.form("user_form", clear_on_submit=False):
        question = st.text_input("Stellen Sie Ihre Frage: ", value="")
        submit_button = st.form_submit_button(label="Frage stellen")

    if submit_button:
        with st.spinner(f"{CONFIG.model.name} generiert Antwort..."):
            answer = TRANSPORTER.qa_query(QAQuestion(question=question))
            if answer.status == 200:
                st.success(f"Antwort: {answer.answer}")
            else:
                st.error(f"{answer.error_msg} (ErrorCode {answer.status})")
