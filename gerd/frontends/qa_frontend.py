"""A gradio frontend to query the QA service and upload files to the vectorstore."""

import logging
import pathlib
from typing import Any, Dict, List, Optional

import gradio as gr

from gerd.backends import TRANSPORTER
from gerd.transport import PromptConfig, QAFileUpload, QAModesEnum, QAQuestion

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

qa_modes_dict: Dict[str, QAModesEnum] = {
    "LLM": QAModesEnum.SEARCH,
    "Analyze": QAModesEnum.ANALYZE,
    "Analyze mult.": QAModesEnum.ANALYZE_MULT_PROMPTS,
    "VectorDB": QAModesEnum.NONE,
}


def get_qa_mode(search_type: str) -> QAModesEnum:
    """Get QAMode from string.

    Parameters:
        search_type: The search type

    Returns:
        The QAMode
    """
    if search_type in qa_modes_dict:
        return qa_modes_dict[search_type]
    else:
        return QAModesEnum.NONE


def query(question: str, search_type: str, k_source: int, search_strategy: str) -> str:
    """Starts the selected QA Mode.

    Parameters:
        question: The question to ask
        search_type: The search type
        k_source: The number of sources
        search_strategy: The search strategy

    Returns:
        The response from the QA service
    """
    q = QAQuestion(
        question=question, search_strategy=search_strategy, max_sources=k_source
    )
    # start search mode
    if search_type == "LLM":
        qa_res = TRANSPORTER.qa_query(q)
        if qa_res.status != 200:
            msg = (
                f"Query was unsuccessful: {qa_res.error_msg}"
                f" (Error Code {qa_res.status})"
            )
            raise gr.Error(msg)
        return qa_res.response
    # start analyze mode
    elif search_type == "Analyze":
        qa_analyze_res = TRANSPORTER.analyze_query()
        if qa_analyze_res.status != 200:
            msg = (
                f"Query was unsuccessful: {qa_analyze_res.error_msg}"
                f" (Error Code {qa_analyze_res.status})"
            )
            raise gr.Error(msg)
        # remove unwanted fields from answer
        qa_res_dic = {
            key: value
            for key, value in vars(qa_analyze_res).items()
            if value is not None
            and value != ""
            and key not in qa_analyze_res.__class__.__dict__
            and key != "sources"
            and key != "status"
            and key != "response"
            and key != "prompt"
        }
        qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())
        return qa_res_str
    # start analyze mult prompts mode
    elif search_type == "Analyze mult.":
        qa_analyze_mult_res = TRANSPORTER.analyze_mult_prompts_query()
        if qa_analyze_mult_res.status != 200:
            msg = (
                f"Query was unsuccessful: {qa_analyze_mult_res.error_msg}"
                f" (Error Code {qa_analyze_mult_res.status})"
            )
            raise gr.Error(msg)
        # remove unwanted fields from answer
        qa_res_dic = {
            key: value
            for key, value in vars(qa_analyze_mult_res).items()
            if value is not None
            and value != ""
            and key not in qa_analyze_mult_res.__class__.__dict__
            and key != "sources"
            and key != "status"
            and key != "response"
            and key != "prompt"
        }
        qa_res_str = ", ".join(f"{key}={value}" for key, value in qa_res_dic.items())
        return qa_res_str
    # start db search mode
    db_res = TRANSPORTER.db_query(q)
    if not db_res:
        msg = f"Database query returned empty!"
        raise gr.Error(msg)
    output = ""
    for doc in db_res:
        output += f"{doc.content}\n"
        output += f"({doc.name} / {doc.page})\n----------\n\n"
    return output


store_set: set[str] = set()


def files_changed(file_paths: Optional[list[str]]) -> None:
    """Check if the file upload element has changed.

    If so, upload the new files to the vectorstore and delete the one that
    have been removed.

    Parameters:
        file_paths: The file paths to upload
    """
    file_paths = file_paths or []
    progress = gr.Progress()
    new_set = set(file_paths)
    new_files = new_set - store_set
    delete_files = store_set - new_set
    for new_file in new_files:
        store_set.add(new_file)
        with pathlib.Path(new_file).open("rb") as file:
            data = QAFileUpload(
                data=file.read(),
                name=pathlib.Path(new_file).name,
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
    for delete_file in delete_files:
        store_set.remove(delete_file)
        res = TRANSPORTER.remove_file(pathlib.Path(delete_file).name)
    progress(100, desc="Fertig!")


def handle_type_radio_selection_change(search_type: str) -> List[Any]:
    """Enable/disable gui elements depend on which mode is selected.

    This order of the updates elements must be considered:
        - input text
        - prompt
        - k_slider
        - search strategy_dropdown

    Parameters:
        search_type: The current search type

    Returns:
        The list of GUI element property changes to update
    """
    if search_type == "LLM":
        return [
            gr.update(interactive=True, placeholder="Wie heißt der Patient?"),
            gr.update(value=TRANSPORTER.get_qa_prompt(get_qa_mode(search_type)).text),
            gr.update(interactive=False),
            gr.update(interactive=False),
        ]
    elif search_type == "VectorDB":
        return [
            gr.update(interactive=True, placeholder="Wie heißt der Patient?"),
            gr.update(value=TRANSPORTER.get_qa_prompt(get_qa_mode(search_type)).text),
            gr.update(interactive=True),
            gr.update(interactive=True),
        ]

    return [
        gr.update(interactive=False, placeholder=""),
        gr.update(value=TRANSPORTER.get_qa_prompt(get_qa_mode(search_type)).text),
        gr.update(interactive=False),
        gr.update(interactive=False),
    ]


def handle_developer_mode_checkbox_change(check: bool) -> List[Any]:
    """Enable/disable developermode.

    Enables or disables the developer mode and the corresponding GUI elements.
    Parameters:
        check: The current state of the developer mode checkbox
    Returns:
        The list of GUI element property changes to update
    """
    return [
        gr.update(visible=check),
        gr.update(visible=check),
        gr.update(visible=check),
        gr.update(visible=check),
        gr.update(
            choices=(
                ["LLM", "Analyze", "Analyze mult.", "VectorDB"]
                if check
                else ["LLM", "Analyze mult."]
            )
        ),
    ]


def set_prompt(
    prompt: str, search_type: str, progress: Optional[gr.Progress] = None
) -> None:
    """Updates the prompt of the selected QA Mode.

    Parameters:
        prompt: The new prompt
        search_type: The search type
        progress: The progress bar to update
    """
    if progress is None:
        progress = gr.Progress()
    progress(0, "Aktualisiere Prompt...")

    answer = TRANSPORTER.set_qa_prompt(
        PromptConfig(text=prompt), get_qa_mode(search_type)
    )
    if answer.error_msg:
        if answer.status != 200:
            msg = (
                f"Prompt konnte nicht aktualisiert werden: {answer.error_msg}"
                f" (Error Code {answer.status})"
            )
            raise gr.Error(msg)
        else:
            msg = f"{answer.error_msg}"
            gr.Warning(msg)
    else:
        gr.Info("Prompt wurde aktualisiert!")
    progress(100, "Fertig!")


demo = gr.Blocks(title="Entlassbriefe QA")


with demo:
    # define the GUI Layout
    developer_mode: bool = False

    gr.Markdown("# GERD - Entlassbriefe QA")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.Files(file_types=[".txt"])
        with gr.Column(scale=1):
            developer_checkbox = gr.Checkbox(
                info="Aktivieren/Deaktivieren von zusätzlichen Modi",
                label="Developer Mode",
                value=developer_mode,
            )
            type_radio = gr.Radio(
                choices=(
                    ["LLM", "Analyze", "Analyze mult.", "VectorDB"]
                    if developer_mode
                    else ["LLM", "Analyze mult."]
                ),
                value="LLM",
                label="Suchmodus",
                interactive=True,
            )
            k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=3,
                interactive=False,
                label="Quellenanzahl",
                visible=developer_mode,
            )
            strategy_dropdown = gr.Dropdown(
                choices=["similarity", "mmr"],
                value="similarity",
                interactive=False,
                label="Suchmodus",
                visible=developer_mode,
            )

    prompt = gr.TextArea(
        value=TRANSPORTER.get_qa_prompt(get_qa_mode(type_radio.value)).text,
        interactive=True,
        label="Prompt",
        visible=developer_mode,
    )
    prompt_submit = gr.Button("Aktualisiere Prompt", visible=developer_mode)
    inp = gr.Textbox(
        label="Stellen Sie eine Frage:", placeholder="Wie heißt der Patient?"
    )
    type_radio.change(
        fn=handle_type_radio_selection_change,
        inputs=type_radio,
        outputs=[inp, prompt, k_slider, strategy_dropdown],
    )
    out = gr.Textbox(label="Antwort")
    file_upload.upload(fn=files_changed, inputs=file_upload, outputs=out)
    file_upload.delete(fn=files_changed, inputs=file_upload, outputs=out)
    file_upload.clear(fn=files_changed, inputs=file_upload, outputs=out)
    btn = gr.Button("Frage stellen")
    btn.click(
        fn=query, inputs=[inp, type_radio, k_slider, strategy_dropdown], outputs=out
    )
    inp.submit(
        fn=query, inputs=[inp, type_radio, k_slider, strategy_dropdown], outputs=out
    )
    prompt_submit.click(fn=set_prompt, inputs=[prompt, type_radio], outputs=out)

    developer_checkbox.change(
        fn=handle_developer_mode_checkbox_change,
        inputs=developer_checkbox,
        outputs=[prompt, prompt_submit, k_slider, strategy_dropdown, type_radio],
    )

if __name__ == "__main__":
    from gerd.config import CONFIG

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gerd").setLevel(CONFIG.logging.level.value.upper())
    demo.launch()
