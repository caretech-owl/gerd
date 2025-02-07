"""A [gradio](https://www.gradio.app/) frontend to train LoRAs with."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import gradio as gr

from gerd.config import CONFIG
from gerd.models.model import ModelConfig
from gerd.training.instruct import InstructTrainingData
from gerd.training.instruct import train_lora as train_instruct
from gerd.training.lora import (
    LoraModules,
    LoraTrainingConfig,
    TrainingFlags,
)
from gerd.training.trainer import Trainer
from gerd.training.unstructured import train_lora as train_unstructured

_LOGGER = logging.getLogger(__name__)

KIOSK_MODE = CONFIG.kiosk_mode


class Global:
    """A singleton class handle to store the current trainer instance."""

    trainer: Trainer | None = None


demo = gr.Blocks(title="GERD Training")
default_config = LoraTrainingConfig()
gr.set_static_paths(paths=[default_config.output_dir.parent])
_LOGGER.info(
    "LoRA output directory: %s", default_config.output_dir.parent.absolute().as_posix()
)


# Should this be limited to relative files or allowed directories
def get_file_list(glob_pattern: str) -> str:
    """Get a list of files matching the glob pattern.

    Parameters:
        glob_pattern: The glob pattern to search for files

    Returns:
        A string with the list of files
    """
    if not glob_pattern:
        return ""
    res = [str(f) for f in Path().glob(glob_pattern)]
    return "\n".join(res) if res else "<No files found>"


def check_trainer() -> dict[str, Any]:
    """Check if the trainer is (still) running.

    When the trainer is running, a progress bar is shown.
    The method returns a gradio property update of 'visible'
    which can be used to activate and deactivate elements based
    on the current training status.

    Returns:
        A dictionary with the status of gradio 'visible' property
    """
    if Global.trainer is not None:
        progress = gr.Progress()
        progress(0)
        while Global.trainer is not None and Global.trainer.thread.is_alive():
            max_steps = max(Global.trainer.tracked.max_steps, 1)
            progress(Global.trainer.tracked.current_steps / max_steps)
            time.sleep(0.5)
        return gr.update(visible=True, value="Training complete")
    return gr.update(visible=False)


def start_training(
    files: list[str] | None,
    model_name: str,
    lora_name: str,
    mode: str,
    data_source: str,
    input_glob: str,
    override: bool,
    modules: list[str],
    flags: list[str],
    epochs: int,
    batch_size: int,
    micro_batch_size: int,
    cutoff_len: int,
    overlap_len: int,
) -> str:
    """Start the training process.

    While training, the method will update the progress bar.

    Parameters:
        files: The list of files to train on
        model_name: The name of the model to train
        lora_name: The name of the LoRA to train
        mode: The training mode
        data_source: The source of the data
        input_glob: The glob pattern to search for files
        override: Whether to override existing models
        modules: The modules to train
        flags: The flags to set
        epochs: The number of epochs to train
        batch_size: The batch size
        micro_batch_size: The micro batch size
        cutoff_len: The cutoff length
        overlap_len: The overlap length

    Returns:
        A string with the status of the training
    """
    if ".." in lora_name:
        msg = "Invalid LoRA name"
        raise gr.Error(msg)
    progress = gr.Progress()
    # a bit verbose conversion to appease mypy
    # since training_mode is a Literal type
    if mode.lower() not in ["unstructured", "instructions"]:
        msg = f"Invalid training mode '{mode}'"
        raise AssertionError(msg)
    training_mode: Literal["unstructured", "instructions"] = (
        "unstructured" if mode.lower() == "unstructured" else "instructions"
    )
    train_config = LoraTrainingConfig(
        model=ModelConfig(name=model_name),
        mode=training_mode,
        output_dir=default_config.output_dir.parent / lora_name,
        override_existing=override,
        modules=LoraModules(**{mod: True for mod in modules}),
        flags=TrainingFlags(**{flag: True for flag in flags}),
        epochs=epochs,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        cutoff_len=cutoff_len,
        input_glob=input_glob,
        overlap_len=overlap_len,
        zip_output=True,
    )

    if files is None or not files and data_source == "Upload":
        msg = "No files uploaded"
        raise gr.Error(msg)
    progress(0)
    try:
        if train_config.mode == "unstructured":
            Global.trainer = train_unstructured(
                train_config,
                (
                    [Path(f).read_text() for f in files]
                    if data_source == "Upload"
                    else None
                ),
            )
        elif train_config.mode == "instructions":

            def load_data() -> InstructTrainingData:
                data = InstructTrainingData()
                for f in files:
                    with Path(f).open("r") as file:
                        data.samples.extend(
                            InstructTrainingData.model_validate(json.load(file)).samples
                        )
                return data

            Global.trainer = train_instruct(
                train_config, load_data() if data_source == "Upload" else None
            )
        else:
            msg = "Invalid training mode"
            raise AssertionError(msg)
    except AssertionError as e:
        msg = str(e)
        raise gr.Error(msg) from e
    return "Training started"


def validate_files(
    file_paths: list[str] | None, mode: str
) -> tuple[list[str], dict[str, bool]]:
    """Validate the uploaded files.

    Whether the property 'interactive' is True depends on whether
    any files were valid.
    Parameters:
        file_paths: The list of file paths
        mode: The training mode

    Returns:
        A tuple with the validated file paths and gradio property 'interactive'
    """
    file_paths = file_paths or []
    if mode.lower() == "instructions":

        def validate(path: Path) -> bool:
            if path.suffix != ".json":
                gr.Warning(f"{path.name} has an invalid file type")
                return False
            try:
                with path.open("r") as f:
                    InstructTrainingData.model_validate(json.load(f))
            except Exception as e:
                gr.Warning(f"{path.name} could not be validated: {e}")
                return False
            return True

    elif mode.lower() == "unstructured":

        def validate(path: Path) -> bool:
            if path.suffix != ".txt":
                gr.Warning(f"{path.name} has an invalid file type")
                return False
            return True

    else:
        gr.Error("Invalid training mode")
        return (file_paths, gr.update(interactive=False))

    res = [f_path for f_path in file_paths if validate(Path(f_path).resolve())]
    return (res, gr.update(interactive=len(res) > 0))


with demo:
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("# 🤖 GERD - LoRA Trainer 💪")
        with gr.Column(scale=1, visible=KIOSK_MODE):
            gr.Markdown(
                f"LoRA name: {default_config.output_dir.stem}<br>"
                f"Base model: {default_config.model.name}"
            )
    status = gr.Textbox("idle", label="Training Status", visible=False)

    with gr.Row(visible=not KIOSK_MODE):
        gr.Markdown("## Configuration")
        with gr.Column():
            text_model_name = gr.Textbox(
                label="Model Name",
                value=default_config.model.name,
            )
            select_training_mode = gr.Radio(
                label="Training mode",
                choices=["Instructions", "Unstructured"],
                value=default_config.mode.capitalize(),
            )
            # test_input_glob = gr.Textbox(
            #     label="File input glob", value=config.input_glob
            # )
            text_lora_name = gr.Textbox(
                label="LoRA Name", value=default_config.output_dir.stem
            )
            check_overriding = gr.Checkbox(
                label="Override existing adapter",
                value=default_config.override_existing,
            )
            # with gr.Accordion(label="Advanced data parameters", open=False):
            #     text_pad_token_id = gr.Number(
            #         label="Pad Token ID", value=str(config.pad_token_id)
            #     )
            #     select_padding_side = gr.Radio(
            #         label="Padding Side",
            #         choices=["right", "left"],
            #         value=config.padding_side,
            #     )

            #     text_train_only_after = gr.Textbox(
            #         label="Train Only After", value=config.train_only_after
            #     )

        with gr.Column():
            check_modules = gr.CheckboxGroup(
                label="Trained Modules",
                choices=[
                    mod_name
                    for mod_name in default_config.modules.model_dump()
                    if mod_name != "default"
                ],
                value=[
                    mod_name
                    for mod_name, v in default_config.modules.model_dump().items()
                    if mod_name != "default" and v is not False
                ],
            )
            check_flags = gr.CheckboxGroup(
                label="Flags",
                choices=[
                    flag_name
                    for flag_name in default_config.flags.model_dump()
                    if flag_name != "default"
                ],
                value=[
                    flag_name
                    for flag_name, v in default_config.flags.model_dump().items()
                    if flag_name != "default" and v is not False
                ],
            )
            slider_epochs = gr.Slider(
                label="Epochs",
                minimum=1,
                maximum=20,
                step=1,
                value=default_config.epochs,
            )
            slider_cutoff_len = gr.Slider(
                label="Cutoff Length",
                minimum=0,
                maximum=1024,
                step=1,
                value=default_config.cutoff_len,
                visible=select_training_mode == "Unstructured",
            )
            slider_overlap_len = gr.Slider(
                label="Overlap Length",
                minimum=0,
                maximum=1024,
                step=1,
                value=default_config.overlap_len,
                visible=select_training_mode == "Unstructured",
            )
            # slider_stop_at_loss = gr.Slider(
            #     label="Stop at Loss",
            #     minimum=0,
            #     maximum=2,
            #     step=0.1,
            #     value=config.stop_at_loss,
            # )
            slider_batch_size = gr.Slider(
                label="Batch Size",
                minimum=4,
                maximum=1024,
                step=4,
                value=default_config.batch_size,
            )
            slider_micro_batch_size = gr.Slider(
                label="Micro Batch Size",
                minimum=1,
                maximum=128,
                step=1,
                value=default_config.micro_batch_size,
            )
            # with gr.Accordion(label="Advanced training parameters", open=False):
            #     select_optimizer = gr.Radio(
            #         label="Optimizer",
            #         choices=[
            #             "adamw_hf",
            #             "adamw_torch",
            #             "adamw_torch_fused",
            #             "adamw_torch_xla",
            #             "adamw_apex_fused",
            #             "adafactor",
            #             "adamw_bnb_8bit",
            #             "adamw_anyprecision",
            #             "sgd",
            #             "adagrad",
            #         ],
            #         value=config.optimizer,
            #     )
            #     select_lr_scheduler = gr.Radio(
            #         label="Learning Rate Scheduler",
            #         choices=[
            #             "linear",
            #             "constant",
            #             "constant_with_warmup",
            #             "cosine",
            #             "cosine_with_restarts",
            #             "polynomial",
            #             "inverse_sqrt",
            #         ],
            #         value=config.lr_scheduler,
            #     )
            #     slider_lora_rank = gr.Slider(
            #         label="Lora Rank",
            #         minimum=1,
            #         maximum=1024,
            #         step=1,
            #         value=config.lora_rank,
            #     )
            #     slider_lora_alpha = gr.Slider(
            #         label="Lora Alpha",
            #         minimum=1,
            #         maximum=1024,
            #         step=1,
            #         value=config.lora_alpha,
            #     )
            #     slider_lora_dropout = gr.Slider(
            #         label="Lora Dropout",
            #         minimum=0,
            #         maximum=1,
            #         step=0.01,
            #         value=config.lora_dropout,
            #     )
            #     slider_save_steps = gr.Slider(
            #         label="Save Steps",
            #         minimum=0,
            #         maximum=5000,
            #         step=50,
            #         value=config.save_steps,
            #     )
            #     slider_warump_steps = gr.Slider(
            #         label="Warmup Steps",
            #         minimum=0,
            #         maximum=1000,
            #         step=50,
            #         value=config.warmup_steps,
            #     )
            #     slider_r = gr.Slider(
            #         label="R", minimum=1, maximum=20, step=1, value=config.r
            #     )
            #     select_bias = gr.Radio(
            #         label="Bias",
            #         choices=["none", "all", "lora_only"],
            #         value=config.bias,
            #     )
            #     # task_type: str = "CAUSAL_LM"
            #     slider_learning_rate = gr.Slider(
            #         label="Learning Rate",
            #         minimum=1e-6,
            #         maximum=1e-2,
            #         step=1e-4,
            #         value=config.learning_rate,
            #     )

    gr.Markdown("## Data")
    select_data_origin = gr.Radio(
        label="Data origin",
        choices=["Upload", "Path"],
        value="Upload",
    )
    select_training_mode.change(
        lambda x: tuple(gr.update(visible=x == "Unstructured") for _ in range(2)),
        inputs=select_training_mode,
        outputs=[slider_cutoff_len, slider_overlap_len],
    )
    file_ext = "txt" if select_training_mode.value == "Unstructured" else "json"
    file_upload = gr.Files(
        label=f"Upload {file_ext} files",
        file_types=(
            [".txt"] if select_training_mode.value == "Unstructured" else [".json"]
        ),
        file_count="multiple",
        height=200,
    )
    text_glob_pattern = gr.Textbox(
        label=f"Glob pattern (should end with .{file_ext})",
        value=default_config.input_glob,
        placeholder=f"file://data/*.{file_ext}",
        visible=select_data_origin == "Path",
    )
    text_training_files = gr.Textbox(
        label="Training files", visible=select_data_origin == "Path"
    )
    text_glob_pattern.blur(
        get_file_list,
        inputs=text_glob_pattern,
        outputs=text_training_files,
    )
    select_data_origin.change(
        lambda x: [gr.update(visible=x == "Upload")]
        + [gr.update(visible=x == "Path") for _ in range(2)],
        inputs=select_data_origin,
        outputs=[file_upload, text_glob_pattern, text_training_files],
    )

    gr.Markdown("## Workflow")
    validate_btn = gr.Button(
        "Validate",
        interactive=False,
    )
    train_btn = gr.Button("Train", interactive=False)

    def get_loras() -> dict[str, Path]:
        """Get a list of available LoRAs.

        LORAs are loaded from the path defined in the default
        [LoraTrainingConfig][gerd.training.lora.LoraTrainingConfig].

        Returns:
            A dictionary with the LoRA names as keys and the paths as values
        """
        return {
            path.stem: path
            for path in default_config.output_dir.parent.iterdir()
            if path.is_file() and path.suffix == ".zip"
        }

    gr.Markdown("## Result")
    dl_lora_choose = gr.Dropdown(
        label="Download LoRA",
        choices=list(get_loras().keys()),
        value=None,
    )
    dl_lora_btn = gr.DownloadButton("Download", interactive=False)
    dl_lora_choose.change(
        lambda x: gr.update(
            interactive=x is not None, value=get_loras()[x] if x else None
        ),
        inputs=dl_lora_choose,
        outputs=dl_lora_btn,
    )

    # this is a hack to update the available LoRAs after training
    # when done with focus, the dropwdown will not render correctly
    dl_lora_choose.blur(
        lambda: gr.update(choices=list(get_loras().keys())),
        outputs=dl_lora_choose,
    )

    validate_btn.click(
        validate_files,
        inputs=[file_upload, select_training_mode],
        outputs=[file_upload, train_btn],
    )

    train_btn.click(
        fn=lambda: tuple(gr.update(interactive=False) for i in range(3)),
        inputs=None,
        outputs=[train_btn, validate_btn, file_upload],
    ).then(lambda: gr.update(visible=True), outputs=status).then(
        start_training,
        inputs=[
            file_upload,
            text_model_name,
            text_lora_name,
            select_training_mode,
            select_data_origin,
            text_glob_pattern,
            check_overriding,
            check_modules,
            check_flags,
            slider_epochs,
            # slider_stop_at_loss,
            slider_batch_size,
            slider_micro_batch_size,
            # select_optimizer,
            # select_lr_scheduler,
            # slider_lora_rank,
            # slider_lora_alpha,
            # slider_lora_dropout,
            # slider_save_steps,
            # slider_warump_steps,
            # slider_r,
            # select_bias,
            # slider_learning_rate,
            # text_pad_token_id,
            # select_padding_side,
            slider_cutoff_len,
            slider_overlap_len,
            # text_train_only_after,
        ],
        outputs=status,
    ).then(
        check_trainer,
        outputs=status,
        concurrency_id="check_trainer",
        concurrency_limit=100,
    ).then(
        fn=lambda: tuple(gr.update(interactive=True) for _ in range(3)),
        inputs=None,
        outputs=[train_btn, validate_btn, file_upload],
    )

    demo.load(lambda: gr.update(visible=True), outputs=status).then(
        check_trainer, outputs=status
    ).then(
        fn=lambda: tuple(gr.update(interactive=True, visible=True) for _ in range(2)),
        inputs=None,
        outputs=[validate_btn, file_upload],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gerd").setLevel(logging.DEBUG)
    demo.launch()
