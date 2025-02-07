"""A simple gradio frontend to interact with the GERD chat and generate service."""

import json
import logging
from typing import Any

import gradio as gr

from gerd.config import CONFIG
from gerd.gen.chat_service import ChatService
from gerd.models.gen import GenerationConfig
from gerd.models.model import PromptConfig
from gerd.training.lora import LoraTrainingConfig

_LOGGER = logging.getLogger(__name__)

KIOSK_MODE = CONFIG.kiosk_mode
"""Whether the frontend is running in kiosk mode.

Kiosk mode reduces the number of options to a minimum and automatically loads the model.
"""

demo = gr.Blocks(title="GERD Generate")
config = GenerationConfig()
# Since this is a generation task we don't need prompt setup
config.model.prompt_setup = []
config.model.prompt_config = PromptConfig()
lora_dir = LoraTrainingConfig().output_dir.parent


class Global:
    """Singleton to store the service."""

    service: ChatService | None = None


def load_model(model_name: str, origin: str) -> dict[str, Any]:
    """Load a global large language model.

    Parameters:
        model_name: The name of the model
        origin: Whether to use an extra LoRA

    Returns:
        The updated interactive state, returns interactive=True when the model is loaded
    """
    if Global.service is not None:
        del Global.service
    model_config = config.model_copy()
    model_config.model.name = model_name
    if origin != "None":
        model_config.model.loras.add(lora_dir / origin)
    Global.service = ChatService(model_config)
    return gr.update(interactive=True)


def generate(textbox: str, temp: float, top_p: float, max_tokens: int) -> str:
    """Generate text from the model.

    Parameters:
        textbox: The text to generate from
        temp: The temperature for the generation
        top_p: The top p value for the generation
        max_tokens: The maximum number of tokens to generate

    Returns:
        The generated text
    """
    if Global.service is None:
        msg = "Model not loaded"
        raise gr.Error(msg)
    Global.service.config.model.top_p = top_p
    Global.service.config.model.temperature = temp
    Global.service.config.model.max_new_tokens = max_tokens
    return Global.service.generate({"message": textbox}).text


with demo:
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("# 🤖 GERD - Generate 📝")
        with gr.Column(scale=1):
            if KIOSK_MODE:
                gr.Markdown(f"Model: {config.model.name}")
                if config.model.loras:
                    not_found = set()
                    for lora in config.model.loras:
                        if not (lora / "adapter_model.safetensors").exists():
                            not_found.add(lora)
                    names = [
                        ("⚠️ " if path in not_found else "✅ ") + path.stem
                        for path in config.model.loras
                    ]
                    gr.Markdown(f"LORAs: {', '.join(names)}")
    with gr.Accordion("Parameter", visible=not KIOSK_MODE):
        with gr.Row():
            model_name = gr.Textbox(config.model.name, label="Model Name")
            origin = gr.Dropdown(
                label="LoRA",
                choices=["None"]
                + [path.stem for path in lora_dir.iterdir() if path.is_dir()],
            )
        with gr.Row():
            temp = gr.Slider(
                label="Temperature",
                minimum=0.05,
                maximum=1.0,
                value=config.model.temperature,
                step=0.05,
            )
            top_p = gr.Slider(
                label="Top P",
                minimum=0.1,
                maximum=1.0,
                value=config.model.top_p,
                step=0.05,
            )
            max_tokens = gr.Slider(
                label="Max Tokens",
                minimum=10,
                maximum=500,
                value=config.model.max_new_tokens,
                step=10,
            )
        btn_load = gr.Button("Load")
    textbox = gr.Textbox(label="Text", lines=10)
    btn_generate = gr.Button("Generate", interactive=Global.service is not None)
    btn_generate.click(lambda: gr.update(interactive=False), outputs=btn_generate).then(
        generate,
        inputs=[textbox, temp, top_p, max_tokens],
        outputs=textbox,
    ).then(lambda: gr.update(interactive=True), outputs=btn_generate)
    btn_load.click(load_model, inputs=[model_name, origin], outputs=btn_generate)
    origin.change(
        fn=lambda x, d: (
            LoraTrainingConfig.model_validate(
                json.loads((lora_dir / x / "training_parameters.json").read_text())
            ).model.name
            if x
            else d
        ),
        inputs=[origin, model_name],
        outputs=[model_name],
    )
    if KIOSK_MODE:
        load_model(config.model.name, "None")
        btn_generate.interactive = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gerd").setLevel(logging.DEBUG)
    demo.launch()
