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

demo = gr.Blocks(title="GERD Instruct")

config = GenerationConfig()
lora_dir = LoraTrainingConfig().output_dir.parent


class Global:
    service: ChatService | None = None


def load_model(model_name: str, origin: str = "None") -> dict[str, Any]:
    if Global.service is not None:
        _LOGGER.debug("Unloading model")
        del Global.service
    model_config = config.model_copy()
    model_config.model.name = model_name
    if origin != "None" and (lora_dir / origin).is_dir():
        model_config.model.loras.add(lora_dir / origin)
    _LOGGER.info("Loading model %s", model_config.model.name)
    Global.service = ChatService(model_config)
    _LOGGER.info("Model loaded")
    return gr.update(interactive=True)


def generate(
    temperature: float,
    top_p: float,
    max_tokens: int,
    system_text: str,
    *args: str,
) -> str:
    fields = dict(zip(config.model.prompt_config.parameters, args, strict=True))
    if Global.service is None:
        msg = "Model not loaded"
        raise gr.Error(msg)
    if system_text:
        Global.service.config.model.prompt_setup = [
            ("system", PromptConfig(text=system_text))
        ]
    Global.service.config.model.top_p = top_p
    Global.service.config.model.temperature = temperature
    Global.service.config.model.max_new_tokens = max_tokens
    Global.service.reset()
    return Global.service.submit_user_message(fields).text


with demo:
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("# 🤖 GERD - Instruct 🛠️")
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
                step=0.01,
            )
            max_tokens = gr.Slider(
                label="Max Tokens",
                minimum=10,
                maximum=500,
                value=config.model.max_new_tokens,
                step=10,
            )
        btn_load = gr.Button("Load")
        text_system = gr.Textbox(
            label="System",
            lines=3,
            value=(
                config.model.prompt_setup[0][1].text
                if config.model.prompt_setup
                else ""
            ),
        )
    if KIOSK_MODE:
        _LOGGER.info("Kiosk mode:\n%s", config.model)
    n_lines = 3 if len(config.model.prompt_config.parameters) < 2 else 1

    input_items = [temp, top_p, max_tokens, text_system]
    with gr.Group():
        input_items.extend(
            [
                gr.Textbox(label=param, lines=n_lines)
                for param in config.model.prompt_config.parameters
            ]
        )
    btn_generate = gr.Button("Generate", interactive=Global.service is not None)
    text_out = gr.Textbox(label="Output", lines=5)
    btn_generate.click(
        generate,
        inputs=input_items,
        outputs=text_out,
    )
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
        load_model(config.model.name)
        btn_generate.interactive = True

    demo.load(
        lambda: gr.update(interactive=Global.service is not None), outputs=btn_generate
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gerd").setLevel(logging.DEBUG)
    _LOGGER.setLevel(logging.DEBUG)
    demo.launch()
