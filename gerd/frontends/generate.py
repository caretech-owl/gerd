import json
import logging
from typing import Any

import gradio as gr

from gerd.gen.chat_service import ChatService
from gerd.models.gen import GenerationConfig
from gerd.models.model import ModelConfig, PromptConfig
from gerd.training.lora import LoraTrainingConfig

_LOGGER = logging.getLogger(__name__)

demo = gr.Blocks(title="GERD")

config = GenerationConfig(
    model=ModelConfig(
        name="meta-llama/Llama-3.2-1B", prompt_config=PromptConfig(text="{message}")
    )
)
lora_dir = LoraTrainingConfig().output_dir.parent


class Global:
    service: ChatService | None = None


def load_model(model_name: str, origin: str) -> dict[str, Any]:
    if Global.service is not None:
        del Global.service
    model_config = config.model_copy()
    model_config.model.name = model_name
    if origin != "None":
        model_config.model.loras.add(lora_dir / origin)
    Global.service = ChatService(model_config)
    return gr.update(interactive=True)


def generate(textbox: str, temp: float, top_p: float, max_tokens: int) -> str:
    if Global.service is None:
        msg = "Model not loaded"
        raise gr.Error(msg)
    Global.service.config.model.top_p = top_p
    Global.service.config.model.temperature = temp
    Global.service.config.model.max_new_tokens = max_tokens
    return Global.service.generate({"message": textbox}).text


with demo:
    gr.Markdown("# GERD - Generate")
    with gr.Accordion("Parameter"):
        with gr.Row():
            model_name = gr.Textbox(config.model.name, label="Model Name")
            origin = gr.Dropdown(
                label="LoRA",
                choices=["None"]
                + [path.stem for path in lora_dir.iterdir() if path.is_dir()],
            )
        with gr.Row():
            temp = gr.Slider(
                label="Temperature", minimum=0.05, maximum=1.0, value=0.5, step=0.05
            )
            top_p = gr.Slider(
                label="Top P", minimum=0.1, maximum=1.0, value=0.9, step=0.05
            )
            max_tokens = gr.Slider(
                label="Max Tokens", minimum=10, maximum=500, value=50, step=10
            )
    btn_load = gr.Button("Load")
    textbox = gr.Textbox(label="Text", lines=10)
    btn_generate = gr.Button("Generate", interactive=Global.service is not None)
    btn_generate.click(
        generate,
        inputs=[textbox, temp, top_p, max_tokens],
        outputs=textbox,
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gerd").setLevel(logging.DEBUG)
    demo.launch()
