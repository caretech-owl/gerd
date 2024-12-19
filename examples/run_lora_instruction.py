from pathlib import Path

from gerd.config import PROJECT_DIR, load_gen_config
from gerd.gen.chat_service import ChatService
from gerd.training.lora import load_training_config

if Path().cwd() != PROJECT_DIR:
    msg = "This example must be run from the project root."
    raise AssertionError(msg)

lora_config = load_training_config("lora_instruct_example")

if not lora_config.output_dir.exists():
    msg = (
        f"LoRA config not found at {lora_config.output_dir}. "
        "Make sure to run 'train_lora_instruction.py' first."
    )
    raise FileNotFoundError(msg)

config = load_gen_config("chat_llama_3_2_1b_instruct")

print("Ohne LoRA\n=========")  # noqa: T201
chat = ChatService(config)
res = chat.submit_user_message({"content": "Bitte erweitere die Zahl 42."})
print(res.text)  # noqa: T201

print("\n\nMit LoRA\n========")  # noqa: T201
config.model.loras.append(lora_config.output_dir)
chat = ChatService(config)
res = chat.submit_user_message({"content": "Bitte erweitere die Zahl 42."})
print(res.text)  # noqa: T201
