from pathlib import Path

from gerd.config import PROJECT_DIR, load_gen_config
from gerd.gen.chat_service import ChatService
from gerd.training.lora import load_training_config

if Path().cwd() != PROJECT_DIR:
    msg = "This example must be run from the project root."
    raise AssertionError(msg)

config = load_gen_config("chat_llama_3_2_1b_instruct")

print("Ohne LoRA\n=========")  # noqa: T201
chat = ChatService(config)
res = chat.submit_user_message({"content": "Bitte erweitere die Zahl 42."})
print(res.text)  # noqa: T201

print("\n\nMit LoRA\n========")  # noqa: T201
config.model.loras.append(load_training_config("lora_instruct_example").output_dir)
chat = ChatService(config)
res = chat.submit_user_message({"content": "Bitte erweitere die Zahl 42."})
print(res.text)  # noqa: T201
