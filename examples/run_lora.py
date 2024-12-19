from pathlib import Path

from gerd.config import PROJECT_DIR, load_gen_config
from gerd.gen.chat_service import ChatService

if Path().cwd() != PROJECT_DIR:
    msg = "This example must be run from the project root."
    raise AssertionError(msg)

config = load_gen_config("chat_llama_3_2_1b")

print("Ohne LoRA\n=========")  # noqa: T201
chat = ChatService(config)
res = chat.generate({"text": "Sehr geehrte Frau Kollegin,"})
print(res.text)  # noqa: T201

print("\n\nMit LoRA\n========")  # noqa: T201
config.model.loras.append(("models/lora"))
chat = ChatService(config)
res = chat.generate({"text": "Sehr geehrte Frau Kollegin,"})
print(res.text)  # noqa: T201
