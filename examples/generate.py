import logging

from gerd.config import load_gen_config
from gerd.gen.chat_service import ChatService
from gerd.models.model import PromptConfig

logging.basicConfig(level=logging.DEBUG)

# For remoter server example
gen = ChatService(load_gen_config("chat_llama_3_1_abl"))
res = gen.generate({"word": "Teleportation"})
print(res)  # noqa: T201
