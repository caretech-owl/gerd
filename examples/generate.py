"""Example of how to use the chat service to generate text."""

import logging

from gerd.config import load_gen_config
from gerd.gen.chat_service import ChatService

logging.basicConfig(level=logging.DEBUG)

gen = ChatService(load_gen_config("chat_llama_3_1_abl"))
res = gen.generate({"word": "Teleportation"})
print(res)  # noqa: T201
