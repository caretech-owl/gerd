"""A chaining example for the chat service."""

import logging

from gerd.config import load_gen_config
from gerd.gen.chat_service import ChatService

logging.basicConfig(level=logging.DEBUG)

gen = ChatService(load_gen_config("gen_chaining"))
res = gen.generate({"question": "What type of mammal lays the biggest eggs?"})
print(f"Result: {res.text}")  # noqa: T201
