import logging

from gerd.config import load_gen_config
from gerd.gen.chat_service import ChatService

logging.basicConfig(level=logging.DEBUG)

# For remoter server example
gen = ChatService(load_gen_config("gen_chaining"))
res = gen.generate({"question": "Welches Tier legt die größten Eier?"})
print(res)  # noqa: T201
