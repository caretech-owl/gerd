import logging

from gerd.config import load_gen_config
from gerd.gen.chat_service import ChatService

logging.basicConfig(level=logging.DEBUG)

conf = load_gen_config("chat_openai")

# For remoter server example
chat = ChatService(load_gen_config("chat_openai").model)
res = chat.submit_user_message("Hello, how are you?", add_prompt=True)
logging.info(res)
