import logging

from gerd.config import load_gen_config
from gerd.gen.chat_service import ChatService
from gerd.models.model import PromptConfig

logging.basicConfig(level=logging.DEBUG)

chat = ChatService(load_gen_config("chat_llama_3_1_abl"))
res = chat.submit_user_message({"word": "Teleportation"})
logging.info(res)

chat.set_prompt(PromptConfig.model_validate({"text": "{message}"}), "user")
res = chat.submit_user_message({"message": "Hallo! Was ist 1+1?"})
logging.info(res)
