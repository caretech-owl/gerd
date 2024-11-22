import logging

from gerd.config import load_gen_config
from gerd.gen.chat_service import ChatService
from gerd.models.model import PromptConfig

logging.basicConfig(level=logging.WARNING)
logging.getLogger("gerd").setLevel(logging.DEBUG)

logging.info(
    "Loading chat service..."
    " When this is the first time you run this script, it will download the model."
    " This may take a few minutes."
)

chat = ChatService(load_gen_config("hello"))
res = chat.submit_user_message({"word": "teleportation"})
logging.info(res)

chat.set_prompt_config(PromptConfig.model_validate({"text": "{message}"}))
res = chat.submit_user_message({"message": "Hello! What is one plus one?"})
logging.info(res)
