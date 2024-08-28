import logging

from gerd.config import load_gen_config
from gerd.gen import GenerationService
from gerd.models.model import PromptConfig

logging.basicConfig(level=logging.DEBUG)

# For remoter server example
gen = GenerationService(load_gen_config())
gen.set_prompt(PromptConfig(text="{prompt}"))
res = gen.generate({"prompt": "Wie heißt du?"})
print(res)  # noqa: T201
