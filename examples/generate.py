import logging

from gerd.config import CONFIG
from gerd.gen import GenerationService
from gerd.models.model import ModelEndpoint, PromptConfig

logging.basicConfig(level=logging.DEBUG)

# For remoter server example
# CONFIG.gen.model.endpoint = ModelEndpoint(url="http://localhost:8080", type="")
gen = GenerationService(CONFIG.gen)
gen.set_prompt(PromptConfig(text="{prompt}"))
res = gen.generate({"prompt": "Wie heißt du?"})
print(res)  # noqa: T201
