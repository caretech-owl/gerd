import logging
from typing import Dict

import gerd.backends.loader as gerd_loader
from gerd.models.model import ChatMessage, ModelConfig, PromptConfig
from gerd.transport import GenResponse

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class ChatService:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = gerd_loader.load_model_from_config(self._config)
        self.messages: list[ChatMessage] = []
        self.reset()

    def reset(self) -> None:
        """Reset the chat history."""
        self.messages.clear()
        if self._config.prompt.text:
            self.messages.append(
                {"role": "system", "content": self._config.prompt.text}
            )

    def set_prompt(self, config: PromptConfig) -> PromptConfig:
        """Set the prompt configuration."""
        self._config.prompt = config
        return self._config.prompt

    def get_prompt(self) -> PromptConfig:
        """Get the prompt configuration."""
        return self._config.prompt

    def submit_user_message(
        self,
        message: str,
        parameters: Dict[str, str] | None = None,
        add_prompt: bool = False,
    ) -> GenResponse:
        template = self._config.prompt.template
        parameters = parameters or {}
        resolved = (
            template.render(**parameters)
            if template
            else self._config.prompt.text.format(**parameters)
        )
        self.messages.append({"role": "user", "content": message})
        _LOGGER.debug(
            "\n====== Resolved prompt =====\n\n%s\n\n=============================",
            "\n".join(m["role"] + ": " + str(m["content"]) for m in self.messages),
        )
        role, response = self._model.create_chat_completion(self.messages)
        _LOGGER.debug(
            "\n====== Response =====\n\n%s: %s\n\n=============================",
            role,
            response,
        )
        self.messages.append({"role": role, "content": response})
        return GenResponse(text=response, prompt=resolved if add_prompt else None)
