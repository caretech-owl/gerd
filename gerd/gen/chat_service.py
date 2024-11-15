import logging
from typing import Dict, Literal

import gerd.backends.loader as gerd_loader
from gerd.models.gen import GenerationConfig
from gerd.models.model import ChatMessage, ModelConfig, PromptConfig
from gerd.transport import GenResponse

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class ChatService:
    def __init__(
        self, config: GenerationConfig, parameters: Dict[str, str] | None = None
    ) -> None:
        self._config = config
        self._model = gerd_loader.load_model_from_config(self._config.model)
        self.messages: list[ChatMessage] = []
        self.reset(parameters)

    def reset(self, parameters: Dict[str, str] | None = None) -> None:
        """Reset the chat history."""
        parameters = parameters or {}
        self.messages.clear()
        for role, message in self._config.model.prompt_setup:
            self.messages.append(
                {
                    "role": role,
                    "content": message.format(parameters),
                }
            )

    def set_prompt_config(
        self,
        config: PromptConfig,
    ) -> PromptConfig:
        """Set the prompt configuration."""
        self._config.model.prompt_config = config
        return self._config.model.prompt_config

    def get_prompt_config(self) -> PromptConfig:
        """Get the prompt configuration."""
        return self._config.model.prompt_config

    def add_message(
        self,
        parameters: Dict[str, str] | None = None,
        role: Literal["user", "system"] = "user",
    ) -> None:
        """Add a message to the chat history."""
        parameters = parameters or {}
        user_prompt: PromptConfig = self._config.model.prompt_config
        self.messages.append(
            {
                "role": role,
                "content": user_prompt.format(parameters),
            }
        )

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        self.reset(parameters)
        self.add_message(parameters, role="user")

        if self._config.features.prompt_chaining:
            from gerd.features.prompt_chaining import PromptChaining

            resolved = "".join(parameters.values())
            response = PromptChaining(
                self._config.features.prompt_chaining,
                self._model,
                self._config.model.prompt_config,
            ).generate(parameters)
        else:
            resolved = self._config.model.prompt_config.format(
                {"messages": self.messages}
            )
            _LOGGER.debug(
                "\n====== Resolved prompt =====\n\n%s\n\n=============================",
                resolved,
            )
            response = self._model.generate(resolved)
            _LOGGER.debug(
                "\n====== Response =====\n\n%s\n\n=============================",
                response,
            )
        return GenResponse(text=response, prompt=resolved)

    def submit_user_message(
        self, parameters: Dict[str, str] | None = None
    ) -> GenResponse:
        self.add_message(parameters, role="user")
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
        return GenResponse(text=response, prompt=self.messages[-2]["content"])
