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
        system_prompt: PromptConfig = self._config.model.prompt.get(
            "system", PromptConfig.model_validate({"text": "{message}"})
        )
        self.messages.append(
            {
                "role": "system",
                "content": system_prompt.format(parameters),
            }
        )

    def set_prompt(
        self, config: PromptConfig, field: Literal["format", "user", "system"]
    ) -> PromptConfig:
        """Set the prompt configuration."""
        self._config.model.prompt[field] = config
        return self._config.model.prompt[field]

    def get_prompt(self, field: Literal["format", "user", "system"]) -> PromptConfig:
        """Get the prompt configuration."""
        return self._config.model.prompt[field]

    def add_message(
        self,
        parameters: Dict[str, str] | None = None,
        role: Literal["user", "system", "assistant"] = "user",
    ) -> None:
        """Add a message to the chat history."""
        parameters = parameters or {}
        user_prompt: PromptConfig = self._config.model.prompt.get(
            role, PromptConfig.model_validate({"text": "{message}"})
        )
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

            response = PromptChaining(
                self._config.features.prompt_chaining,
                self._model,
                self._config.model.prompt.get("format", PromptConfig("{message}")),
            ).generate(parameters)
        else:
            if "format" in self._config.model.prompt:
                resolved = self._config.model.prompt["format"].format(
                    {"messages": self.messages}
                )
            else:
                resolved = "".join([message["content"] for message in self.messages])
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
