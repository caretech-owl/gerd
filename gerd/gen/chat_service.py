"""Implementation of the ChatService class.

This features the currently favoured approach of instruction-based work with
large language models. Thus, models fined tuned for chat or instructions work
best with this service. The service can be used to generate text as well as long
as the model features a chat template.
In this case this service should be prefered over the
[GenerationService][gerd.gen.generation_service.GenerationService] since it is
easier to setup a prompt according to the model's requirements.
"""

import logging
from copy import deepcopy
from threading import Lock
from types import TracebackType
from typing import Dict, Literal, Optional, Type

import gerd.loader as gerd_loader
from gerd.models.gen import GenerationConfig
from gerd.models.model import ChatMessage, PromptConfig
from gerd.transport import GenResponse

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class ChatService:
    """Service to generate text based on a chat history."""

    def __init__(
        self, config: GenerationConfig, parameters: Dict[str, str] | None = None
    ) -> None:
        """The service is initialized with a config and parameters.

        The parameters are used to initialize the message history.
        However, future reset will not consider them.
        loads a model according to this config.
        The used LLM is loaded according to the model configuration
        right on initialization.
        """
        self.config = self._enter_config = config
        self.messages: list[ChatMessage] = []
        self._enter_messages = self.messages
        self._model = gerd_loader.load_model_from_config(self.config.model)
        self._enter_lock: Lock | None = (
            Lock() if not isinstance(self._model, gerd_loader.RemoteLLM) else None
        )
        # Lock is only needed for local models

        self.reset(parameters)

    def __enter__(self) -> "ChatService":
        """Enter the runtime context related to this object.

        When a remote model is used, a deepcopy of the service is returned.
        For local models, the service itself is returned but config and message history
        restored when the context is exited.
        """
        _LOGGER.debug("Entering ChatService context.")
        if self._enter_lock:
            self._enter_lock.acquire()
            self._enter_config = deepcopy(self.config)
            self._enter_messages = deepcopy(self.messages)
            service = self
        else:
            service = deepcopy(self)
        return service

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Exit the runtime context related to this object."""
        _LOGGER.debug("Exiting ChatService context.")
        if self._enter_lock:
            self.config = self._enter_config
            self.messages = self._enter_messages
            self._enter_lock.release()

    def reset(self, parameters: Dict[str, str] | None = None) -> None:
        """Reset the chat history."""
        parameters = parameters or {}
        self.messages.clear()
        for role, message in self.config.model.prompt_setup:
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
        self.config.model.prompt_config = config
        return self.config.model.prompt_config

    def get_prompt_config(self) -> PromptConfig:
        """Get the prompt configuration."""
        return self.config.model.prompt_config

    def add_message(
        self,
        parameters: Dict[str, str] | None = None,
        role: Literal["user", "system", "assistant"] = "user",
        prompt_config: Optional[PromptConfig] = None,
    ) -> None:
        """Add a message to the chat history."""
        parameters = parameters or {}
        user_prompt: PromptConfig = prompt_config or self.config.model.prompt_config
        self.messages.append(
            {
                "role": role,
                "content": user_prompt.format(parameters),
            }
        )

    def generate(self, parameters: Dict[str, str]) -> GenResponse:
        """Generate a response based on the chat history.

        This method can be used as a replacement for
        [GenerationService.generate][gerd.gen.generation_service.GenerationService.generate]
        in cases where the used model provides a chat template.
        When this is the case, using this method is more reliable as it requires less
        manual configuration to set up the prompt according to the model's requirements.

        Parameters:
            parameters: The parameters to format the prompt with

        Returns:
            The generation result
        """
        self.reset(parameters)
        self.add_message(parameters, role="user")

        if self.config.features.prompt_chaining:
            for i, prompt_config in enumerate(
                self.config.features.prompt_chaining.prompts, 1
            ):
                self.reset()
                res = self.submit_user_message(parameters, prompt_config=prompt_config)
                parameters[f"response_{i}"] = res.text
            response = res.text
            resolved = "\n".join(parameters.values())
        else:
            resolved = "\n".join(m["content"] for m in self.messages)
            _LOGGER.debug(
                "\n===== Resolved prompt ======\n\n%s\n\n============================",
                resolved,
            )
            response = self._model.generate(resolved, self.config.model)
            _LOGGER.debug(
                "\n========= Response =========\n\n%s\n\n============================",
                response,
            )
        return GenResponse(text=response, prompt=resolved)

    def submit_user_message(
        self,
        parameters: Dict[str, str] | None = None,
        prompt_config: Optional[PromptConfig] = None,
    ) -> GenResponse:
        """Submit a message with the user role and generates a response.

        The service's prompt configuration is used to format the prompt unless
        a different prompt configuration is provided.
        Parameters:
            parameters: The parameters to format the prompt with
            prompt_config: The optional prompt configuration to be used

        Returns:
            The generation result
        """
        self.add_message(parameters, role="user", prompt_config=prompt_config)
        _LOGGER.debug(
            "\n===== Resolved prompt ======\n\n%s\n\n============================",
            "\n".join(m["role"] + ": " + str(m["content"]) for m in self.messages),
        )
        role, response = self._model.create_chat_completion(
            self.messages, self.config.model
        )
        _LOGGER.debug(
            "\n========= Response =========\n\n%s: %s\n\n============================",
            role,
            response,
        )
        self.messages.append({"role": role, "content": response})
        if len(self.messages) < 2:
            msg = "Not enough messages in history to access the second-to-last message."
            raise IndexError(msg)
        return GenResponse(text=response, prompt=self.messages[-2]["content"])
