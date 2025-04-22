"""Unit tests for the chat service."""

import json
from threading import Thread
from time import sleep

import pytest
from pytest_mock import MockerFixture

from gerd.gen.chat_service import ChatService
from gerd.loader import MockLLM, RemoteLLM
from gerd.models.gen import GenerationConfig
from gerd.models.model import ChatMessage, ModelConfig, ModelEndpoint, PromptConfig


@pytest.fixture
def chat_service_local(
    mocker: MockerFixture, generation_config: GenerationConfig
) -> ChatService:
    """A fixture that returns a ChatService instance.

    Parameters:
        mocker: The mocker fixture
        generation_config: The generation configuration fixture

    Returns:
        A ChatService instance
    """
    _ = mocker.patch(
        "gerd.loader.load_model_from_config",
        return_value=MockLLM(generation_config.model),
    )
    return ChatService(generation_config)


@pytest.fixture
def chat_service_remote(
    generation_config: GenerationConfig
) -> ChatService:
    """A fixture that returns a ChatService instance.

    Parameters:
        mocker: The mocker fixture
        generation_config: The generation configuration fixture

    Returns:
        A ChatService instance
    """
    generation_config.model.name = "not_used"
    generation_config.model.endpoint = ModelEndpoint(
        url="not_used",
        type="openai",
    )
    return ChatService(generation_config)


def test_context_local(chat_service_local: ChatService) -> None:
    """Test the context manager of the ChatService class for local LLM.

    Parameters:
        chat_service_local: The ChatService fixture
    """
    tested = False
    my_messages = [ChatMessage(role="user", content="Hello")]
    chat_service_local.messages = my_messages

    def acquire_lock(chat: ChatService) -> None:
        nonlocal tested
        with chat as chat2:
            if not tested:
                msg = "Lock has been acquired"
                raise RuntimeError(msg)

    with chat_service_local as chat:
        assert chat.messages == my_messages
        chat.messages.clear()
        assert chat._enter_lock is not None  # noqa: SLF001
        assert id(chat) == id(chat_service_local)
        thread = Thread(target=acquire_lock, args=(chat,))
        thread.start()
        thread.join(timeout=0.1)
        tested = True
    assert len(chat_service_local.messages) > 0


def test_context_remote(chat_service_remote: ChatService) -> None:
    """Test the context manager of the ChatService class for remote LLM.

    Parameters:
        chat_service_remote: The ChatService fixture with RemoteLLM
    """
    with chat_service_remote as chat:
        chat.messages = [ChatMessage(role="user", content="Hello")]
        assert chat._enter_lock is None  # noqa: SLF001
        assert id(chat) != id(chat_service_remote)
        assert id(chat.messages) != id(chat_service_remote.messages)
        with chat as chat2:
            assert id(chat) != id(chat2)
        assert chat_service_remote.messages != chat.messages


def test_context_remote_template(chat_service_remote: ChatService) -> None:
    """Test the context manager with a template config and remote LLM.

    Parameters:
        chat_service_remote: The ChatService fixture with RemoteLLM
    """
    chat_service_remote.set_prompt_config(PromptConfig(is_template=True, text="Hello!"))
    with chat_service_remote as chat:
        chat.messages = [ChatMessage(role="user", content="Hello")]
        assert chat._enter_lock is None  # noqa: SLF001
        assert id(chat) != id(chat_service_remote)
        assert id(chat.messages) != id(chat_service_remote.messages)
        with chat as chat2:
            assert id(chat) != id(chat2)
        assert chat_service_remote.messages != chat.messages


def test_context_local_template(chat_service_local: ChatService) -> None:
    """Test the context manager with a template config and local LLM.

    Parameters:
        chat_service_local: The ChatService fixture with local LLM
    """
    chat_service_local.set_prompt_config(PromptConfig(is_template=True, text="Hello!"))
    with chat_service_local as chat:
        assert chat.get_prompt_config().is_template
        assert chat._enter_lock is not None  # noqa: SLF001


def test_model_edit(mocker: MockerFixture, chat_service_remote: ChatService) -> None:
    """Test wether model edits are considered.

    Parameters:
        chat_service_remote: The ChatService fixture
    """
    from requests.models import Response

    chat_service_remote.set_prompt_config(
        PromptConfig(is_template=True, text="{message}")
    )
    res = Response()
    res.status_code = 200
    res._content = b'{"choices":[{"message":{"role":"assistant","content":"Hello!"}}]}'  # noqa: SLF001
    chat_completion = mocker.patch(
        "requests.post",
        return_value=res
    )
    test_model = "test_model"
    chat_service_remote.config.model.name = test_model
    chat_service_remote.submit_user_message({"message": "Hello"})
    parsed = json.loads(chat_completion.call_args_list[-1][1]["data"])
    assert parsed["model"] == test_model
    with chat_service_remote as chat:
        another_model = "another_model"
        chat.config.model.name = another_model
        chat.submit_user_message({"message": "Hello"})
        parsed = json.loads(chat_completion.call_args_list[-1][1]["data"])
        assert parsed["model"] == another_model
