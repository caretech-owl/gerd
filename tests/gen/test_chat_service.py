from threading import Thread
from time import sleep
import pytest
from pytest_mock import MockerFixture

from gerd.gen.chat_service import ChatService
from gerd.loader import MockLLM, RemoteLLM
from gerd.models.gen import GenerationConfig
from gerd.models.model import ChatMessage, ModelConfig, ModelEndpoint


@pytest.fixture
def chat_service_local(
    mocker: MockerFixture, generation_config: GenerationConfig
) -> ChatService:
    """A fixture that returns a GenerationService instance.

    Parameters:
        mocker: The mocker fixture
        generation_config: The generation configuration fixture

    Returns:
        A GenerationService instance
    """
    _ = mocker.patch(
        "gerd.loader.load_model_from_config",
        return_value=MockLLM(generation_config.model),
    )
    return ChatService(generation_config)


@pytest.fixture
def chat_service_remote(
    mocker: MockerFixture, generation_config: GenerationConfig
) -> ChatService:
    """A fixture that returns a GenerationService instance.

    Parameters:
        mocker: The mocker fixture
        generation_config: The generation configuration fixture

    Returns:
        A GenerationService instance
    """
    _ = mocker.patch(
        "gerd.loader.load_model_from_config",
        return_value=RemoteLLM(
            ModelConfig(
                name="not_used",
                endpoint=ModelEndpoint(
                    url="not_used",
                    type="llama.cpp",
                    key=None,
                ),
            )
        ),
    )
    return ChatService(generation_config)


def test_context_local(
    chat_service_local: ChatService, generation_config: GenerationConfig
) -> None:
    """Test the context manager of the GenerationService class.

    Parameters:
        gen_service: The GenerationService fixture
        generation_config: The generation configuration fixture
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
        assert chat.config == generation_config
        assert chat.messages == my_messages
        chat.messages = [ChatMessage(role="system", content="bar")]
        assert chat.messages != my_messages
        assert chat._enter_lock is not None  # noqa: SLF001
        assert id(chat) == id(chat_service_local)
        thread = Thread(target=acquire_lock, args=(chat,))
        thread.start()
        thread.join(timeout=0.1)
        tested = True
    assert chat.messages == my_messages

def test_context_remote(
    chat_service_remote: ChatService,
    generation_config: GenerationConfig,
) -> None:
    """Test the context manager of the GenerationService class.

    Parameters:
        gen_service: The GenerationService fixture
        generation_config: The generation configuration fixture
    """
    with chat_service_remote as chat:
        assert chat.config == generation_config
        chat.messages = [ChatMessage(role="user", content="Hello")]
        assert chat._enter_lock is None  # noqa: SLF001
        assert id(chat) != id(chat_service_remote)
        with chat as chat2:
            assert id(chat) != id(chat2)
        assert chat_service_remote.messages != chat.messages
