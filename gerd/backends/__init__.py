"""This module contains backend implementations that manage services.

These backends can be used by frontends such as gradio.
Furthermore, the backend module contains service implementations for loading LLMs or
vector stores for Retrieval Augmented Generation.
"""

from gerd.backends.bridge import Bridge
from gerd.transport import Transport

TRANSPORTER: Transport = Bridge()
"""The default transporter that connects the backend services to the frontend."""
