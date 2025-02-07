"""Server configuration model for REST backends."""

from pydantic import BaseModel


class ServerConfig(BaseModel):
    """Server configuration model for REST backends."""

    host: str
    """The host of the server."""
    port: int
    """The port of the server."""
    api_prefix: str
    """The prefix of the API."""
