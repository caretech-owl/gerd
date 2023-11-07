from enum import Enum

from pydantic import BaseModel


class ServerConfig(BaseModel):
    port: int
    ip: str
    enabled: bool
