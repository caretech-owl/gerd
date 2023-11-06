from enum import Enum

from pydantic import BaseModel


class BackendMode(Enum):
    DIRECT = "direct"
    REST = "rest"


class BackendConfig(BaseModel):
    port: int
    ip: str
    mode: BackendMode
