from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str
    port: int
    api_prefix: str
