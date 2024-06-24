from enum import Enum

from pydantic import BaseModel


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class LoggingConfig(BaseModel):
    level: LogLevel
