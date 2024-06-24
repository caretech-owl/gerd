from enum import Enum

from pydantic import BaseModel


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

    def as_int(self) -> int:
        return {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.FATAL: 50,
        }[self]


class LoggingConfig(BaseModel):
    level: LogLevel
