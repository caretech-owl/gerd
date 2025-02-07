"""Logging configuration and utilities."""

from enum import Enum

from pydantic import BaseModel


class LogLevel(Enum):
    """Wrapper for string-based log levels.

    Translates log levels to integers for Python's logging framework.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

    def as_int(self) -> int:
        """Convert the log level to an integer."""
        return {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.FATAL: 50,
        }[self]


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: LogLevel
    """The log level."""
