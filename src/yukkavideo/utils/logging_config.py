"""
Logging Configuration for Yukka Video AI

Provides centralized logging setup for the entire application.
"""

import logging
import sys
from typing import Optional, Dict
from pathlib import Path


# Global logger registry
_loggers: Dict[str, logging.Logger] = {}

_default_format = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    name: str = "yukkavideo",
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to log file.
        format_string: Custom format string for log messages.
        name: Name for the logger.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging(level=logging.DEBUG)
        >>> logger.info("Application started")
    """
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Format string
    fmt = format_string or _default_format
    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "yukkavideo") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger("yukkavideo.inference")
        >>> logger.debug("Debug message")
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger


def set_level(level: int, name: str = "yukkavideo") -> None:
    """
    Set logging level for a specific logger.

    Args:
        level: Logging level.
        name: Name of the logger.
    """
    if name in _loggers:
        _loggers[name].setLevel(level)
        for handler in _loggers[name].handlers:
            handler.setLevel(level)


class LogLevel:
    """Convenience class for logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ContextLogger:
    """
    Context manager for temporary logging level changes.

    Example:
        >>> with ContextLogger("yukkavideo", logging.DEBUG):
        ...     # Code that logs at DEBUG level
        ...     pass
    """

    def __init__(self, name: str, temp_level: int):
        """
        Initialize the context logger.

        Args:
            name: Name of the logger.
            temp_level: Temporary logging level.
        """
        self.name = name
        self.temp_level = temp_level
        self.original_level: Optional[int] = None
        self.logger: Optional[logging.Logger] = None

    def __enter__(self) -> logging.Logger:
        """Enter the context and set temporary level."""
        self.logger = get_logger(self.name)
        self.original_level = self.logger.level
        self.logger.setLevel(self.temp_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore original level."""
        if self.logger and self.original_level is not None:
            self.logger.setLevel(self.original_level)
