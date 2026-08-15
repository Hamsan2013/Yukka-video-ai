"""
Test logging utilities.
"""

import pytest
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yukkavideo.utils.logging_config import (
    setup_logging,
    get_logger,
    set_level,
    LogLevel,
    ContextLogger,
)


class TestLoggingSetup:
    """Tests for logging setup functions."""

    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a logger."""
        logger = setup_logging(name="test_logger_1")
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger."""
        logger = get_logger("test_logger_2")
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_with_custom_level(self):
        """Test setup_logging with custom log level."""
        logger = setup_logging(level=logging.DEBUG, name="test_logger_3")
        
        assert logger.level == logging.DEBUG

    def test_set_level_changes_logger_level(self):
        """Test that set_level changes logger level."""
        logger = setup_logging(level=logging.INFO, name="test_logger_4")
        set_level(logging.WARNING, name="test_logger_4")
        
        assert logger.level == logging.WARNING


class TestLogLevel:
    """Tests for LogLevel class."""

    def test_log_level_constants(self):
        """Test that LogLevel has correct constants."""
        assert LogLevel.DEBUG == logging.DEBUG
        assert LogLevel.INFO == logging.INFO
        assert LogLevel.WARNING == logging.WARNING
        assert LogLevel.ERROR == logging.ERROR
        assert LogLevel.CRITICAL == logging.CRITICAL


class TestContextLogger:
    """Tests for ContextLogger context manager."""

    def test_context_logger_temporary_level(self):
        """Test that ContextLogger temporarily changes level."""
        logger = setup_logging(level=logging.INFO, name="test_logger_5")
        original_level = logger.level
        
        with ContextLogger("test_logger_5", logging.DEBUG):
            assert logger.level == logging.DEBUG
        
        # Level should be restored after context
        assert logger.level == original_level

    def test_context_logger_returns_logger(self):
        """Test that ContextLogger returns logger on enter."""
        with ContextLogger("test_logger_6", logging.DEBUG) as logger:
            assert logger is not None
            assert isinstance(logger, logging.Logger)
