"""Tests for logging functionality"""

import json
import logging
import pytest
from io import StringIO
from unittest.mock import patch

from llm_api_router.logging_config import (
    LogConfig,
    SensitiveDataFilter,
    StructuredFormatter,
    TextFormatter,
    setup_logging,
    get_logger,
    generate_request_id,
)


class TestLogConfig:
    """Test LogConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == "text"
        assert config.enable_request_id is True
        assert config.filter_sensitive is True
        assert len(config.sensitive_patterns) > 0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = LogConfig(
            level="DEBUG",
            format="json",
            enable_request_id=False,
            filter_sensitive=False
        )
        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.enable_request_id is False
        assert config.filter_sensitive is False


class TestSensitiveDataFilter:
    """Test sensitive data filtering"""
    
    def test_filter_bearer_token(self):
        """Test filtering of Bearer tokens"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        text = "Authorization: Bearer sk-1234567890abcdef"
        filtered = filter.filter(text)
        assert "sk-1234567890abcdef" not in filtered
        assert "***MASKED***" in filtered
    
    def test_filter_api_key(self):
        """Test filtering of API keys"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        text = 'api_key="my-secret-key-12345"'
        filtered = filter.filter(text)
        assert "my-secret-key-12345" not in filtered
        assert "***MASKED***" in filtered
    
    def test_filter_openai_key(self):
        """Test filtering of OpenAI style keys"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        text = "Key: sk-proj-1234567890abcdefghijklmnop"
        filtered = filter.filter(text)
        assert "sk-proj-1234567890abcdefghijklmnop" not in filtered
        assert "***MASKED***" in filtered
    
    def test_filter_dict_authorization(self):
        """Test filtering authorization from dictionary"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        data = {
            "headers": {
                "Authorization": "Bearer secret-token",
                "Content-Type": "application/json"
            }
        }
        filtered = filter.filter_dict(data)
        assert filtered["headers"]["Authorization"] == "***MASKED***"
        assert filtered["headers"]["Content-Type"] == "application/json"
    
    def test_filter_dict_api_key(self):
        """Test filtering api_key from dictionary"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        data = {
            "config": {
                "api_key": "my-secret-key",
                "model": "gpt-4"
            }
        }
        filtered = filter.filter_dict(data)
        assert filtered["config"]["api_key"] == "***MASKED***"
        assert filtered["config"]["model"] == "gpt-4"
    
    def test_filter_dict_nested(self):
        """Test filtering nested dictionaries"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        data = {
            "level1": {
                "level2": {
                    "secret": "my-secret",
                    "token": "Bearer abc123"
                }
            }
        }
        filtered = filter.filter_dict(data)
        assert filtered["level1"]["level2"]["secret"] == "***MASKED***"
        assert filtered["level1"]["level2"]["token"] == "***MASKED***"
    
    def test_filter_dict_list(self):
        """Test filtering lists in dictionary"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        data = {
            "items": [
                {"api_key": "key1"},
                {"api_key": "key2"}
            ]
        }
        filtered = filter.filter_dict(data)
        assert filtered["items"][0]["api_key"] == "***MASKED***"
        assert filtered["items"][1]["api_key"] == "***MASKED***"
    
    def test_no_filter_safe_data(self):
        """Test that safe data is not filtered"""
        filter = SensitiveDataFilter(LogConfig().sensitive_patterns)
        text = "This is a safe message with no secrets"
        filtered = filter.filter(text)
        assert filtered == text


class TestStructuredFormatter:
    """Test JSON structured formatter"""
    
    def test_basic_format(self):
        """Test basic JSON formatting"""
        formatter = StructuredFormatter(filter_sensitive=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data
    
    def test_format_with_request_id(self):
        """Test formatting with request ID"""
        formatter = StructuredFormatter(filter_sensitive=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.request_id = "req-12345"
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["request_id"] == "req-12345"
    
    def test_format_with_provider(self):
        """Test formatting with provider"""
        formatter = StructuredFormatter(filter_sensitive=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.provider = "OpenAI"
        record.model = "gpt-4"
        record.latency_ms = 123
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["provider"] == "OpenAI"
        assert data["model"] == "gpt-4"
        assert data["latency_ms"] == 123
    
    def test_format_with_sensitive_filter(self):
        """Test formatting with sensitive data filtering"""
        formatter = StructuredFormatter(filter_sensitive=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Token: Bearer sk-1234567890",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert "sk-1234567890" not in data["message"]
        assert "***MASKED***" in data["message"]


class TestTextFormatter:
    """Test text formatter"""
    
    def test_basic_format(self):
        """Test basic text formatting"""
        formatter = TextFormatter(filter_sensitive=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        output = formatter.format(record)
        
        assert "test" in output
        assert "INFO" in output
        assert "Test message" in output
    
    def test_format_with_request_id(self):
        """Test formatting with request ID"""
        formatter = TextFormatter(filter_sensitive=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.request_id = "req-12345"
        
        output = formatter.format(record)
        assert "[req-12345]" in output
        assert "Test message" in output
    
    def test_format_with_sensitive_filter(self):
        """Test formatting with sensitive data filtering"""
        formatter = TextFormatter(filter_sensitive=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Authorization: Bearer sk-secret-123",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        assert "sk-secret-123" not in output
        assert "***MASKED***" in output


class TestLoggingSetup:
    """Test logging setup and configuration"""
    
    def test_setup_default(self):
        """Test default setup"""
        logger = setup_logging()
        assert logger.name == "llm_api_router"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_with_config(self):
        """Test setup with custom config"""
        config = LogConfig(level="DEBUG", format="json")
        logger = setup_logging(config)
        assert logger.level == logging.DEBUG
    
    def test_setup_json_format(self):
        """Test setup with JSON format"""
        # Clear any existing handlers first
        logger = logging.getLogger("llm_api_router")
        logger.handlers.clear()
        
        config = LogConfig(format="json")
        logger = setup_logging(config)
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)
    
    def test_setup_text_format(self):
        """Test setup with text format"""
        # Clear any existing handlers first
        logger = logging.getLogger("llm_api_router")
        logger.handlers.clear()
        
        config = LogConfig(format="text")
        logger = setup_logging(config)
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, TextFormatter)
    
    def test_get_logger(self):
        """Test get_logger function"""
        logger = get_logger("test")
        assert logger.name == "llm_api_router.test"
    
    def test_get_logger_default(self):
        """Test get_logger with default name"""
        logger = get_logger()
        assert logger.name == "llm_api_router"


class TestRequestID:
    """Test request ID generation"""
    
    def test_generate_request_id(self):
        """Test request ID generation"""
        request_id = generate_request_id()
        assert isinstance(request_id, str)
        assert len(request_id) > 0
    
    def test_request_id_unique(self):
        """Test that request IDs are unique"""
        id1 = generate_request_id()
        id2 = generate_request_id()
        assert id1 != id2


class TestLoggingIntegration:
    """Integration tests for logging"""
    
    def test_log_with_all_fields(self):
        """Test logging with all custom fields"""
        config = LogConfig(format="json", filter_sensitive=False)
        logger = setup_logging(config)
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(filter_sensitive=False))
        logger.handlers = [handler]
        
        # Log with custom fields
        logger.info(
            "Test message",
            extra={
                "request_id": "req-123",
                "provider": "OpenAI",
                "model": "gpt-4",
                "latency_ms": 100,
                "tokens": 50
            }
        )
        
        output = stream.getvalue()
        data = json.loads(output)
        
        assert data["message"] == "Test message"
        assert data["request_id"] == "req-123"
        assert data["provider"] == "OpenAI"
        assert data["model"] == "gpt-4"
        assert data["latency_ms"] == 100
        assert data["tokens"] == 50
    
    def test_log_filtering_in_practice(self):
        """Test that sensitive data is filtered in real logging"""
        config = LogConfig(format="text", filter_sensitive=True)
        logger = setup_logging(config)
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(TextFormatter(filter_sensitive=True))
        logger.handlers = [handler]
        
        # Log with sensitive data
        logger.info("API Key: sk-secret-key-12345")
        
        output = stream.getvalue()
        assert "sk-secret-key-12345" not in output
        assert "***MASKED***" in output
