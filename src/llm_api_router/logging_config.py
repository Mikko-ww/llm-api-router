"""Logging configuration and utilities for LLM API Router"""

import logging
import json
import re
import uuid
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LogConfig:
    """Configuration for logging system"""
    
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "text"  # "text" or "json"
    enable_request_id: bool = True
    filter_sensitive: bool = True
    log_requests: bool = True
    log_responses: bool = True
    log_errors: bool = True
    
    # Sensitive patterns to filter
    sensitive_patterns: list = field(default_factory=lambda: [
        r"Bearer\s+\S+",  # Bearer tokens
        r"api[_-]?key['\"]?\s*[:=]\s*['\"]?[\w-]+",  # API keys
        r"sk-[\w-]+",  # OpenAI style keys
    ])


class SensitiveDataFilter:
    """Filter sensitive information from log messages"""
    
    def __init__(self, patterns: list):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def filter(self, text: str) -> str:
        """Replace sensitive data with masked placeholder"""
        if not text:
            return text
        
        filtered = text
        for pattern in self.patterns:
            filtered = pattern.sub("***MASKED***", filtered)
        return filtered
    
    def filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from dictionary recursively"""
        if not isinstance(data, dict):
            return data
        
        filtered = {}
        for key, value in data.items():
            # Filter known sensitive keys
            if key.lower() in ["authorization", "api_key", "api-key", "apikey", "token", "secret"]:
                filtered[key] = "***MASKED***"
            elif isinstance(value, str):
                filtered[key] = self.filter(value)
            elif isinstance(value, dict):
                filtered[key] = self.filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [self.filter_dict(item) if isinstance(item, dict) else item for item in value]
            else:
                filtered[key] = value
        
        return filtered


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, filter_sensitive: bool = True, sensitive_patterns: Optional[list] = None):
        super().__init__()
        self.filter_sensitive = filter_sensitive
        self.sensitive_filter = SensitiveDataFilter(
            sensitive_patterns or LogConfig().sensitive_patterns
        ) if filter_sensitive else None
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add provider if available
        if hasattr(record, "provider"):
            log_data["provider"] = record.provider
        
        # Add extra fields
        for key in ["model", "latency_ms", "status_code", "tokens", "attempt"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Filter sensitive data
        if self.filter_sensitive and self.sensitive_filter:
            log_data = self.sensitive_filter.filter_dict(log_data)
        
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Text formatter with sensitive data filtering"""
    
    def __init__(self, filter_sensitive: bool = True, sensitive_patterns: Optional[list] = None):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.filter_sensitive = filter_sensitive
        self.sensitive_filter = SensitiveDataFilter(
            sensitive_patterns or LogConfig().sensitive_patterns
        ) if filter_sensitive else None
    
    def format(self, record: logging.LogRecord) -> str:
        # Add request ID to message if available
        if hasattr(record, "request_id"):
            original_msg = record.msg
            record.msg = f"[{record.request_id}] {record.msg}"
            formatted = super().format(record)
            record.msg = original_msg
        else:
            formatted = super().format(record)
        
        # Filter sensitive data
        if self.filter_sensitive and self.sensitive_filter:
            formatted = self.sensitive_filter.filter(formatted)
        
        return formatted


def setup_logging(config: Optional[LogConfig] = None) -> logging.Logger:
    """
    Setup and configure logging for LLM API Router
    
    Args:
        config: Logging configuration, uses defaults if not provided
        
    Returns:
        Configured logger instance
    """
    config = config or LogConfig()
    
    # Get or create logger
    logger = logging.getLogger("llm_api_router")
    
    # Set level
    level = getattr(logging, config.level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Set formatter based on config
    if config.format == "json":
        formatter = StructuredFormatter(
            filter_sensitive=config.filter_sensitive,
            sensitive_patterns=config.sensitive_patterns
        )
    else:
        formatter = TextFormatter(
            filter_sensitive=config.filter_sensitive,
            sensitive_patterns=config.sensitive_patterns
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name, defaults to "llm_api_router"
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"llm_api_router.{name}")
    return logging.getLogger("llm_api_router")


def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())


# Initialize default logger
_default_logger = setup_logging()
