"""
Example: Logging Configuration and Usage

This example demonstrates how to use the logging system in llm-api-router.
"""

import os
from llm_api_router import (
    Client,
    ProviderConfig,
    LogConfig,
    setup_logging,
)

# Example 1: Default logging (text format, INFO level)
print("=" * 60)
print("Example 1: Default Logging")
print("=" * 60)

config = ProviderConfig(
    provider_type="openai",
    api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-demo"),
    default_model="gpt-3.5-turbo"
)

# Note: With default config, logging is initialized automatically
# You can also explicitly set up logging:
logger = setup_logging()
logger.info("Application started with default logging")

print("\n")

# Example 2: JSON structured logging
print("=" * 60)
print("Example 2: JSON Structured Logging")
print("=" * 60)

log_config = LogConfig(
    level="INFO",
    format="json",  # Use JSON format for structured logs
    enable_request_id=True,
    filter_sensitive=True,
)

config_with_json_logging = ProviderConfig(
    provider_type="openai",
    api_key="sk-test-key-12345",  # This will be masked in logs
    default_model="gpt-3.5-turbo",
    log_config=log_config
)

# Setup logging with JSON format
json_logger = setup_logging(log_config)
json_logger.info("Using JSON structured logging")
json_logger.info("API key will be masked: sk-test-key-12345")

print("\n")

# Example 3: Debug level logging
print("=" * 60)
print("Example 3: Debug Level Logging")
print("=" * 60)

debug_config = LogConfig(
    level="DEBUG",  # Set to DEBUG for detailed logs
    format="text",
    enable_request_id=True,
)

debug_logger = setup_logging(debug_config)
debug_logger.debug("This is a debug message")
debug_logger.info("This is an info message")
debug_logger.warning("This is a warning message")
debug_logger.error("This is an error message")

print("\n")

# Example 4: Logging with custom fields (request ID, provider, etc.)
print("=" * 60)
print("Example 4: Logging with Custom Fields")
print("=" * 60)

logger = setup_logging(LogConfig(format="json"))

# Log with additional context
logger.info(
    "Processing request",
    extra={
        "request_id": "req-abc123",
        "provider": "OpenAI",
        "model": "gpt-4",
        "latency_ms": 150,
        "tokens": 100,
    }
)

print("\n")

# Example 5: Sensitive data filtering
print("=" * 60)
print("Example 5: Sensitive Data Filtering")
print("=" * 60)

secure_config = LogConfig(
    level="INFO",
    format="text",
    filter_sensitive=True,  # Enable sensitive data filtering
)

secure_logger = setup_logging(secure_config)

# These will be automatically masked
secure_logger.info("Authorization: Bearer sk-proj-1234567890abcdef")
secure_logger.info('Configuration: {"api_key": "my-secret-key"}')
secure_logger.info("Using API key: sk-test-another-key")

print("\n")

# Example 6: Disabling sensitive data filtering (use with caution!)
print("=" * 60)
print("Example 6: Logging Without Filtering (Debug Mode)")
print("=" * 60)

debug_no_filter_config = LogConfig(
    level="DEBUG",
    format="text",
    filter_sensitive=False,  # Disable filtering for debugging
)

unfiltered_logger = setup_logging(debug_no_filter_config)
unfiltered_logger.warning("⚠️  Sensitive data filtering is DISABLED")
unfiltered_logger.debug("API key visible: sk-debug-key-12345")
unfiltered_logger.warning("⚠️  Never use this in production!")

print("\n")

# Example 7: Simulating actual API request logging
print("=" * 60)
print("Example 7: Simulated API Request Logging")
print("=" * 60)

from llm_api_router.logging_config import generate_request_id

api_logger = setup_logging(LogConfig(format="json", level="INFO"))

request_id = generate_request_id()

# Log request
api_logger.info(
    "Sending chat completion request",
    extra={
        "request_id": request_id,
        "provider": "OpenAI",
        "model": "gpt-3.5-turbo",
        "message_count": 2,
    }
)

# Simulate successful response
api_logger.info(
    "Chat completion successful",
    extra={
        "request_id": request_id,
        "provider": "OpenAI",
        "latency_ms": 245,
        "tokens": 156,
    }
)

print("\n")
print("=" * 60)
print("Logging examples completed!")
print("=" * 60)
print("\nKey Features Demonstrated:")
print("1. Default text-based logging")
print("2. JSON structured logging for machine parsing")
print("3. Configurable log levels (DEBUG, INFO, WARNING, ERROR)")
print("4. Custom fields (request_id, provider, model, latency, tokens)")
print("5. Automatic sensitive data filtering (API keys, tokens)")
print("6. Request ID tracking for correlation")
print("7. Real-world API request/response logging patterns")
