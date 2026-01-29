# Logging System Documentation

The LLM API Router includes a comprehensive logging system that provides structured logging, sensitive data filtering, request tracking, and flexible configuration.

## Features

- **Structured Logging**: Support for both text and JSON formats
- **Sensitive Data Filtering**: Automatically masks API keys, tokens, and other sensitive information
- **Request ID Tracking**: Unique IDs for correlating logs across operations
- **Configurable Levels**: Support for DEBUG, INFO, WARNING, ERROR, and CRITICAL levels
- **Rich Context**: Log provider names, models, latency, token usage, and more
- **Retry Tracking**: Automatic logging of retry attempts with exponential backoff

## Quick Start

### Basic Usage

```python
from llm_api_router import Client, ProviderConfig

# Logging is enabled by default with INFO level
config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="gpt-3.5-turbo"
)

client = Client(config)
# Logs will automatically be generated for requests, responses, and errors
```

### Custom Logging Configuration

```python
from llm_api_router import ProviderConfig, LogConfig

# Configure logging
log_config = LogConfig(
    level="DEBUG",           # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="json",          # Format: "text" or "json"
    enable_request_id=True, # Enable unique request ID tracking
    filter_sensitive=True,  # Filter sensitive data (API keys, tokens)
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="gpt-3.5-turbo",
    log_config=log_config
)

client = Client(config)
```

## Configuration Options

### LogConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | str | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `format` | str | "text" | Output format ("text" or "json") |
| `enable_request_id` | bool | True | Generate unique request IDs |
| `filter_sensitive` | bool | True | Filter sensitive data from logs |
| `log_requests` | bool | True | Log API requests |
| `log_responses` | bool | True | Log API responses |
| `log_errors` | bool | True | Log errors |
| `sensitive_patterns` | list | [default patterns] | Regex patterns for sensitive data |

### Default Sensitive Patterns

The following patterns are filtered by default:

- Bearer tokens: `Bearer \S+`
- API keys: `api[_-]?key['\"]?\s*[:=]\s*['\"]?[\w-]+`
- OpenAI-style keys: `sk-[\w-]+`
- Common fields: `authorization`, `api_key`, `token`, `secret`

## Log Formats

### Text Format

Human-readable format with timestamps:

```
2026-01-29 16:46:05 - llm_api_router.OpenAIProvider - INFO - [req-123] Sending chat completion request
2026-01-29 16:46:05 - llm_api_router.OpenAIProvider - INFO - [req-123] Chat completion successful
```

### JSON Format

Structured format for machine parsing:

```json
{
  "timestamp": "2026-01-29T16:46:05.468899+00:00",
  "level": "INFO",
  "logger": "llm_api_router.OpenAIProvider",
  "message": "Sending chat completion request",
  "request_id": "req-123",
  "provider": "OpenAI",
  "model": "gpt-3.5-turbo",
  "message_count": 2
}

{
  "timestamp": "2026-01-29T16:46:05.469804+00:00",
  "level": "INFO",
  "logger": "llm_api_router.OpenAIProvider",
  "message": "Chat completion successful",
  "request_id": "req-123",
  "provider": "OpenAI",
  "latency_ms": 245,
  "tokens": 156
}
```

## Logged Information

### Request Logs

For each API request, the following information is logged:

- Request ID (unique identifier)
- Provider name (e.g., "OpenAI", "Anthropic")
- Model name
- Number of messages
- Stream mode indicator

Example:
```python
logger.info(
    "Sending chat completion request",
    extra={
        "request_id": "req-abc123",
        "provider": "OpenAI",
        "model": "gpt-3.5-turbo",
        "message_count": 2
    }
)
```

### Response Logs

For successful responses, the following is logged:

- Request ID (for correlation)
- Provider name
- Latency in milliseconds
- Token usage (prompt, completion, total)

Example:
```python
logger.info(
    "Chat completion successful",
    extra={
        "request_id": "req-abc123",
        "provider": "OpenAI",
        "latency_ms": 245,
        "tokens": 156
    }
)
```

### Error Logs

For errors, comprehensive information is logged:

- Request ID
- Provider name
- HTTP status code
- Error message
- Exception traceback (for DEBUG level)

Example:
```python
logger.error(
    "API error: Rate limit exceeded",
    extra={
        "request_id": "req-abc123",
        "provider": "OpenAI",
        "status_code": 429
    }
)
```

### Retry Logs

When retries occur, the following is logged:

- Retry attempt number
- Delay before retry
- Error type
- Request ID

Example:
```python
logger.warning(
    "Retry attempt 1/3 after 1.0s: RateLimitError",
    extra={
        "attempt": 1,
        "delay": 1.0,
        "request_id": "req-abc123",
        "status_code": 429
    }
)
```

## Advanced Usage

### Manual Logger Setup

```python
from llm_api_router import setup_logging, LogConfig

# Setup logger with custom configuration
logger = setup_logging(LogConfig(
    level="DEBUG",
    format="json"
))

logger.info("Application started")
```

### Get Named Logger

```python
from llm_api_router import get_logger

# Get a named logger for a specific module
logger = get_logger("my_module")
logger.info("Custom module log")
```

### Request ID Generation

```python
from llm_api_router.logging_config import generate_request_id

# Generate unique request IDs for tracking
request_id = generate_request_id()
logger.info("Processing request", extra={"request_id": request_id})
```

### Custom Sensitive Patterns

```python
log_config = LogConfig(
    filter_sensitive=True,
    sensitive_patterns=[
        r"password['\"]?\s*[:=]\s*['\"]?[\w-]+",  # Passwords
        r"token['\"]?\s*[:=]\s*['\"]?[\w-]+",     # Tokens
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
    ]
)
```

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: Normal operational messages (requests, responses)
- **WARNING**: Unexpected situations (retries, degraded performance)
- **ERROR**: Errors that need attention (API failures)
- **CRITICAL**: Critical failures (system-wide issues)

### 2. Enable Sensitive Data Filtering in Production

Always enable sensitive data filtering in production:

```python
log_config = LogConfig(filter_sensitive=True)
```

### 3. Use JSON Format for Log Aggregation

For centralized logging systems (ELK, Splunk, CloudWatch), use JSON format:

```python
log_config = LogConfig(format="json")
```

### 4. Enable Request ID Tracking

Request IDs help correlate logs across multiple operations:

```python
log_config = LogConfig(enable_request_id=True)
```

### 5. Set Appropriate Log Levels

- Development: `DEBUG` or `INFO`
- Staging: `INFO`
- Production: `INFO` or `WARNING`

### 6. Monitor Log Volume

Be mindful of log volume in production. Consider:
- Setting level to `WARNING` for high-traffic applications
- Using structured logging for efficient parsing
- Setting up log rotation and retention policies

## Integration with Log Aggregation Systems

### Elasticsearch/Logstash/Kibana (ELK)

```python
# Use JSON format for easy ingestion
log_config = LogConfig(
    format="json",
    level="INFO"
)
```

### AWS CloudWatch

```python
# JSON format works well with CloudWatch Insights
log_config = LogConfig(
    format="json",
    level="INFO"
)
```

### Splunk

```python
# Structured logs for Splunk
log_config = LogConfig(
    format="json",
    level="INFO"
)
```

## Troubleshooting

### Logs Not Appearing

1. Check log level configuration
2. Ensure logging is initialized before making requests
3. Verify that handlers are attached to the logger

### Sensitive Data Not Filtered

1. Verify `filter_sensitive=True` in LogConfig
2. Check if your sensitive pattern is included in default patterns
3. Add custom patterns if needed

### Duplicate Logs

This can happen if logging is initialized multiple times. Use the setup function once:

```python
# Do this once at application startup
logger = setup_logging(log_config)
```

## Examples

See `examples/logging_example.py` for a comprehensive demonstration of all logging features.

## API Reference

### LogConfig

```python
@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "text"
    enable_request_id: bool = True
    filter_sensitive: bool = True
    log_requests: bool = True
    log_responses: bool = True
    log_errors: bool = True
    sensitive_patterns: list = field(default_factory=lambda: [...])
```

### setup_logging()

```python
def setup_logging(config: Optional[LogConfig] = None) -> logging.Logger:
    """
    Setup and configure logging for LLM API Router
    
    Args:
        config: Logging configuration, uses defaults if not provided
        
    Returns:
        Configured logger instance
    """
```

### get_logger()

```python
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name, defaults to "llm_api_router"
        
    Returns:
        Logger instance
    """
```

### generate_request_id()

```python
def generate_request_id() -> str:
    """Generate a unique request ID"""
```
