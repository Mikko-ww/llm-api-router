# Error Handling and Retry Mechanism

## Overview

LLM API Router implements a robust error handling and retry mechanism to handle transient failures gracefully. This document describes the error handling features and how to use them.

## Exception Hierarchy

All exceptions in LLM API Router inherit from `LLMRouterError`, which provides a consistent interface for error handling:

```python
from llm_api_router.exceptions import (
    LLMRouterError,           # Base exception
    AuthenticationError,      # HTTP 401
    PermissionError,         # HTTP 403
    NotFoundError,           # HTTP 404
    BadRequestError,         # HTTP 400
    RateLimitError,          # HTTP 429
    ProviderError,           # HTTP 5xx or other errors
    TimeoutError,            # Request timeout
    NetworkError,            # Network/connection errors
    StreamError,             # Stream parsing errors
    RetryExhaustedError,     # All retries exhausted
)
```

### Exception Attributes

All exceptions include the following attributes:
- `message`: Error message
- `provider`: Provider name (e.g., "OpenAI", "Anthropic")
- `status_code`: HTTP status code (if applicable)
- `details`: Additional error details (dictionary)

Example:
```python
try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    print(f"Rate limited by {e.provider}: {e.message}")
    print(f"Status code: {e.status_code}")
    if "retry_after" in e.details:
        print(f"Retry after: {e.details['retry_after']} seconds")
```

## Retry Mechanism

### Automatic Retry

The library automatically retries requests on transient failures:
- Network errors (connection failures, timeouts)
- Rate limit errors (HTTP 429)
- Server errors (HTTP 5xx)

**Non-retryable errors** (fail immediately):
- Authentication errors (HTTP 401)
- Permission errors (HTTP 403)
- Not found errors (HTTP 404)
- Bad request errors (HTTP 400)

### RetryConfig

Configure retry behavior using `RetryConfig`:

```python
from llm_api_router import Client, ProviderConfig, RetryConfig

# Custom retry configuration
retry_config = RetryConfig(
    max_retries=5,              # Maximum number of retry attempts
    initial_delay=1.0,          # Initial delay in seconds
    max_delay=60.0,             # Maximum delay in seconds
    exponential_base=2.0,       # Exponential backoff base
    retry_on_status_codes=(429, 500, 502, 503, 504)  # Status codes to retry
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    retry_config=retry_config
)

client = Client(config)
```

### Default Retry Behavior

If no `RetryConfig` is provided, the library uses these defaults:
- **max_retries**: 3
- **initial_delay**: 1.0 second
- **max_delay**: 60.0 seconds
- **exponential_base**: 2.0
- **retry_on_status_codes**: (429, 500, 502, 503, 504)

### Exponential Backoff

Retry delays follow an exponential backoff pattern:
- Attempt 1: 1 second delay
- Attempt 2: 2 seconds delay
- Attempt 3: 4 seconds delay
- Attempt 4: 8 seconds delay
- And so on, capped at `max_delay`

### Disabling Retries

To disable retries entirely:

```python
retry_config = RetryConfig(max_retries=0)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    retry_config=retry_config
)
```

## Timeout Configuration

Configure request timeouts using the `timeout` parameter:

```python
config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    timeout=30.0  # 30 second timeout (default is 60.0)
)

client = Client(config)
```

The timeout applies to:
- HTTP connection establishment
- Request sending
- Response reception

## Error Handling Best Practices

### 1. Catch Specific Exceptions

```python
from llm_api_router.exceptions import (
    AuthenticationError,
    RateLimitError,
    RetryExhaustedError,
    LLMRouterError
)

try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError as e:
    # Handle authentication issues
    print(f"Invalid API key: {e.message}")
except RateLimitError as e:
    # Handle rate limits (though retries are automatic)
    print(f"Rate limited: {e.message}")
except RetryExhaustedError as e:
    # All retry attempts failed
    print(f"Request failed after retries: {e.message}")
    print(f"Original error: {e.details.get('original_error')}")
except LLMRouterError as e:
    # Catch-all for any LLM Router errors
    print(f"Error: {e.message}")
```

### 2. Use Context Managers

Always use context managers to ensure proper cleanup:

```python
# Synchronous
with Client(config) as client:
    response = client.chat.completions.create(...)

# Asynchronous
async with AsyncClient(config) as client:
    response = await client.chat.completions.create(...)
```

### 3. Handle Stream Errors

When using streaming, handle `StreamError`:

```python
from llm_api_router.exceptions import StreamError

try:
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")
except StreamError as e:
    print(f"Stream parsing error: {e.message}")
```

### 4. Log Error Details

Use the error attributes for logging:

```python
import logging

logger = logging.getLogger(__name__)

try:
    response = client.chat.completions.create(...)
except LLMRouterError as e:
    logger.error(
        "LLM request failed",
        extra={
            "provider": e.provider,
            "status_code": e.status_code,
            "message": e.message,
            "details": e.details
        }
    )
```

## Provider-Specific Error Handling

Different providers may return different error formats. The library normalizes these into consistent exceptions:

### OpenAI Errors
```python
# OpenAI returns: {"error": {"message": "...", "type": "..."}}
# Normalized to: AuthenticationError, RateLimitError, etc.
```

### Anthropic Errors
```python
# Anthropic returns: {"error": {"type": "...", "message": "..."}}
# Normalized to: AuthenticationError, RateLimitError, etc.
```

All providers use the same exception types, making error handling consistent across providers.

## Testing Error Handling

When testing, you can disable retries for faster tests:

```python
import pytest
from llm_api_router import ProviderConfig, RetryConfig

@pytest.fixture
def no_retry_config():
    return ProviderConfig(
        provider_type="openai",
        api_key="test-key",
        retry_config=RetryConfig(max_retries=0)
    )

def test_authentication_error(no_retry_config):
    # Test fails immediately without retries
    with pytest.raises(AuthenticationError):
        client = Client(no_retry_config)
        # ... test code ...
```

## Summary

The error handling and retry mechanism in LLM API Router provides:

✓ **Automatic retries** for transient failures  
✓ **Exponential backoff** to avoid overwhelming servers  
✓ **Consistent exceptions** across all providers  
✓ **Configurable behavior** for different use cases  
✓ **Detailed error information** for debugging  
✓ **Timeout control** for time-sensitive applications  

This makes your applications more robust and resilient to temporary failures.
