# Unified LLM API Router Library (llm-api-router)

[![Tests](https://github.com/Mikko-ww/llm-api-router/actions/workflows/tests.yml/badge.svg)](https://github.com/Mikko-ww/llm-api-router/actions/workflows/tests.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`llm-api-router` is a Python library designed to provide a unified, consistent, and type-safe interface for various Large Language Model (LLM) providers (such as OpenAI, Anthropic, DeepSeek, Google Gemini, etc.). It strictly adheres to the design style of the OpenAI Python SDK, minimizing the learning curve and supporting zero-code modification when switching underlying model providers.

## Core Features

- **Unified Interface**: Provides a `client.chat.completions.create` interface similar to the official OpenAI SDK.
- **Multi-Vendor Support**: Supports OpenAI, OpenRouter, DeepSeek, Anthropic, Google Gemini, Zhipu (ChatGLM), Alibaba (DashScope), and more.
- **Zero-Code Switching**: Switch underlying model providers simply by modifying the configuration.
- **Streaming Support**: Unified Server-Sent Events (SSE) streaming response handling, automatically managing streaming differences across vendors.
- **Async Support**: Native support for `asyncio` and `await` calls.
- **Type Safety**: Comprehensive Type Hints, strictly checked via MyPy.
- **Embeddings API**: Unified text embeddings interface supporting OpenAI, Gemini, Zhipu, and Aliyun providers.
- **Function Calling**: Unified tool/function calling support for OpenAI and Anthropic providers.
- **Connection Pool Optimization**: Configurable HTTP connection pooling with fine-grained timeout control for optimal performance and resource efficiency.
- **Response Caching**: Optional response caching with memory and Redis backends to reduce redundant API calls and improve performance.

## Architecture Design

This project is designed using the **Bridge Pattern**:

- **Client (Abstraction Layer)**: The `Client` and `AsyncClient` classes are responsible for exposing the unified API interface. Internally, they use `ProviderFactory` to dynamically load specific vendor implementations.
- **ProviderAdapter (Implementation Layer)**: `BaseProvider` defines the unified conversion interface. Concrete subclasses (such as `OpenAIProvider`, `AnthropicProvider`) are responsible for converting unified requests into specific vendor HTTP requests and normalizing the responses.
- **HTTP Engine**: Uses `httpx` under the hood to handle all synchronous and asynchronous HTTP communications.

## Installation

The project uses `uv` for package management.

```bash
# Install dependencies
pip install llm-api-router

# Or in a development environment
uv pip install -e .
```

## Quick Start

### 1. Basic Usage (OpenRouter Example)

```python
from llm_api_router import Client, ProviderConfig

# OpenRouter Configuration
config = ProviderConfig(
    provider_type="openrouter",
    api_key="sk-or-...",
    default_model="nvidia/nemotron-3-nano-30b-a3b:free"
)

with Client(config) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello, please introduce yourself"}]
    )
    print(response.choices[0].message.content)
```

### 2. Switching Providers (e.g., DeepSeek, Anthropic)

Simply change the configuration, **no code changes required**:

```python
# DeepSeek
deepseek_config = ProviderConfig(
    provider_type="deepseek",
    api_key="sk-...",
    default_model="deepseek-chat"
)

# Anthropic (Claude)
anthropic_config = ProviderConfig(
    provider_type="anthropic",
    api_key="sk-ant-...",
    default_model="claude-3-5-sonnet-20240620"
)

# Google Gemini
gemini_config = ProviderConfig(
    provider_type="gemini",
    api_key="AIza...",
    default_model="gemini-1.5-flash"
)

# ZhipuAI (ChatGLM)
zhipu_config = ProviderConfig(
    provider_type="zhipu",
    api_key="id.secret",  # Zhipu API Key (no manual token generation needed, library handles it)
    default_model="glm-4"
)

# Alibaba (DashScope / Qwen)
aliyun_config = ProviderConfig(
    provider_type="aliyun",
    api_key="sk-...",
    default_model="qwen-max"
)

# Ollama (Local Models)
ollama_config = ProviderConfig(
    provider_type="ollama",
    api_key="not-required",  # Ollama doesn't require an API key
    base_url="http://localhost:11434",  # Default Ollama server URL
    default_model="llama3.2"
)

# Initialize client with DeepSeek configuration
with Client(deepseek_config) as client:
    # ... calling logic remains unchanged
    pass
```

### 3. Streaming Response

```python
with Client(gemini_config) as client:
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "Write a poem about AI"}],
        stream=True
    )
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
```

### 4. Async Call

```python
import asyncio
from llm_api_router import AsyncClient, ProviderConfig

async def main():
    config = ProviderConfig(
        provider_type="aliyun",
        api_key="sk-...",
        default_model="qwen-turbo"
    )

    async with AsyncClient(config) as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Concurrency test"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

## Embeddings API

The library provides a unified interface for creating text embeddings, supporting multiple providers:

### Basic Usage

```python
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="text-embedding-3-small"
)

with Client(config) as client:
    # Single text embedding
    response = client.embeddings.create(input="Hello, world!")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    
    # Batch embeddings
    response = client.embeddings.create(
        input=["Text 1", "Text 2", "Text 3"]
    )
    print(f"Created {len(response.data)} embeddings")
```

### Custom Dimensions (OpenAI text-embedding-3-* models)

```python
response = client.embeddings.create(
    input="Test text",
    model="text-embedding-3-small",
    dimensions=256  # Reduce dimensions to save storage
)
```

### Async Embeddings

```python
async with AsyncClient(config) as client:
    response = await client.embeddings.create(
        input=["Async text 1", "Async text 2"]
    )
```

### Supported Embedding Providers

| Provider | Default Model | Notes |
|---|---|---|
| **OpenAI** | text-embedding-3-small | Supports custom dimensions |
| **Gemini** | embedding-001 | Uses batchEmbedContents API |
| **Zhipu** | embedding-3 | OpenAI-compatible format |
| **Aliyun** | text-embedding-v2 | DashScope format |

For complete examples, see [examples/embeddings_example.py](examples/embeddings_example.py).

## Function Calling (Tool Use)

The library provides unified support for function calling (also known as tool use), allowing models to intelligently call functions to retrieve information or perform actions. This feature is supported by OpenAI, Anthropic, and other compatible providers.

### Basic Function Calling

```python
from llm_api_router import Client, ProviderConfig, Tool, FunctionDefinition
import json

# Define a tool
tools = [
    Tool(
        type="function",
        function=FunctionDefinition(
            name="get_weather",
            description="Get the current weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        )
    )
]

config = ProviderConfig(
    provider_type="openai",  # or "anthropic"
    api_key="your-api-key",
    default_model="gpt-4"
)

with Client(config) as client:
    # Request with tools
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tools,
        tool_choice="auto"  # "auto", "required", or "none"
    )
    
    # Check if model wants to call a function
    message = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"Calling {function_name} with args: {function_args}")
            
            # Execute function and get result
            result = get_weather(**function_args)  # Your function
            
            # Send result back to model
            # (Add tool result message and make another request)
```

### Multi-turn Function Calling

For complex interactions, you can have multiple turns of function calling:

```python
messages = [
    {"role": "user", "content": "Plan a trip to Tokyo"}
]

response = client.chat.completions.create(messages=messages, tools=tools)

# Model decides to call functions
if response.choices[0].message.tool_calls:
    # Add assistant message
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in response.choices[0].message.tool_calls
        ]
    })
    
    # Add function results
    for tool_call in response.choices[0].message.tool_calls:
        result = execute_function(tool_call.function.name, 
                                 json.loads(tool_call.function.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # Get final response
    final_response = client.chat.completions.create(messages=messages, tools=tools)
    print(final_response.choices[0].message.content)
```

### Supported Function Calling Providers

| Provider | Status | Notes |
|---|---|---|
| **OpenAI** | ✅ Fully Supported | Native tool calling support |
| **Anthropic** | ✅ Fully Supported | Converted to Anthropic's tool use format |

For complete examples, see:
- [examples/function_calling_example.py](examples/function_calling_example.py) - Basic function calling
- [examples/multi_turn_function_calling.py](examples/multi_turn_function_calling.py) - Advanced multi-turn agent example

## Error Handling and Retry

The library provides robust error handling with automatic retry for transient failures:

### Automatic Retry

Requests are automatically retried on:
- Network errors (connection failures, timeouts)
- Rate limit errors (HTTP 429)
- Server errors (HTTP 5xx)

```python
from llm_api_router import Client, ProviderConfig, RetryConfig

# Custom retry configuration
retry_config = RetryConfig(
    max_retries=5,              # Maximum retry attempts (default: 3)
    initial_delay=1.0,          # Initial delay in seconds (default: 1.0)
    max_delay=60.0,             # Maximum delay in seconds (default: 60.0)
    exponential_base=2.0,       # Exponential backoff base (default: 2.0)
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    retry_config=retry_config,
    timeout=30.0  # Request timeout in seconds (default: 60.0)
)
```

### Exception Handling

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
    print(f"Invalid API key: {e.message}")
except RetryExhaustedError as e:
    print(f"Request failed after retries: {e.message}")
except LLMRouterError as e:
    print(f"Error: {e.message}")
```

For detailed error handling documentation, see [docs/error-handling.md](docs/error-handling.md).

## Performance Optimization

The library includes advanced HTTP connection pool optimization features for improved performance and resource efficiency. You can configure:

- Connection pool limits and keepalive settings
- Fine-grained timeout control (connect, read, write, pool)
- Streaming buffer sizes

For detailed performance optimization documentation and configuration examples, see [docs/connection_pool_optimization.md](docs/connection_pool_optimization.md).

## Response Caching

The library supports optional response caching to reduce redundant API calls and improve performance. Caching is especially useful for:

- Development and testing environments
- Repeated queries with identical parameters
- Reducing API costs
- Improving response times for cached requests

### Basic Cache Configuration

```python
from llm_api_router import Client, ProviderConfig
from llm_api_router.cache import CacheConfig

# Configure cache with memory backend
cache_config = CacheConfig(
    enabled=True,
    backend="memory",        # "memory" or "redis"
    ttl=3600,               # Time-to-live in seconds (1 hour)
    max_size=1000           # Maximum cached items (memory backend only)
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    cache_config=cache_config
)

with Client(config) as client:
    # First call - hits the API
    response1 = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    
    # Second identical call - hits the cache (much faster!)
    response2 = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    
    # Get cache statistics
    stats = client.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

### Redis Cache Backend

For production environments or distributed systems, use Redis backend:

```python
# Requires: pip install redis

cache_config = CacheConfig(
    enabled=True,
    backend="redis",
    ttl=3600,
    redis_url="redis://localhost:6379/0",  # Redis connection URL
    redis_prefix="myapp:"                   # Key prefix for organization
)
```

### Cache Management

```python
# Get cache statistics
stats = client.get_cache_stats()
print(f"Backend: {stats['backend']}")
print(f"Size: {stats['size']}")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")

# Clear cache
client.clear_cache()
```

### Important Notes

- **Streaming requests are not cached** - Each streaming request hits the API
- Cache keys are generated from request content (messages, model, temperature, etc.)
- Non-content fields like `request_id` and `stream` don't affect cache keys
- Both `Client` and `AsyncClient` support caching

For more examples, see [examples/cache_example.py](examples/cache_example.py).

## Supported Model Providers

| Provider | provider_type | Typical Models | Notes |
|---|---|---|---|
| **OpenAI** | `openai` | gpt-4, gpt-3.5-turbo | Official format |
| **OpenRouter** | `openrouter` | * | Aggregation Gateway |
| **DeepSeek** | `deepseek` | deepseek-chat | OpenAI Compatible |
| **Anthropic** | `anthropic` | claude-3-opus | Handles System Prompt extraction automatically |
| **Google Gemini** | `gemini` | gemini-1.5-pro | Supports System Instruction |
| **ZhipuAI** | `zhipu` | glm-4 | Automatically handles JWT authentication |
| **Alibaba** | `aliyun` | qwen-max | Supports DashScope native protocol |
| **Ollama** | `ollama` | llama3.2, mistral, etc. | Local models, no API key required |

## Using Ollama (Local Models)

[Ollama](https://ollama.com/) allows you to run open-source LLMs locally on your machine. This is perfect for:
- Privacy-sensitive applications
- Offline development
- Cost-free experimentation with various models

### Installing and Running Ollama

1. **Install Ollama**: Visit [https://ollama.com/download](https://ollama.com/download) and follow the installation instructions for your OS.

2. **Start the Ollama service** (it typically starts automatically after installation):
   ```bash
   ollama serve
   ```

3. **Pull a model**:
   ```bash
   ollama pull llama3.2
   # or other models like:
   # ollama pull mistral
   # ollama pull codellama
   ```

4. **Verify the service is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Using Ollama with llm-api-router

```python
from llm_api_router import Client, ProviderConfig

# Configure Ollama provider
config = ProviderConfig(
    provider_type="ollama",
    api_key="not-required",  # Ollama doesn't require authentication
    base_url="http://localhost:11434",  # Default Ollama server URL
    default_model="llama3.2"  # Use any model you've pulled
)

# Use it just like any other provider
with Client(config) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello! Introduce yourself."}]
    )
    print(response.choices[0].message.content)
    
    # Streaming also works
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "Write a short poem about AI"}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

**Note**: Ollama uses NDJSON (newline-delimited JSON) for streaming, which is automatically handled by the adapter.

## Development & Testing

This project uses `uv` to manage the development environment.

1. **Install Development Dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

2. **Run Tests**:
   ```bash
   uv run pytest
   ```

3. **Static Type Checking**:
   ```bash
   uv run mypy src/llm_api_router
   ```
