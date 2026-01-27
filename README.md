# Unified LLM API Router Library (llm-api-router)

`llm-api-router` is a Python library designed to provide a unified, consistent, and type-safe interface for various Large Language Model (LLM) providers (such as OpenAI, Anthropic, DeepSeek, Google Gemini, etc.). It strictly adheres to the design style of the OpenAI Python SDK, minimizing the learning curve and supporting zero-code modification when switching underlying model providers.

## Core Features

- **Unified Interface**: Provides a `client.chat.completions.create` interface similar to the official OpenAI SDK.
- **Multi-Vendor Support**: Supports OpenAI, OpenRouter, DeepSeek, Anthropic, Google Gemini, Zhipu (ChatGLM), Alibaba (DashScope), and more.
- **Zero-Code Switching**: Switch underlying model providers simply by modifying the configuration.
- **Streaming Support**: Unified Server-Sent Events (SSE) streaming response handling, automatically managing streaming differences across vendors.
- **Async Support**: Native support for `asyncio` and `await` calls.
- **Type Safety**: Comprehensive Type Hints, strictly checked via MyPy.

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
