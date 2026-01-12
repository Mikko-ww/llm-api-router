# Quickstart: Unified LLM API Router

## Installation

```bash
pip install llm-api-router
```

## Basic Usage (Synchronous)

```python
from llm_api_router import Client, ProviderConfig

# Configure OpenAI
config = ProviderConfig(
    provider_type="openai",
    api_key="sk-...",
    default_model="gpt-4"
)

client = Client(config)

# Send request
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Switching Providers

```python
# Switch to Azure OpenAI (hypothetical example)
config = ProviderConfig(
    provider_type="azure",
    api_key="...",
    base_url="https://my-resource.openai.azure.com/",
    api_version="2023-05-15",
    default_model="gpt-4-deployment"
)

client = Client(config)
# ... code remains exactly the same
```

## Streaming

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Async Usage

```python
import asyncio
from llm_api_router import AsyncClient, ProviderConfig

async def main():
    config = ProviderConfig(provider_type="openai", api_key="...")
    client = AsyncClient(config)

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```
