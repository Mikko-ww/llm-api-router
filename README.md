# 统一大语言模型 API 路由库 (llm-api-router)

`llm-api-router` 是一个 Python 库，旨在为不同的大语言模型（LLM）提供商（如 OpenAI、Azure OpenAI、Anthropic 等）提供统一、一致且类型安全的接口。它严格遵循 OpenAI Python SDK 的设计风格，降低学习成本，并支持零代码修改切换底层模型。

## 核心特性

- **统一接口**: 提供类似 OpenAI 官方 SDK 的 `client.chat.completions.create` 接口。
- **多厂商支持**: 目前支持 OpenAI，架构设计易于扩展至 Azure, Anthropic, Google 等。
- **零代码切换**: 仅需修改配置即可切换底层模型提供商。
- **流式支持**: 统一的 Server-Sent Events (SSE) 流式响应处理。
- **异步支持**: 原生支持 `asyncio` 和 `await` 调用。
- **类型安全**: 全面的类型提示 (Type Hints)，通过 MyPy 严格检查。

## 架构设计

本项目采用 **桥接模式 (Bridge Pattern)** 进行设计：

- **Client (抽象层)**: `Client` 和 `AsyncClient` 类负责对外暴露统一的 API 接口，处理参数校验和用户交互。
- **ProviderAdapter (实现层)**: `BaseProvider` 定义了统一的转换接口，具体子类（如 `OpenAIProvider`）负责将统一请求转换为特定厂商的 HTTP 请求，并将响应归一化。
- **HTTP 引擎**: 底层使用 `httpx` 处理所有同步和异步 HTTP 通信。

## 安装

项目使用 `uv` 进行包管理。

```bash
# 安装依赖
pip install llm-api-router

# 或者在开发环境中
uv pip install -e .
```

## 快速开始

### 1. 基础调用 (同步)

```python
from llm_api_router import Client, ProviderConfig

# 配置 OpenAI
config = ProviderConfig(
    provider_type="openai",
    api_key="sk-...",
    default_model="gpt-4"
)

with Client(config) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}]
    )
    print(response.choices[0].message.content)
```

### 2. 流式响应 (Streaming)

```python
with Client(config) as client:
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "讲一个长故事"}],
        stream=True
    )
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
```

### 3. 异步调用 (Async)

```python
import asyncio
from llm_api_router import AsyncClient, ProviderConfig

async def main():
    config = ProviderConfig(
        provider_type="openai",
        api_key="sk-...",
        default_model="gpt-3.5-turbo"
    )

    async with AsyncClient(config) as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "并发测试"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

### 4. 切换提供商

无需修改代码逻辑，只需更改配置对象：

```python
# 假设未来扩展了 Azure 支持
config = ProviderConfig(
    provider_type="azure",  # 切换类型
    api_key="azure-key...",
    base_url="https://my-resource.openai.azure.com",
    api_version="2023-05-15",
    default_model="gpt-4-deployment"
)
# Client 初始化代码保持不变
```

## 开发与测试

本项目使用 `uv` 管理开发环境。

1. **安装开发依赖**:
   ```bash
   uv pip install -e ".[dev]"
   ```

2. **运行测试**:
   ```bash
   uv run pytest
   ```

3. **静态类型检查**:
   ```bash
   uv run mypy src/llm_api_router
   ```