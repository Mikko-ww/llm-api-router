# 统一大语言模型 API 路由库 (llm-api-router)

`llm-api-router` 是一个 Python 库，旨在为不同的大语言模型（LLM）提供商（如 OpenAI、Anthropic、DeepSeek、Google Gemini 等）提供统一、一致且类型安全的接口。它严格遵循 OpenAI Python SDK 的设计风格，降低学习成本，并支持零代码修改切换底层模型。

## 核心特性

- **统一接口**: 提供类似 OpenAI 官方 SDK 的 `client.chat.completions.create` 接口。
- **多厂商支持**: 支持 OpenAI, OpenRouter, DeepSeek, Anthropic, Google Gemini, Zhipu (ChatGLM), Alibaba (DashScope) 等。
- **零代码切换**: 仅需修改配置即可切换底层模型提供商。
- **流式支持**: 统一的 Server-Sent Events (SSE) 流式响应处理，自动处理不同厂商的流式差异。
- **异步支持**: 原生支持 `asyncio` 和 `await` 调用。
- **类型安全**: 全面的类型提示 (Type Hints)，通过 MyPy 严格检查。

## 架构设计

本项目采用 **桥接模式 (Bridge Pattern)** 进行设计：

- **Client (抽象层)**: `Client` 和 `AsyncClient` 类负责对外暴露统一的 API 接口。内部使用 `ProviderFactory` 动态加载具体的厂商实现。
- **ProviderAdapter (实现层)**: `BaseProvider` 定义了统一的转换接口，具体子类（如 `OpenAIProvider`, `AnthropicProvider`）负责将统一请求转换为特定厂商的 HTTP 请求，并将响应归一化。
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

### 1. 基础调用 (OpenRouter 示例)

```python
from llm_api_router import Client, ProviderConfig

# OpenRouter 配置
config = ProviderConfig(
    provider_type="openrouter",
    api_key="sk-or-...",
    default_model="nvidia/nemotron-3-nano-30b-a3b:free"
)

with Client(config) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}]
    )
    print(response.choices[0].message.content)
```

### 2. 切换提供商 (如 DeepSeek, Anthropic)

只需更改配置，**代码完全无需修改**：

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

# 智谱 (ZhipuAI / ChatGLM)
zhipu_config = ProviderConfig(
    provider_type="zhipu",
    api_key="id.secret",  # 智谱 API Key (无需手动生成 token，库会自动处理)
    default_model="glm-4"
)

# 阿里云 (DashScope / Qwen)
aliyun_config = ProviderConfig(
    provider_type="aliyun",
    api_key="sk-...",
    default_model="qwen-max"
)

# 使用 DeepSeek 配置初始化客户端
with Client(deepseek_config) as client:
    # ... 调用逻辑不变
    pass
```

### 3. 流式响应 (Streaming)

```python
with Client(gemini_config) as client:
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "写一首关于AI的诗"}],
        stream=True
    )
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
```

### 4. 异步调用 (Async)

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
            messages=[{"role": "user", "content": "并发测试"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

## 支持的模型提供商

| 提供商 | provider_type | 典型模型 | 备注 |
|---|---|---|---|
| **OpenAI** | `openai` | gpt-4, gpt-3.5-turbo | 官方格式 |
| **OpenRouter** | `openrouter` | * | 聚合网关 |
| **DeepSeek** | `deepseek` | deepseek-chat | OpenAI 兼容 |
| **Anthropic** | `anthropic` | claude-3-opus | 自动处理 System Prompt 提取 |
| **Google Gemini** | `gemini` | gemini-1.5-pro | 支持 System Instruction |
| **ZhipuAI** | `zhipu` | glm-4 | 自动处理 JWT 鉴权 |
| **Alibaba** | `aliyun` | qwen-max | 支持 DashScope 原生协议 |

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