# 统一大语言模型 API 路由库 (llm-api-router)

[![Tests](https://github.com/Mikko-ww/llm-api-router/actions/workflows/tests.yml/badge.svg)](https://github.com/Mikko-ww/llm-api-router/actions/workflows/tests.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`llm-api-router` 是一个 Python 库，旨在为不同的大语言模型（LLM）提供商（如 OpenAI、Anthropic、DeepSeek、Google Gemini 等）提供统一、一致且类型安全的接口。它严格遵循 OpenAI Python SDK 的设计风格，降低学习成本，并支持零代码修改切换底层模型。

## 核心特性

- **统一接口**: 提供类似 OpenAI 官方 SDK 的 `client.chat.completions.create` 接口。
- **多厂商支持**: 支持 OpenAI, OpenRouter, DeepSeek, Anthropic, Google Gemini, Zhipu (ChatGLM), Alibaba (DashScope) 等。
- **零代码切换**: 仅需修改配置即可切换底层模型提供商。
- **流式支持**: 统一的 Server-Sent Events (SSE) 流式响应处理，自动处理不同厂商的流式差异。
- **异步支持**: 原生支持 `asyncio` 和 `await` 调用。
- **类型安全**: 全面的类型提示 (Type Hints)，通过 MyPy 严格检查。
- **Embeddings API**: 统一的文本嵌入（Embeddings）接口，支持 OpenAI, Gemini, Zhipu 和 Aliyun 提供商。
- **函数调用 (Function Calling)**: 统一的工具/函数调用支持，适配 OpenAI 和 Anthropic 提供商。
- **连接池优化**: 可配置的 HTTP 连接池，支持细粒度的超时控制，以获得最佳性能和资源效率。
- **响应缓存**: 可选的响应缓存支持（内存和 Redis 后端），以减少冗余 API 调用并提高性能。

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

# Ollama (本地模型)
ollama_config = ProviderConfig(
    provider_type="ollama",
    api_key="not-required",  # Ollama 不需要 API 密钥
    base_url="http://localhost:11434",  # 默认 Ollama 服务器 URL
    default_model="llama3.2"
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

## Embeddings API

本库提供了一个统一的接口来创建文本嵌入，支持多个提供商：

### 基础用法

```python
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="text-embedding-3-small"
)

with Client(config) as client:
    # 单个文本嵌入
    response = client.embeddings.create(input="Hello, world!")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    
    # 批量文本嵌入
    response = client.embeddings.create(
        input=["Text 1", "Text 2", "Text 3"]
    )
    print(f"Created {len(response.data)} embeddings")
```

### 自定义维度 (OpenAI text-embedding-3-* 模型)

```python
response = client.embeddings.create(
    input="Test text",
    model="text-embedding-3-small",
    dimensions=256  # 减少维度以节省存储
)
```

### 异步 Embeddings

```python
async with AsyncClient(config) as client:
    response = await client.embeddings.create(
        input=["Async text 1", "Async text 2"]
    )
```

### 支持的 Embedding 提供商

| 提供商 | 默认模型 | 备注 |
|---|---|---|
| **OpenAI** | text-embedding-3-small | 支持自定义维度 |
| **Gemini** | embedding-001 | 使用 batchEmbedContents API |
| **Zhipu** | embedding-3 | OpenAI 兼容格式 |
| **Aliyun** | text-embedding-v2 | DashScope 格式 |

完整示例请参考 [examples/embeddings_example.py](examples/embeddings_example.py)。

## 函数调用 (Tool Use)

本库提供了对函数调用（也称为工具使用）的统一支持，允许模型智能地调用函数来获取信息或执行操作。此功能由 OpenAI、Anthropic 和其他兼容的提供商支持。

### 基础函数调用

```python
from llm_api_router import Client, ProviderConfig, Tool, FunctionDefinition
import json

# 定义工具
tools = [
    Tool(
        type="function",
        function=FunctionDefinition(
            name="get_weather",
            description="获取给定位置的当前天气",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和州，例如 San Francisco, CA"
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
    provider_type="openai",  # 或 "anthropic"
    api_key="your-api-key",
    default_model="gpt-4"
)

with Client(config) as client:
    # 带有工具的请求
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "旧金山的天气怎么样？"}],
        tools=tools,
        tool_choice="auto"  # "auto", "required", 或 "none"
    )
    
    # 检查模型是否想要调用函数
    message = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"Calling {function_name} with args: {function_args}")
            
            # 执行函数并获取结果
            result = get_weather(**function_args)  # 你的函数
            
            # 将结果发送回模型
            # (添加工具结果消息并再次发出请求)
```

### 多轮函数调用

对于复杂的交互，您可以进行多轮函数调用：

```python
messages = [
    {"role": "user", "content": "计划一次去东京的旅行"}
]

response = client.chat.completions.create(messages=messages, tools=tools)

# 模型决定调用函数
if response.choices[0].message.tool_calls:
    # 添加助手消息
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
    
    # 添加函数结果
    for tool_call in response.choices[0].message.tool_calls:
        result = execute_function(tool_call.function.name, 
                                 json.loads(tool_call.function.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # 获取最终响应
    final_response = client.chat.completions.create(messages=messages, tools=tools)
    print(final_response.choices[0].message.content)
```

### 支持函数调用的提供商

| 提供商 | 状态 | 备注 |
|---|---|---|
| **OpenAI** | ✅ 完全支持 | 原生工具调用支持 |
| **Anthropic** | ✅ 完全支持 | 转换为 Anthropic 的工具使用格式 |

完整示例请参考：
- [examples/function_calling_example.py](examples/function_calling_example.py) - 基础函数调用
- [examples/multi_turn_function_calling.py](examples/multi_turn_function_calling.py) - 高级多轮代理示例

## 错误处理与重试

本库提供了强大的错误处理机制，并针对临时故障自动重试：

### 自动重试

在以下情况下会自动重试请求：
- 网络错误（连接失败、超时）
- 速率限制错误 (HTTP 429)
- 服务器错误 (HTTP 5xx)

```python
from llm_api_router import Client, ProviderConfig, RetryConfig

# 自定义重试配置
retry_config = RetryConfig(
    max_retries=5,              # 最大重试次数 (默认: 3)
    initial_delay=1.0,          # 初始延迟秒数 (默认: 1.0)
    max_delay=60.0,             # 最大延迟秒数 (默认: 60.0)
    exponential_base=2.0,       # 指数退避基数 (默认: 2.0)
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    retry_config=retry_config,
    timeout=30.0  # 请求超时秒数 (默认: 60.0)
)
```

### 异常处理

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

详细的错误处理文档请参考 [docs/error-handling.md](docs/error-handling.md)。

## 性能优化

本库包含高级 HTTP 连接池优化功能，以提高性能和资源效率。您可以配置：

- 连接池限制和 keepalive 设置
- 细粒度的超时控制（连接、读取、写入、池）
- 流式缓冲区大小

详细的性能优化文档和配置示例请参考 [docs/connection_pool_optimization.md](docs/connection_pool_optimization.md)。

## 响应缓存

本库支持可选的响应缓存，以减少冗余 API 调用并提高性能。缓存特别适用于：

- 开发和测试环境
- 具有相同参数的重复查询
- 降低 API 成本
- 提高缓存请求的响应速度

### 基础缓存配置

```python
from llm_api_router import Client, ProviderConfig
from llm_api_router.cache import CacheConfig

# 配置内存后端缓存
cache_config = CacheConfig(
    enabled=True,
    backend="memory",        # "memory" 或 "redis"
    ttl=3600,               # 生存时间秒数 (1 小时)
    max_size=1000           # 最大缓存项数 (仅内存后端)
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    cache_config=cache_config
)

with Client(config) as client:
    # 第一次调用 - 请求 API
    response1 = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    
    # 第二次相同的调用 - 命中缓存 (更快!)
    response2 = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is Python?"}]
    )
    
    # 获取缓存统计
    stats = client.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

### Redis 缓存后端

对于生产环境或分布式系统，请使用 Redis 后端：

```python
# 需要: pip install redis

cache_config = CacheConfig(
    enabled=True,
    backend="redis",
    ttl=3600,
    redis_url="redis://localhost:6379/0",  # Redis 连接 URL
    redis_prefix="myapp:"                   # 组织键前缀
)
```

### 缓存管理

```python
# 获取缓存统计
stats = client.get_cache_stats()
print(f"Backend: {stats['backend']}")
print(f"Size: {stats['size']}")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")

# 清除缓存
client.clear_cache()
```

### 重要提示

- **流式请求不会被缓存** - 每个流式请求都会请求 API
- 缓存键由请求内容（消息、模型、温度等）生成
- 非内容字段如 `request_id` 和 `stream` 不会影响缓存键
- `Client` 和 `AsyncClient` 都支持缓存

更多示例请参考 [examples/cache_example.py](examples/cache_example.py)。

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
| **Ollama** | `ollama` | llama3.2, mistral 等 | 本地模型，无需 API 密钥 |

## 使用 Ollama（本地模型）

[Ollama](https://ollama.com/) 允许您在本地机器上运行开源大语言模型。这非常适合：
- 隐私敏感的应用
- 离线开发
- 免费体验各种模型

### 安装和运行 Ollama

1. **安装 Ollama**：访问 [https://ollama.com/download](https://ollama.com/download) 并按照您的操作系统的安装说明进行操作。

2. **启动 Ollama 服务**（通常在安装后会自动启动）：
   ```bash
   ollama serve
   ```

3. **拉取模型**：
   ```bash
   ollama pull llama3.2
   # 或其他模型，如：
   # ollama pull mistral
   # ollama pull codellama
   ```

4. **验证服务是否运行**：
   ```bash
   curl http://localhost:11434/api/tags
   ```

### 在 llm-api-router 中使用 Ollama

```python
from llm_api_router import Client, ProviderConfig

# 配置 Ollama 提供商
config = ProviderConfig(
    provider_type="ollama",
    api_key="not-required",  # Ollama 不需要身份验证
    base_url="http://localhost:11434",  # 默认 Ollama 服务器 URL
    default_model="llama3.2"  # 使用您已拉取的任何模型
)

# 像使用其他提供商一样使用它
with Client(config) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "你好！请介绍一下你自己。"}]
    )
    print(response.choices[0].message.content)
    
    # 流式响应也可以工作
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "写一首关于人工智能的短诗"}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

**注意**：Ollama 使用 NDJSON（换行分隔的 JSON）进行流式传输，适配器会自动处理。

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