# LLM API Router

<p align="center">
  <strong>统一的大语言模型 API 路由库</strong>
</p>

<p align="center">
  <a href="https://github.com/Mikko-ww/llm-api-router/actions"><img src="https://github.com/Mikko-ww/llm-api-router/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/llm-api-router/"><img src="https://img.shields.io/pypi/v/llm-api-router" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

## 概述

`llm-api-router` 是一个 Python 库，为各种大语言模型提供商（如 OpenAI、Anthropic、Google Gemini 等）提供统一、一致且类型安全的接口。

### 核心特性

- 🔄 **统一接口** - 类似 OpenAI SDK 的 `client.chat.completions.create` 风格
- 🌐 **多厂商支持** - OpenAI、Anthropic、Gemini、DeepSeek、智谱、阿里云等
- ⚡ **零代码切换** - 仅需修改配置即可切换底层模型提供商
- 🌊 **流式支持** - 统一的 SSE 流式响应处理
- 🔧 **异步支持** - 原生支持 `asyncio` 和 `await`
- 📊 **可观测性** - 内置日志、指标收集、缓存等功能

## 快速开始

### 安装

```bash
pip install llm-api-router
```

### 基础用法

```python
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key="sk-...",
    default_model="gpt-3.5-turbo"
)

with Client(config) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
```

### 切换提供商

只需更改配置，代码无需修改：

```python
# 使用 Anthropic
config = ProviderConfig(
    provider_type="anthropic",
    api_key="sk-ant-...",
    default_model="claude-3-haiku-20240307"
)

# 使用本地 Ollama
config = ProviderConfig(
    provider_type="ollama",
    api_key="not-required",
    base_url="http://localhost:11434",
    default_model="llama3.2"
)
```

## 支持的提供商

| 提供商 | Chat | Embeddings | Function Calling |
|--------|:----:|:----------:|:----------------:|
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | - | ✅ |
| Google Gemini | ✅ | ✅ | - |
| DeepSeek | ✅ | - | - |
| 智谱 AI | ✅ | ✅ | - |
| 阿里云 | ✅ | ✅ | - |
| Ollama | ✅ | - | - |
| OpenRouter | ✅ | - | - |
| xAI | ✅ | - | - |

## 高级特性

- [响应缓存](user-guide/caching.md) - 减少重复 API 调用
- [配置管理](user-guide/configuration.md) - 详细配置参考
- [提供商支持](user-guide/providers.md) - 各提供商详情

## 下一步

- 📖 查看 [安装指南](getting-started/installation.md) 了解详细安装步骤
- 🚀 阅读 [快速开始](getting-started/quickstart.md) 开始使用
- 📚 浏览 [API 参考](api-reference/client.md) 了解完整 API
