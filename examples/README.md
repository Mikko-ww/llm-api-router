# 示例项目

本目录包含 LLM API Router 的各种使用示例，帮助你快速上手。

## 示例列表

### 基础示例

| 文件 | 描述 |
|------|------|
| [quick_start.py](quick_start.py) | 5 行代码快速开始 |
| [ollama_example.py](ollama_example.py) | 本地 Ollama 使用 |
| [logging_example.py](logging_example.py) | 日志配置示例 |

### 功能示例

| 文件 | 描述 |
|------|------|
| [function_calling_example.py](function_calling_example.py) | Function Calling 示例 |
| [multi_turn_function_calling.py](multi_turn_function_calling.py) | 多轮 Function Calling |
| [embeddings_example.py](embeddings_example.py) | 文本嵌入示例 |
| [cache_example.py](cache_example.py) | 缓存使用示例 |
| [error_handling_example.py](error_handling_example.py) | 错误处理示例 |

### 高级示例

| 文件 | 描述 |
|------|------|
| [metrics_example.py](metrics_example.py) | 指标收集示例 |
| [connection_pool_optimization.py](connection_pool_optimization.py) | 连接池优化 |
| [provider_switcher.py](provider_switcher.py) | 动态切换提供商 |
| [chatbot_demo.py](chatbot_demo.py) | 完整聊天机器人 |
| [rag_example.py](rag_example.py) | RAG 应用示例 |

### 配置文件

| 文件 | 描述 |
|------|------|
| [grafana_dashboard.json](grafana_dashboard.json) | Grafana 仪表板配置 |

## 运行示例

### 1. 安装依赖

```bash
# 安装基础包
pip install llm-api-router

# 或者从源码安装（开发模式）
pip install -e ".[dev]"
```

### 2. 设置环境变量

```bash
export OPENAI_API_KEY="sk-your-api-key"
export ANTHROPIC_API_KEY="sk-ant-your-api-key"
# 根据需要设置其他提供商的 API Key
```

### 3. 运行示例

```bash
# 运行快速开始示例
python examples/quick_start.py

# 运行聊天机器人
python examples/chatbot_demo.py

# 运行 RAG 示例
python examples/rag_example.py
```

## 快速开始示例

最简单的使用方式：

```python
# examples/quick_start.py
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(provider_type="openai", api_key="sk-xxx", default_model="gpt-3.5-turbo")
with Client(config) as client:
    response = client.chat.completions.create(messages=[{"role": "user", "content": "Hello!"}])
    print(response.choices[0].message.content)
```

## 提供商切换示例

动态切换不同提供商：

```python
# examples/provider_switcher.py
from llm_api_router import Client, ProviderConfig

providers = {
    "openai": ProviderConfig(provider_type="openai", api_key="sk-xxx", default_model="gpt-4o"),
    "anthropic": ProviderConfig(provider_type="anthropic", api_key="sk-ant-xxx", default_model="claude-3-5-sonnet-20241022"),
    "local": ProviderConfig(provider_type="ollama", api_key="", base_url="http://localhost:11434", default_model="llama3.2"),
}

# 根据环境或配置选择提供商
provider_name = "openai"  # 可以从配置文件或环境变量读取
with Client(providers[provider_name]) as client:
    response = client.chat.completions.create(messages=[{"role": "user", "content": "Hello!"}])
    print(response.choices[0].message.content)
```

## 需要帮助？

- 查看 [文档](https://llm-api-router.readthedocs.io/)
- 提交 [Issue](https://github.com/Mikko-ww/llm-api-router/issues)
- 参考 [FAQ](../docs/faq.md)
