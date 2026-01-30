# 提供商配置

LLM API Router 支持多个 LLM 提供商，本文档详细介绍各提供商的配置方式。

## 支持的提供商

| 提供商 | Provider Type | 模型示例 | 特点 |
|--------|--------------|----------|------|
| OpenAI | `openai` | gpt-4o, gpt-3.5-turbo | 最广泛使用 |
| Anthropic | `anthropic` | claude-3-5-sonnet | 长上下文，推理能力强 |
| Google Gemini | `gemini` | gemini-1.5-flash | 多模态，免费额度 |
| DeepSeek | `deepseek` | deepseek-chat | 高性价比 |
| 阿里云百炼 | `aliyun` | qwen-turbo | 国内部署 |
| 智谱 AI | `zhipu` | glm-4 | 中文优化 |
| xAI | `xai` | grok-beta | 最新推理 |
| OpenRouter | `openrouter` | 多种模型 | 统一访问多家模型 |
| Ollama | `ollama` | llama3.2 | 本地部署 |

## 基础配置

所有提供商使用统一的 `ProviderConfig` 配置：

```python
from llm_api_router import ProviderConfig

config = ProviderConfig(
    provider_type="openai",      # 必填：提供商类型
    api_key="sk-xxx",            # 必填：API 密钥
    default_model="gpt-3.5-turbo", # 可选：默认模型
    base_url=None,               # 可选：自定义 API 地址
    timeout=30.0,                # 可选：请求超时时间（秒）
    max_retries=3,               # 可选：最大重试次数
)
```

## OpenAI

```python
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key="sk-xxx",  # 或 os.environ["OPENAI_API_KEY"]
    default_model="gpt-4o"
)

with Client(config) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### 可用模型

- `gpt-4o` - 最新多模态模型
- `gpt-4o-mini` - 轻量版
- `gpt-4-turbo` - 高性能
- `gpt-3.5-turbo` - 经济实惠

### Azure OpenAI

使用 Azure 部署的 OpenAI 模型：

```python
config = ProviderConfig(
    provider_type="openai",
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    default_model="gpt-4"
)
```

## Anthropic

```python
config = ProviderConfig(
    provider_type="anthropic",
    api_key="sk-ant-xxx",
    default_model="claude-3-5-sonnet-20241022"
)
```

### 可用模型

- `claude-3-5-sonnet-20241022` - 最新 Sonnet
- `claude-3-opus-20240229` - 最强大
- `claude-3-haiku-20240307` - 最快速

## Google Gemini

```python
config = ProviderConfig(
    provider_type="gemini",
    api_key="AIza-xxx",
    default_model="gemini-1.5-flash"
)
```

### 可用模型

- `gemini-1.5-pro` - 高性能
- `gemini-1.5-flash` - 快速响应
- `gemini-1.5-flash-8b` - 轻量版

## DeepSeek

```python
config = ProviderConfig(
    provider_type="deepseek",
    api_key="sk-xxx",
    default_model="deepseek-chat"
)
```

### 可用模型

- `deepseek-chat` - 对话模型
- `deepseek-coder` - 代码模型

## 阿里云百炼

```python
config = ProviderConfig(
    provider_type="aliyun",
    api_key="sk-xxx",
    default_model="qwen-turbo"
)
```

### 可用模型

- `qwen-turbo` - 快速版
- `qwen-plus` - 增强版
- `qwen-max` - 最强版

## 智谱 AI

```python
config = ProviderConfig(
    provider_type="zhipu",
    api_key="xxx.xxx",
    default_model="glm-4"
)
```

### 可用模型

- `glm-4` - 最新版
- `glm-4-flash` - 快速版
- `glm-3-turbo` - 经济版

## xAI (Grok)

```python
config = ProviderConfig(
    provider_type="xai",
    api_key="xai-xxx",
    default_model="grok-beta"
)
```

## OpenRouter

OpenRouter 提供统一访问多家模型的能力：

```python
config = ProviderConfig(
    provider_type="openrouter",
    api_key="sk-or-xxx",
    default_model="openai/gpt-4o"
)
```

### 模型格式

OpenRouter 模型使用 `provider/model` 格式：

- `openai/gpt-4o`
- `anthropic/claude-3-sonnet`
- `meta-llama/llama-3-70b-instruct`

## Ollama (本地部署)

Ollama 允许在本地运行开源模型：

```python
config = ProviderConfig(
    provider_type="ollama",
    api_key="not-required",  # Ollama 不需要 API Key
    base_url="http://localhost:11434",
    default_model="llama3.2"
)
```

### 启动 Ollama

```bash
# 安装 Ollama（macOS）
brew install ollama

# 启动服务
ollama serve

# 拉取模型
ollama pull llama3.2
ollama pull codellama
ollama pull mistral
```

## 多提供商使用

可以使用 `create_client` 工厂函数快速创建客户端：

```python
from llm_api_router import create_client, ProviderConfig

# 定义多个提供商配置
configs = {
    "openai": ProviderConfig(
        provider_type="openai",
        api_key="sk-xxx",
        default_model="gpt-4o"
    ),
    "anthropic": ProviderConfig(
        provider_type="anthropic", 
        api_key="sk-ant-xxx",
        default_model="claude-3-5-sonnet-20241022"
    ),
    "local": ProviderConfig(
        provider_type="ollama",
        api_key="",
        base_url="http://localhost:11434",
        default_model="llama3.2"
    ),
}

# 根据需求选择提供商
provider = "openai"  # 可动态切换
with create_client(configs[provider]) as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

## 负载均衡

使用 `LoadBalancer` 在多个提供商间分配请求：

```python
from llm_api_router import LoadBalancer, ProviderConfig

# 配置多个提供商
configs = [
    ProviderConfig(provider_type="openai", api_key="sk-1", default_model="gpt-4o"),
    ProviderConfig(provider_type="anthropic", api_key="sk-ant-1", default_model="claude-3-5-sonnet-20241022"),
]

# 创建负载均衡器
lb = LoadBalancer(configs, strategy="round_robin")

# 发送请求（自动选择提供商）
response = lb.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

支持的策略：

- `round_robin` - 轮询
- `random` - 随机
- `weighted` - 加权
- `least_connections` - 最少连接

参见 [负载均衡配置](configuration.md#负载均衡) 了解更多。
