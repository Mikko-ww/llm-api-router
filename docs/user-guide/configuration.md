# 配置参考

本文档详细介绍 LLM API Router 的所有配置选项。

## ProviderConfig

`ProviderConfig` 是配置 LLM 提供商的核心类。

```python
from llm_api_router import ProviderConfig

config = ProviderConfig(
    provider_type: str,           # 必填
    api_key: str,                 # 必填
    default_model: str = None,    # 可选
    base_url: str = None,         # 可选
    timeout: float = 30.0,        # 可选
    max_retries: int = 3,         # 可选
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `provider_type` | `str` | - | 提供商类型，如 `"openai"`, `"anthropic"` |
| `api_key` | `str` | - | API 密钥 |
| `default_model` | `str` | `None` | 默认使用的模型 |
| `base_url` | `str` | `None` | 自定义 API 地址 |
| `timeout` | `float` | `30.0` | 请求超时时间（秒） |
| `max_retries` | `int` | `3` | 最大重试次数 |

## Client 配置

### 同步客户端

```python
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(...)

# 使用上下文管理器（推荐）
with Client(config) as client:
    response = client.chat.completions.create(...)

# 手动管理
client = Client(config)
try:
    response = client.chat.completions.create(...)
finally:
    client.close()
```

### 异步客户端

```python
from llm_api_router import AsyncClient, ProviderConfig

config = ProviderConfig(...)

# 使用上下文管理器（推荐）
async with AsyncClient(config) as client:
    response = await client.chat.completions.create(...)
```

## 重试配置

内置指数退避重试机制：

```python
from llm_api_router import Client, ProviderConfig, RetryConfig

# 自定义重试配置
retry_config = RetryConfig(
    max_retries=5,              # 最大重试次数
    initial_delay=1.0,          # 初始延迟（秒）
    max_delay=60.0,             # 最大延迟（秒）
    exponential_base=2.0,       # 指数基数
    jitter=0.1,                 # 抖动因子
    retryable_errors=[          # 可重试的错误类型
        "rate_limit_error",
        "timeout_error", 
        "server_error",
    ],
)

config = ProviderConfig(
    provider_type="openai",
    api_key="sk-xxx",
    default_model="gpt-4o",
    max_retries=5,
)
```

## 缓存配置

使用缓存减少重复请求：

```python
from llm_api_router import Client, ProviderConfig
from llm_api_router.cache import Cache, MemoryCache, DiskCache

# 内存缓存
cache = MemoryCache(
    max_size=1000,           # 最大缓存条目数
    ttl=3600,                # 生存时间（秒）
)

# 磁盘缓存
cache = DiskCache(
    cache_dir=".llm_cache",  # 缓存目录
    max_size_mb=500,         # 最大缓存大小（MB）
    ttl=86400,               # 生存时间（秒）
)

# 使用缓存
config = ProviderConfig(...)
with Client(config, cache=cache) as client:
    # 相同请求会使用缓存
    response = client.chat.completions.create(...)
```

## 速率限制配置

控制请求速率：

```python
from llm_api_router import RateLimiter

# 创建速率限制器
limiter = RateLimiter(
    requests_per_minute=60,      # 每分钟请求数
    tokens_per_minute=100000,    # 每分钟 token 数
    max_concurrent=10,           # 最大并发数
)

# 使用速率限制器
async with limiter.acquire():
    response = await client.chat.completions.create(...)
```

## 负载均衡配置

多提供商负载均衡：

```python
from llm_api_router import LoadBalancer, ProviderConfig

configs = [
    ProviderConfig(provider_type="openai", api_key="sk-1", default_model="gpt-4o"),
    ProviderConfig(provider_type="openai", api_key="sk-2", default_model="gpt-4o"),
]

# 轮询策略
lb = LoadBalancer(configs, strategy="round_robin")

# 加权策略
lb = LoadBalancer(
    configs, 
    strategy="weighted",
    weights=[0.7, 0.3],  # 第一个处理70%请求
)

# 最少连接策略
lb = LoadBalancer(configs, strategy="least_connections")

# 随机策略
lb = LoadBalancer(configs, strategy="random")
```

### 负载均衡策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `round_robin` | 轮询每个提供商 | 均匀分配负载 |
| `weighted` | 按权重分配 | 主备切换 |
| `least_connections` | 选择最少连接的 | 动态负载 |
| `random` | 随机选择 | 简单场景 |

## Prompt 模板配置

```python
from llm_api_router import PromptTemplate

# 创建模板
template = PromptTemplate(
    name="qa_template",
    template="请根据以下上下文回答问题：\n\n上下文：{context}\n\n问题：{question}",
    variables=["context", "question"],
    default_values={"context": "无上下文"},
)

# 渲染模板
prompt = template.render(
    context="Python 是一种编程语言",
    question="Python 是什么？"
)
```

## 会话管理配置

```python
from llm_api_router import ConversationManager

# 创建会话管理器
manager = ConversationManager(
    max_history=50,           # 最大历史消息数
    max_tokens=4000,          # 最大 token 数
    system_message="你是一个有用的助手",  # 系统消息
)

# 添加消息
manager.add_user_message("你好")
manager.add_assistant_message("你好！有什么我可以帮助你的？")

# 获取完整消息列表
messages = manager.get_messages()
```

## 日志配置

```python
from llm_api_router import configure_logging

# 配置日志
configure_logging(
    level="INFO",              # 日志级别
    format="json",             # 输出格式：json 或 text
    output="stderr",           # 输出位置
    include_request=True,      # 是否包含请求详情
    include_response=True,     # 是否包含响应详情
)
```

### 日志级别

- `DEBUG` - 详细调试信息
- `INFO` - 一般信息
- `WARNING` - 警告信息
- `ERROR` - 错误信息
- `CRITICAL` - 严重错误

## 指标配置

```python
from llm_api_router import MetricsCollector

# 创建指标收集器
metrics = MetricsCollector()

# 获取指标
stats = metrics.get_stats()
print(f"总请求数: {stats['total_requests']}")
print(f"平均延迟: {stats['avg_latency_ms']}ms")
print(f"错误率: {stats['error_rate']}%")

# 导出 Prometheus 格式
prometheus_metrics = metrics.export_prometheus()
```

## 环境变量

支持通过环境变量配置：

```bash
# API Keys
export OPENAI_API_KEY="sk-xxx"
export ANTHROPIC_API_KEY="sk-ant-xxx"
export GEMINI_API_KEY="AIza-xxx"

# 可选配置
export LLM_ROUTER_TIMEOUT="60"
export LLM_ROUTER_MAX_RETRIES="5"
export LLM_ROUTER_LOG_LEVEL="DEBUG"
```

```python
import os
from llm_api_router import ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    timeout=float(os.environ.get("LLM_ROUTER_TIMEOUT", 30)),
)
```

## 配置文件

支持 YAML 配置文件：

```yaml
# config.yaml
providers:
  openai:
    provider_type: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4o
    timeout: 30
    max_retries: 3
    
  anthropic:
    provider_type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-3-5-sonnet-20241022

cache:
  type: disk
  cache_dir: .llm_cache
  max_size_mb: 500
  ttl: 86400

logging:
  level: INFO
  format: json
```

使用 CLI 验证配置：

```bash
llm-router validate config.yaml
```
