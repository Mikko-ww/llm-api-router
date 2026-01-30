# 缓存配置

LLM API Router 提供内置的缓存支持，帮助减少重复请求和降低成本。

## 快速开始

```python
from llm_api_router import Client, ProviderConfig
from llm_api_router.cache import MemoryCache

# 创建内存缓存
cache = MemoryCache(max_size=1000, ttl=3600)

config = ProviderConfig(
    provider_type="openai",
    api_key="sk-xxx",
    default_model="gpt-3.5-turbo"
)

with Client(config, cache=cache) as client:
    # 第一次请求：调用 API
    response1 = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # 第二次相同请求：使用缓存
    response2 = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

## 缓存类型

### MemoryCache

内存缓存适用于单进程应用：

```python
from llm_api_router.cache import MemoryCache

cache = MemoryCache(
    max_size=1000,     # 最大缓存条目数
    ttl=3600,          # 生存时间（秒），超时后自动失效
)
```

**特点：**

- 速度快，无 I/O 开销
- 进程重启后缓存丢失
- 适合开发和测试环境

### DiskCache

磁盘缓存适用于需要持久化的场景：

```python
from llm_api_router.cache import DiskCache

cache = DiskCache(
    cache_dir=".llm_cache",  # 缓存目录
    max_size_mb=500,         # 最大缓存大小（MB）
    ttl=86400,               # 生存时间（秒）
)
```

**特点：**

- 缓存持久化到磁盘
- 进程重启后缓存保留
- 适合生产环境

## 缓存键生成

缓存键基于以下因素自动生成：

- Provider 类型
- 模型名称
- 消息内容
- Temperature 等参数

相同的请求参数会生成相同的缓存键。

## 禁用缓存

对于特定请求，可以通过参数禁用缓存：

```python
# 使用 bypass_cache 参数（如果支持）
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    # 某些实现可能支持此参数
)
```

## 清除缓存

```python
# 清除所有缓存
cache.clear()

# 获取缓存统计
stats = cache.stats()
print(f"命中次数: {stats['hits']}")
print(f"未命中次数: {stats['misses']}")
print(f"命中率: {stats['hit_rate']:.2%}")
```

## 最佳实践

1. **开发环境**：使用 `MemoryCache`，快速迭代
2. **生产环境**：使用 `DiskCache`，避免重复 API 调用
3. **设置合理 TTL**：根据数据时效性调整
4. **监控命中率**：过低的命中率可能需要调整策略

## 缓存与流式响应

> ⚠️ 流式响应默认不会被缓存，因为流式输出是增量的。

如需缓存流式响应的完整内容，请在应用层自行实现。
