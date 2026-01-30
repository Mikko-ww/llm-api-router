# Rate Limiter

LLM API Router 提供了客户端速率限制功能，帮助你避免超过 API 服务商的速率限制。

## 配置

```python
from llm_api_router.rate_limiter import RateLimiterConfig, RateLimiter

config = RateLimiterConfig(
    enabled=True,                    # 启用速率限制
    backend="token_bucket",          # 算法: "token_bucket" 或 "sliding_window"
    requests_per_minute=60,          # 每分钟最大请求数
    burst_size=10,                   # 最大突发请求数（仅 token_bucket）
    wait_timeout=30.0,               # 等待超时时间（秒）
)

limiter = RateLimiter(config)
```

## 算法选择

### Token Bucket（令牌桶）

默认算法。允许短时间内的突发流量，同时保持长期平均速率。

**优点：**
- 允许突发流量
- 平滑的速率限制
- 适合 API 调用场景

**配置示例：**
```python
config = RateLimiterConfig(
    enabled=True,
    backend="token_bucket",
    requests_per_minute=60,  # 平均每秒 1 个请求
    burst_size=10,           # 允许最多 10 个突发请求
)
```

### Sliding Window（滑动窗口）

更严格的速率限制，在固定时间窗口内精确计数请求。

**优点：**
- 精确的请求计数
- 更可预测的行为
- 适合严格限制场景

**配置示例：**
```python
config = RateLimiterConfig(
    enabled=True,
    backend="sliding_window",
    requests_per_minute=60,
)
```

## 基本用法

### 立即获取

```python
allowed, wait_time = limiter.acquire()
if allowed:
    # 执行 API 调用
    pass
else:
    print(f"需要等待 {wait_time:.2f} 秒")
```

### 等待获取

```python
# 阻塞直到获取许可（或超时）
acquired = limiter.wait_and_acquire(timeout=5.0)
if acquired:
    # 执行 API 调用
    pass
else:
    print("超时")
```

### 异步等待

```python
acquired = await limiter.wait_and_acquire_async(timeout=5.0)
if acquired:
    # 执行 API 调用
    pass
```

## 上下文管理器

```python
from llm_api_router.rate_limiter import RateLimitContext

# 同步
with RateLimitContext(limiter, key="openai") as acquired:
    if acquired:
        # 执行 API 调用
        pass

# 异步
async with RateLimitContext(limiter, key="openai") as acquired:
    if acquired:
        # 执行 API 调用
        pass
```

## 多 Provider 限制

为不同 Provider 设置独立的速率限制：

```python
# 使用不同的 key 来分隔不同 Provider 的限制
limiter.acquire(key="provider:openai")
limiter.acquire(key="provider:anthropic")
limiter.acquire(key="provider:gemini")
```

## 获取统计信息

```python
stats = limiter.get_stats()
print(f"总请求数: {stats['total_requests']}")
print(f"被拒绝数: {stats['rejected_requests']}")
print(f"拒绝率: {stats['rejection_rate']:.1%}")
```

## 辅助方法

```python
# 获取剩余请求数
remaining = limiter.get_remaining(key="default")

# 重置速率限制
limiter.reset(key="default")
```

## 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | False | 是否启用速率限制 |
| `backend` | str | "token_bucket" | 算法类型 |
| `requests_per_minute` | int | 60 | 每分钟最大请求数 |
| `requests_per_day` | int | None | 每天最大请求数（可选） |
| `tokens_per_minute` | int | None | 每分钟最大 token 数（可选） |
| `burst_size` | int | None | 最大突发请求数 |
| `wait_timeout` | float | 30.0 | 等待超时时间（秒） |

## 最佳实践

1. **选择合适的算法**
   - API 调用场景推荐 Token Bucket
   - 需要严格限制时使用 Sliding Window

2. **设置合理的超时**
   - 避免过长的等待时间阻塞应用
   - 根据业务需求设置 `wait_timeout`

3. **使用 Provider 分离**
   - 为每个 Provider 使用独立的 key
   - 可以根据不同 Provider 的限制调整配置

4. **监控统计数据**
   - 定期检查 `get_stats()` 了解使用情况
   - 根据拒绝率调整配置
