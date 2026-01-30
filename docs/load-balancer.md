# 负载均衡器 (Load Balancer)

负载均衡器模块提供多端点管理、负载分发和自动故障转移功能，确保高可用性和最优资源利用。

## 功能特性

- **多种选择策略**: 轮询、加权、最低延迟、随机、故障转移
- **健康检查**: 自动标记不健康端点并在恢复后重新启用
- **统计追踪**: 请求计数、成功率、延迟监控
- **线程安全**: 支持并发操作
- **动态管理**: 运行时添加/移除端点

## 快速开始

### 基础用法

```python
from llm_api_router import LoadBalancer, Endpoint

# 创建端点
endpoints = [
    Endpoint(name="primary", provider="openai", weight=3),
    Endpoint(name="secondary", provider="anthropic", weight=1),
]

# 创建负载均衡器
lb = LoadBalancer(
    endpoints=endpoints,
    strategy="weighted",  # 可选: round_robin, weighted, least_latency, random, failover
)

# 获取端点并发起请求
endpoint = lb.get_endpoint()
try:
    # 发起实际请求...
    response = make_request(endpoint)
    lb.mark_success(endpoint, latency=response.elapsed)
except Exception as e:
    lb.mark_failure(endpoint)
```

### 故障转移模式

```python
from llm_api_router import LoadBalancer, Endpoint, LoadBalancerConfig

# 优先级越低，优先级越高
endpoints = [
    Endpoint(name="primary", provider="openai", priority=0),
    Endpoint(name="backup1", provider="anthropic", priority=1),
    Endpoint(name="backup2", provider="gemini", priority=2),
]

config = LoadBalancerConfig(
    failure_threshold=3,    # 连续3次失败后标记为不健康
    recovery_time=60.0,     # 60秒后重试不健康端点
)

lb = LoadBalancer(
    endpoints=endpoints,
    strategy="failover",
    config=config,
)
```

## 选择策略

### Round Robin (轮询)
按顺序循环选择端点。

```python
lb = LoadBalancer(endpoints=endpoints, strategy="round_robin")
# 依次返回: a, b, c, a, b, c, ...
```

### Weighted (加权)
根据权重随机选择，权重越高被选中概率越大。

```python
endpoints = [
    Endpoint(name="high", weight=10),  # 被选中概率约 10/11
    Endpoint(name="low", weight=1),    # 被选中概率约 1/11
]
lb = LoadBalancer(endpoints=endpoints, strategy="weighted")
```

### Least Latency (最低延迟)
选择平均响应延迟最低的端点。

```python
lb = LoadBalancer(endpoints=endpoints, strategy="least_latency")

# 记录延迟以启用策略
lb.mark_success(endpoint, latency=0.5)  # 记录延迟
```

### Random (随机)
完全随机选择。

```python
lb = LoadBalancer(endpoints=endpoints, strategy="random")
```

### Failover (故障转移)
按优先级选择，总是选择优先级最高（priority 最低）的健康端点。

```python
endpoints = [
    Endpoint(name="primary", priority=0),   # 首选
    Endpoint(name="secondary", priority=1), # 备用
]
lb = LoadBalancer(endpoints=endpoints, strategy="failover")
```

## 健康管理

### 标记成功/失败

```python
endpoint = lb.get_endpoint()

try:
    result = call_api(endpoint)
    lb.mark_success(endpoint, latency=result.elapsed)
except Exception:
    lb.mark_failure(endpoint)
```

### 端点状态

端点有三种状态：
- **HEALTHY**: 正常，可用
- **DEGRADED**: 有失败记录但未达阈值，仍可用
- **UNHEALTHY**: 连续失败达到阈值，暂时排除

### 自动恢复

不健康的端点在 `recovery_time` 后会被重新尝试：

```python
config = LoadBalancerConfig(
    failure_threshold=3,   # 3次连续失败后不健康
    recovery_time=30.0,    # 30秒后重试
)
```

## 统计信息

### 获取端点统计

```python
# 获取单个端点统计
stats = lb.get_stats("primary")
print(f"成功率: {stats['success_rate']:.2%}")
print(f"平均延迟: {stats['avg_latency']:.3f}s")

# 获取所有端点统计
all_stats = lb.get_stats()
for name, stats in all_stats.items():
    print(f"{name}: {stats['total_requests']} requests, {stats['status']}")
```

### 统计字段

| 字段 | 描述 |
|------|------|
| status | 端点状态 (healthy/degraded/unhealthy) |
| total_requests | 总请求数 |
| successful_requests | 成功请求数 |
| failed_requests | 失败请求数 |
| success_rate | 成功率 |
| consecutive_failures | 当前连续失败次数 |
| avg_latency | 平均延迟（秒） |

## 动态端点管理

### 添加端点

```python
new_endpoint = Endpoint(name="new", provider="openai")
lb.add_endpoint(new_endpoint)
```

### 移除端点

```python
lb.remove_endpoint("old")
```

### 重置统计

```python
lb.reset_stats("primary")  # 重置单个
lb.reset_stats()           # 重置所有
```

## 完整示例

```python
from llm_api_router import (
    LoadBalancer,
    Endpoint,
    LoadBalancerConfig,
    create_load_balancer,
)

# 方式1: 使用类
endpoints = [
    Endpoint(
        name="openai-1",
        provider="openai",
        url="https://api.openai.com/v1",
        weight=2,
    ),
    Endpoint(
        name="openai-2",
        provider="openai",
        url="https://api-backup.openai.com/v1",
        weight=1,
    ),
]

config = LoadBalancerConfig(
    failure_threshold=3,
    recovery_time=60.0,
    latency_window=20,
)

lb = LoadBalancer(
    endpoints=endpoints,
    strategy="weighted",
    config=config,
)

# 方式2: 使用便捷函数
lb = create_load_balancer(
    endpoints=[
        {"name": "primary", "provider": "openai", "weight": 3},
        {"name": "backup", "provider": "anthropic", "weight": 1},
    ],
    strategy="weighted",
    failure_threshold=3,
    recovery_time=60.0,
)

# 使用负载均衡器
def call_with_failover():
    tried = []
    
    while True:
        endpoint = lb.get_endpoint(exclude=tried)
        if not endpoint:
            raise Exception("All endpoints failed")
        
        tried.append(endpoint.name)
        
        try:
            result = call_api(endpoint)
            lb.mark_success(endpoint, latency=result.elapsed)
            return result
        except Exception:
            lb.mark_failure(endpoint)
            continue
```

## API 参考

### Endpoint

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| name | str | 必填 | 端点唯一标识 |
| url | str | "" | 基础 URL |
| provider | str | "" | 提供商名称 |
| api_key | str | "" | API 密钥 |
| weight | int | 1 | 权重（用于加权策略） |
| priority | int | 0 | 优先级（用于故障转移） |
| metadata | dict | {} | 额外元数据 |

### LoadBalancerConfig

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| failure_threshold | int | 3 | 不健康阈值 |
| recovery_time | float | 30.0 | 恢复等待时间（秒） |
| latency_window | int | 10 | 延迟样本窗口大小 |
| degraded_threshold | int | 1 | 降级阈值 |

### LoadBalancer 方法

| 方法 | 描述 |
|------|------|
| get_endpoint(exclude=None) | 获取下一个端点 |
| mark_success(endpoint, latency=None) | 标记请求成功 |
| mark_failure(endpoint) | 标记请求失败 |
| get_stats(endpoint_name=None) | 获取统计信息 |
| get_healthy_endpoints() | 获取健康端点列表 |
| add_endpoint(endpoint) | 添加新端点 |
| remove_endpoint(name) | 移除端点 |
| reset_stats(name=None) | 重置统计 |
| set_strategy(strategy) | 切换策略 |
