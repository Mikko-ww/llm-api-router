# 异常参考

LLM API Router 定义了一套清晰的异常层次结构，方便错误处理。

## 异常层次结构

```
LLMRouterError (基类)
├── AuthenticationError    # 认证失败
├── RateLimitError         # 速率限制
├── InvalidRequestError    # 无效请求
├── APIError              # API 错误
├── TimeoutError          # 请求超时
├── ProviderError         # 提供商错误
└── ConfigurationError    # 配置错误
```

## 基类

### LLMRouterError

所有异常的基类：

```python
class LLMRouterError(Exception):
    """LLM API Router 基础异常"""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)
```

## 具体异常

### AuthenticationError

API 密钥无效或过期：

```python
class AuthenticationError(LLMRouterError):
    """认证失败异常"""
    pass
```

**常见原因：**

- API Key 无效
- API Key 已过期
- API Key 权限不足

**处理示例：**

```python
from llm_api_router.exceptions import AuthenticationError

try:
    response = client.chat.completions.create(...)
except AuthenticationError as e:
    print(f"认证失败: {e.message}")
    print(f"提供商: {e.provider}")
    # 提示用户检查 API Key
```

### RateLimitError

请求超过速率限制：

```python
class RateLimitError(LLMRouterError):
    """速率限制异常"""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after  # 建议等待时间（秒）
```

**处理示例：**

```python
from llm_api_router.exceptions import RateLimitError
import time

try:
    response = client.chat.completions.create(...)
except RateLimitError as e:
    wait_time = e.retry_after or 60
    print(f"达到速率限制，{wait_time} 秒后重试")
    time.sleep(wait_time)
    # 重试请求
```

### InvalidRequestError

请求参数无效：

```python
class InvalidRequestError(LLMRouterError):
    """无效请求异常"""
    pass
```

**常见原因：**

- 模型名称错误
- 参数格式错误
- 消息内容为空

### APIError

API 调用失败：

```python
class APIError(LLMRouterError):
    """API 错误异常"""
    pass
```

**常见原因：**

- 服务暂时不可用
- 提供商服务器错误
- 网络问题

### TimeoutError

请求超时：

```python
class TimeoutError(LLMRouterError):
    """超时异常"""
    pass
```

**处理示例：**

```python
from llm_api_router.exceptions import TimeoutError

try:
    response = client.chat.completions.create(...)
except TimeoutError as e:
    print(f"请求超时: {e.message}")
    # 可以增加 timeout 参数重试
```

### ProviderError

提供商特定错误：

```python
class ProviderError(LLMRouterError):
    """提供商错误异常"""
    pass
```

### ConfigurationError

配置错误：

```python
class ConfigurationError(LLMRouterError):
    """配置错误异常"""
    pass
```

**常见原因：**

- 缺少必需的配置项
- 配置值类型错误
- 不支持的提供商类型

## 最佳实践

### 捕获特定异常

```python
from llm_api_router.exceptions import (
    LLMRouterError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
)

try:
    response = client.chat.completions.create(...)
    
except AuthenticationError as e:
    # 处理认证错误
    log.error(f"API Key 无效: {e}")
    raise
    
except RateLimitError as e:
    # 处理速率限制
    time.sleep(e.retry_after or 60)
    # 重试...
    
except TimeoutError as e:
    # 处理超时
    log.warning(f"请求超时: {e}")
    # 重试或降级...
    
except LLMRouterError as e:
    # 处理其他 LLM Router 错误
    log.error(f"LLM 调用失败: {e}")
    raise
```

### 与重试机制结合

```python
from llm_api_router.retry import RetryConfig

# 配置自动重试
config = ProviderConfig(
    provider_type="openai",
    api_key="sk-xxx",
    max_retries=3,  # 自动重试 3 次
)
```

### 日志记录

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    response = client.chat.completions.create(...)
except LLMRouterError as e:
    logger.exception(
        "LLM 调用失败",
        extra={
            "provider": e.provider,
            "status_code": e.status_code,
        }
    )
```
