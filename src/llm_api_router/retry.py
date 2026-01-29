"""重试逻辑和装饰器实现"""
import time
import asyncio
from functools import wraps
from typing import Callable, Any, TypeVar, Optional
import httpx
from .types import RetryConfig
from .exceptions import (
    RetryExhaustedError,
    RateLimitError,
    ProviderError,
    TimeoutError,
    NetworkError,
)

T = TypeVar('T')


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """
    计算指数退避延迟时间
    
    Args:
        attempt: 当前重试次数 (从0开始)
        config: 重试配置
        
    Returns:
        延迟时间（秒）
    """
    delay = config.initial_delay * (config.exponential_base ** attempt)
    return min(delay, config.max_delay)


def should_retry(exception: Exception, status_code: Optional[int], config: RetryConfig) -> bool:
    """
    判断是否应该重试
    
    Args:
        exception: 捕获的异常
        status_code: HTTP状态码（如果有）
        config: 重试配置
        
    Returns:
        是否应该重试
    """
    # 对于特定的HTTP状态码，总是重试
    if status_code and status_code in config.retry_on_status_codes:
        return True
    
    # 网络错误或超时错误，总是重试
    if isinstance(exception, (NetworkError, TimeoutError, httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    
    # 速率限制错误，重试
    if isinstance(exception, RateLimitError):
        return True
    
    # 服务器错误，重试
    if isinstance(exception, ProviderError) and status_code and status_code >= 500:
        return True
    
    return False


def with_retry(config: Optional[RetryConfig] = None):
    """
    同步重试装饰器
    
    Args:
        config: 重试配置，如果为None则使用默认配置
    """
    retry_config = config or RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            last_status_code = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 提取状态码（如果异常包含）
                    status_code = getattr(e, 'status_code', None)
                    last_status_code = status_code
                    
                    # 最后一次尝试失败，不再重试
                    if attempt >= retry_config.max_retries:
                        break
                    
                    # 判断是否应该重试
                    if not should_retry(e, status_code, retry_config):
                        raise
                    
                    # 计算延迟时间
                    delay = calculate_backoff_delay(attempt, retry_config)
                    
                    # 等待后重试
                    time.sleep(delay)
            
            # 所有重试都失败
            raise RetryExhaustedError(
                f"请求失败，已重试 {retry_config.max_retries} 次: {str(last_exception)}",
                status_code=last_status_code,
                details={
                    "max_retries": retry_config.max_retries,
                    "original_error": str(last_exception),
                    "error_type": type(last_exception).__name__
                }
            )
        
        return wrapper
    return decorator


def with_retry_async(config: Optional[RetryConfig] = None):
    """
    异步重试装饰器
    
    Args:
        config: 重试配置，如果为None则使用默认配置
    """
    retry_config = config or RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            last_status_code = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 提取状态码（如果异常包含）
                    status_code = getattr(e, 'status_code', None)
                    last_status_code = status_code
                    
                    # 最后一次尝试失败，不再重试
                    if attempt >= retry_config.max_retries:
                        break
                    
                    # 判断是否应该重试
                    if not should_retry(e, status_code, retry_config):
                        raise
                    
                    # 计算延迟时间
                    delay = calculate_backoff_delay(attempt, retry_config)
                    
                    # 等待后重试
                    await asyncio.sleep(delay)
            
            # 所有重试都失败
            raise RetryExhaustedError(
                f"请求失败，已重试 {retry_config.max_retries} 次: {str(last_exception)}",
                status_code=last_status_code,
                details={
                    "max_retries": retry_config.max_retries,
                    "original_error": str(last_exception),
                    "error_type": type(last_exception).__name__
                }
            )
        
        return wrapper
    return decorator
