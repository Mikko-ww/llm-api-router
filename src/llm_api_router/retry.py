"""Retry utilities for handling transient failures."""
import time
import asyncio
import random
from functools import wraps
from typing import Callable, TypeVar, Any
import httpx
from .types import RetryConfig
from .exceptions import MaxRetriesExceededError, RequestTimeoutError

T = TypeVar('T')


def with_retry(retry_config: RetryConfig):
    """Decorator to add retry logic to synchronous functions.
    
    Args:
        retry_config: RetryConfig instance with retry settings
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    last_exception = e
                    status_code = e.response.status_code
                    
                    # Don't retry on non-retryable status codes
                    if status_code not in retry_config.retry_on_status_codes:
                        raise
                    
                    # If this was the last attempt, raise
                    if attempt >= retry_config.max_retries:
                        raise MaxRetriesExceededError(
                            f"Max retries ({retry_config.max_retries}) exceeded",
                            status_code=status_code,
                            details={"attempts": attempt + 1}
                        ) from e
                    
                    # Calculate delay and wait
                    delay = retry_config.calculate_delay(attempt)
                    time.sleep(delay)
                    
                except httpx.TimeoutException as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise
                    if attempt >= retry_config.max_retries:
                        raise RequestTimeoutError(
                            f"Request timeout after {retry_config.max_retries} retries",
                            details={"attempts": attempt + 1}
                        ) from e
                    
                    # Calculate delay and wait
                    delay = retry_config.calculate_delay(attempt)
                    time.sleep(delay)
                    
                except httpx.RequestError as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise
                    if attempt >= retry_config.max_retries:
                        raise MaxRetriesExceededError(
                            f"Max retries ({retry_config.max_retries}) exceeded: {str(e)}",
                            details={"attempts": attempt + 1}
                        ) from e
                    
                    # Calculate delay and wait
                    delay = retry_config.calculate_delay(attempt)
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            
            raise RuntimeError("Unexpected retry logic error")
        
        return wrapper
    return decorator


def with_retry_async(retry_config: RetryConfig):
    """Decorator to add retry logic to asynchronous functions.
    
    Args:
        retry_config: RetryConfig instance with retry settings
    
    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    last_exception = e
                    status_code = e.response.status_code
                    
                    # Don't retry on non-retryable status codes
                    if status_code not in retry_config.retry_on_status_codes:
                        raise
                    
                    # If this was the last attempt, raise
                    if attempt >= retry_config.max_retries:
                        raise MaxRetriesExceededError(
                            f"Max retries ({retry_config.max_retries}) exceeded",
                            status_code=status_code,
                            details={"attempts": attempt + 1}
                        ) from e
                    
                    # Calculate delay and wait
                    delay = retry_config.calculate_delay(attempt)
                    await asyncio.sleep(delay)
                    
                except httpx.TimeoutException as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise
                    if attempt >= retry_config.max_retries:
                        raise RequestTimeoutError(
                            f"Request timeout after {retry_config.max_retries} retries",
                            details={"attempts": attempt + 1}
                        ) from e
                    
                    # Calculate delay and wait
                    delay = retry_config.calculate_delay(attempt)
                    await asyncio.sleep(delay)
                    
                except httpx.RequestError as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise
                    if attempt >= retry_config.max_retries:
                        raise MaxRetriesExceededError(
                            f"Max retries ({retry_config.max_retries}) exceeded: {str(e)}",
                            details={"attempts": attempt + 1}
                        ) from e
                    
                    # Calculate delay and wait
                    delay = retry_config.calculate_delay(attempt)
                    await asyncio.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            
            raise RuntimeError("Unexpected retry logic error")
        
        return wrapper
    return decorator
