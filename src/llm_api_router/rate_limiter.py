"""
Rate limiter module for LLM API Router.

This module provides client-side rate limiting to prevent exceeding
API rate limits and ensure fair usage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import time
from threading import Lock
import asyncio


@dataclass
class RateLimiterConfig:
    """Rate limiter configuration"""
    enabled: bool = False  # Whether rate limiting is enabled
    backend: str = "token_bucket"  # Algorithm: "token_bucket" or "sliding_window"
    requests_per_minute: int = 60  # Maximum requests per minute
    requests_per_day: Optional[int] = None  # Maximum requests per day (optional)
    tokens_per_minute: Optional[int] = None  # Token-based rate limit (optional)
    burst_size: Optional[int] = None  # Maximum burst size (for token bucket)
    wait_timeout: float = 30.0  # Maximum time to wait for rate limit (seconds)


class RateLimiterBackend(ABC):
    """Abstract base class for rate limiter backends"""
    
    @abstractmethod
    def check_and_consume(self, key: str, cost: int = 1) -> Tuple[bool, float]:
        """
        Check if request is allowed and consume quota.
        
        Args:
            key: Rate limit key (e.g., "default", "provider:openai")
            cost: Cost of this request (default: 1)
            
        Returns:
            Tuple of (allowed, wait_time_seconds)
            - allowed: True if request is allowed
            - wait_time_seconds: Time to wait before retry if not allowed
        """
        pass
    
    @abstractmethod
    def get_remaining(self, key: str) -> int:
        """Get remaining quota for the key"""
        pass
    
    @abstractmethod
    def reset(self, key: str) -> None:
        """Reset quota for the key"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        pass


class TokenBucketBackend(RateLimiterBackend):
    """
    Token Bucket rate limiter backend.
    
    This implementation allows for burst traffic while maintaining
    an average rate limit over time.
    """
    
    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,  # maximum bucket size
    ):
        """
        Initialize token bucket backend.
        
        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum bucket capacity (burst size)
        """
        self._rate = rate
        self._capacity = capacity
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = Lock()
        self._total_requests = 0
        self._rejected_requests = 0
    
    def _get_bucket(self, key: str) -> Dict[str, float]:
        """Get or create bucket for key"""
        if key not in self._buckets:
            self._buckets[key] = {
                'tokens': float(self._capacity),
                'last_update': time.time()
            }
        return self._buckets[key]
    
    def _refill(self, bucket: Dict[str, float]) -> None:
        """Refill bucket based on elapsed time"""
        now = time.time()
        elapsed = now - bucket['last_update']
        bucket['tokens'] = min(
            self._capacity,
            bucket['tokens'] + elapsed * self._rate
        )
        bucket['last_update'] = now
    
    def check_and_consume(self, key: str, cost: int = 1) -> Tuple[bool, float]:
        """Check if request is allowed and consume tokens"""
        with self._lock:
            self._total_requests += 1
            bucket = self._get_bucket(key)
            self._refill(bucket)
            
            if bucket['tokens'] >= cost:
                bucket['tokens'] -= cost
                return (True, 0.0)
            else:
                # Calculate wait time
                tokens_needed = cost - bucket['tokens']
                wait_time = tokens_needed / self._rate
                self._rejected_requests += 1
                return (False, wait_time)
    
    def get_remaining(self, key: str) -> int:
        """Get remaining tokens for the key"""
        with self._lock:
            if key not in self._buckets:
                return self._capacity
            bucket = self._get_bucket(key)
            self._refill(bucket)
            return int(bucket['tokens'])
    
    def reset(self, key: str) -> None:
        """Reset bucket for the key"""
        with self._lock:
            if key in self._buckets:
                self._buckets[key] = {
                    'tokens': float(self._capacity),
                    'last_update': time.time()
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self._lock:
            return {
                'backend': 'token_bucket',
                'rate': self._rate,
                'capacity': self._capacity,
                'buckets': len(self._buckets),
                'total_requests': self._total_requests,
                'rejected_requests': self._rejected_requests,
                'rejection_rate': (
                    self._rejected_requests / self._total_requests 
                    if self._total_requests > 0 else 0.0
                )
            }


class SlidingWindowBackend(RateLimiterBackend):
    """
    Sliding Window rate limiter backend.
    
    This implementation provides more accurate rate limiting
    by tracking request timestamps within a sliding window.
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
    ):
        """
        Initialize sliding window backend.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Window size in seconds (default: 60)
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._windows: Dict[str, list] = {}
        self._lock = Lock()
        self._total_requests = 0
        self._rejected_requests = 0
    
    def _clean_window(self, key: str) -> None:
        """Remove expired timestamps from window"""
        if key not in self._windows:
            self._windows[key] = []
            return
        
        cutoff = time.time() - self._window_seconds
        self._windows[key] = [ts for ts in self._windows[key] if ts > cutoff]
    
    def check_and_consume(self, key: str, cost: int = 1) -> Tuple[bool, float]:
        """Check if request is allowed and record timestamp"""
        with self._lock:
            self._total_requests += 1
            self._clean_window(key)
            
            current_count = len(self._windows[key])
            
            if current_count + cost <= self._max_requests:
                # Add timestamps for this request
                now = time.time()
                for _ in range(cost):
                    self._windows[key].append(now)
                return (True, 0.0)
            else:
                # Calculate wait time based on oldest request
                if self._windows[key]:
                    oldest = min(self._windows[key])
                    wait_time = oldest + self._window_seconds - time.time()
                    wait_time = max(0.0, wait_time)
                else:
                    wait_time = 0.0
                self._rejected_requests += 1
                return (False, wait_time)
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for the key"""
        with self._lock:
            self._clean_window(key)
            return max(0, self._max_requests - len(self._windows.get(key, [])))
    
    def reset(self, key: str) -> None:
        """Reset window for the key"""
        with self._lock:
            if key in self._windows:
                self._windows[key] = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self._lock:
            return {
                'backend': 'sliding_window',
                'max_requests': self._max_requests,
                'window_seconds': self._window_seconds,
                'windows': len(self._windows),
                'total_requests': self._total_requests,
                'rejected_requests': self._rejected_requests,
                'rejection_rate': (
                    self._rejected_requests / self._total_requests 
                    if self._total_requests > 0 else 0.0
                )
            }


class RateLimiter:
    """
    Rate limiter manager for LLM API requests.
    
    This class provides a simple interface for rate limiting
    and supports multiple algorithms (token bucket, sliding window).
    """
    
    def __init__(self, config: RateLimiterConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiter configuration
        """
        self.config = config
        self._backend: Optional[RateLimiterBackend] = None
        self._lock = Lock()
        
        if config.enabled:
            if config.backend == "token_bucket":
                # Convert requests per minute to rate per second
                rate = config.requests_per_minute / 60.0
                capacity = config.burst_size or config.requests_per_minute
                self._backend = TokenBucketBackend(rate=rate, capacity=capacity)
            elif config.backend == "sliding_window":
                self._backend = SlidingWindowBackend(
                    max_requests=config.requests_per_minute,
                    window_seconds=60.0
                )
            else:
                raise ValueError(f"Unsupported rate limiter backend: {config.backend}")
    
    @property
    def enabled(self) -> bool:
        """Check if rate limiting is enabled"""
        return self.config.enabled and self._backend is not None
    
    def acquire(self, key: str = "default", cost: int = 1) -> Tuple[bool, float]:
        """
        Try to acquire permission for a request.
        
        Args:
            key: Rate limit key (e.g., "default", "provider:openai")
            cost: Cost of this request
            
        Returns:
            Tuple of (allowed, wait_time_seconds)
        """
        if not self.enabled:
            return (True, 0.0)
        return self._backend.check_and_consume(key, cost)
    
    def wait_and_acquire(
        self,
        key: str = "default",
        cost: int = 1,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Wait until rate limit allows and acquire permission.
        
        Args:
            key: Rate limit key
            cost: Cost of this request
            timeout: Maximum time to wait (uses config.wait_timeout if None)
            
        Returns:
            True if acquired, False if timeout
        """
        if not self.enabled:
            return True
        
        if timeout is None:
            timeout = self.config.wait_timeout
        
        start_time = time.time()
        
        while True:
            allowed, wait_time = self.acquire(key, cost)
            if allowed:
                return True
            
            elapsed = time.time() - start_time
            if elapsed + wait_time > timeout:
                return False
            
            # Sleep for the wait time (with a small buffer)
            time.sleep(min(wait_time + 0.01, timeout - elapsed))
    
    async def wait_and_acquire_async(
        self,
        key: str = "default",
        cost: int = 1,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Async version of wait_and_acquire.
        
        Args:
            key: Rate limit key
            cost: Cost of this request
            timeout: Maximum time to wait
            
        Returns:
            True if acquired, False if timeout
        """
        if not self.enabled:
            return True
        
        if timeout is None:
            timeout = self.config.wait_timeout
        
        start_time = time.time()
        
        while True:
            allowed, wait_time = self.acquire(key, cost)
            if allowed:
                return True
            
            elapsed = time.time() - start_time
            if elapsed + wait_time > timeout:
                return False
            
            # Async sleep for the wait time
            await asyncio.sleep(min(wait_time + 0.01, timeout - elapsed))
    
    def get_remaining(self, key: str = "default") -> int:
        """Get remaining requests for the key"""
        if not self.enabled:
            return -1  # Unlimited
        return self._backend.get_remaining(key)
    
    def reset(self, key: str = "default") -> None:
        """Reset rate limit for the key"""
        if self.enabled:
            self._backend.reset(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        stats = self._backend.get_stats()
        stats['enabled'] = True
        stats['requests_per_minute'] = self.config.requests_per_minute
        stats['wait_timeout'] = self.config.wait_timeout
        return stats


# Context manager for rate limiting
class RateLimitContext:
    """
    Context manager for rate limiting.
    
    Usage:
        rate_limiter = RateLimiter(config)
        async with RateLimitContext(rate_limiter, key="openai") as acquired:
            if acquired:
                # Make API call
                pass
            else:
                # Handle rate limit exceeded
                pass
    """
    
    def __init__(
        self,
        rate_limiter: RateLimiter,
        key: str = "default",
        cost: int = 1,
        timeout: Optional[float] = None,
        wait: bool = True
    ):
        """
        Initialize rate limit context.
        
        Args:
            rate_limiter: The rate limiter instance
            key: Rate limit key
            cost: Cost of this request
            timeout: Maximum timeout for waiting
            wait: Whether to wait for rate limit or fail immediately
        """
        self._rate_limiter = rate_limiter
        self._key = key
        self._cost = cost
        self._timeout = timeout
        self._wait = wait
        self._acquired = False
    
    def __enter__(self) -> bool:
        if self._wait:
            self._acquired = self._rate_limiter.wait_and_acquire(
                self._key, self._cost, self._timeout
            )
        else:
            self._acquired, _ = self._rate_limiter.acquire(self._key, self._cost)
        return self._acquired
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def __aenter__(self) -> bool:
        if self._wait:
            self._acquired = await self._rate_limiter.wait_and_acquire_async(
                self._key, self._cost, self._timeout
            )
        else:
            self._acquired, _ = self._rate_limiter.acquire(self._key, self._cost)
        return self._acquired
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
