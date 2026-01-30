"""
Unit tests for rate_limiter module.
"""

import pytest
import time
from unittest.mock import patch
from llm_api_router.rate_limiter import (
    RateLimiterConfig,
    TokenBucketBackend,
    SlidingWindowBackend,
    RateLimiter,
    RateLimitContext,
)


class TestRateLimiterConfig:
    """Tests for RateLimiterConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = RateLimiterConfig()
        
        assert config.enabled is False
        assert config.backend == "token_bucket"
        assert config.requests_per_minute == 60
        assert config.requests_per_day is None
        assert config.tokens_per_minute is None
        assert config.burst_size is None
        assert config.wait_timeout == 30.0
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = RateLimiterConfig(
            enabled=True,
            backend="sliding_window",
            requests_per_minute=100,
            requests_per_day=10000,
            tokens_per_minute=50000,
            burst_size=20,
            wait_timeout=60.0
        )
        
        assert config.enabled is True
        assert config.backend == "sliding_window"
        assert config.requests_per_minute == 100
        assert config.requests_per_day == 10000
        assert config.tokens_per_minute == 50000
        assert config.burst_size == 20
        assert config.wait_timeout == 60.0


class TestTokenBucketBackend:
    """Tests for TokenBucketBackend"""
    
    def test_initialization(self):
        """Test backend initialization"""
        backend = TokenBucketBackend(rate=1.0, capacity=10)
        
        stats = backend.get_stats()
        assert stats['backend'] == 'token_bucket'
        assert stats['rate'] == 1.0
        assert stats['capacity'] == 10
    
    def test_allow_requests_within_capacity(self):
        """Test requests within capacity are allowed"""
        backend = TokenBucketBackend(rate=1.0, capacity=5)
        
        for i in range(5):
            allowed, wait_time = backend.check_and_consume("test")
            assert allowed is True, f"Request {i+1} should be allowed"
            assert wait_time == 0.0
    
    def test_reject_requests_exceeding_capacity(self):
        """Test requests exceeding capacity are rejected"""
        backend = TokenBucketBackend(rate=1.0, capacity=3)
        
        # Consume all tokens
        for _ in range(3):
            backend.check_and_consume("test")
        
        # Next request should be rejected
        allowed, wait_time = backend.check_and_consume("test")
        assert allowed is False
        assert wait_time > 0
    
    def test_token_refill(self):
        """Test tokens are refilled over time"""
        backend = TokenBucketBackend(rate=10.0, capacity=5)  # 10 tokens/second
        
        # Consume all tokens
        for _ in range(5):
            backend.check_and_consume("test")
        
        # Wait for refill
        time.sleep(0.2)  # Should add ~2 tokens
        
        remaining = backend.get_remaining("test")
        assert remaining >= 1  # At least 1 token should be available
    
    def test_cost_parameter(self):
        """Test cost parameter consumes multiple tokens"""
        backend = TokenBucketBackend(rate=1.0, capacity=10)
        
        allowed, _ = backend.check_and_consume("test", cost=5)
        assert allowed is True
        
        remaining = backend.get_remaining("test")
        assert remaining == 5
    
    def test_separate_keys(self):
        """Test different keys have separate buckets"""
        backend = TokenBucketBackend(rate=1.0, capacity=3)
        
        # Consume all tokens for key1
        for _ in range(3):
            backend.check_and_consume("key1")
        
        # key2 should still have tokens
        allowed, _ = backend.check_and_consume("key2")
        assert allowed is True
    
    def test_reset(self):
        """Test reset refills the bucket"""
        backend = TokenBucketBackend(rate=1.0, capacity=5)
        
        # Consume all tokens
        for _ in range(5):
            backend.check_and_consume("test")
        
        # Reset
        backend.reset("test")
        
        # Should have full capacity
        remaining = backend.get_remaining("test")
        assert remaining == 5
    
    def test_stats(self):
        """Test statistics tracking"""
        backend = TokenBucketBackend(rate=1.0, capacity=2)
        
        # Make some requests
        backend.check_and_consume("test")  # Allowed
        backend.check_and_consume("test")  # Allowed
        backend.check_and_consume("test")  # Rejected
        
        stats = backend.get_stats()
        assert stats['total_requests'] == 3
        assert stats['rejected_requests'] == 1


class TestSlidingWindowBackend:
    """Tests for SlidingWindowBackend"""
    
    def test_initialization(self):
        """Test backend initialization"""
        backend = SlidingWindowBackend(max_requests=10, window_seconds=60.0)
        
        stats = backend.get_stats()
        assert stats['backend'] == 'sliding_window'
        assert stats['max_requests'] == 10
        assert stats['window_seconds'] == 60.0
    
    def test_allow_requests_within_limit(self):
        """Test requests within limit are allowed"""
        backend = SlidingWindowBackend(max_requests=5, window_seconds=60.0)
        
        for i in range(5):
            allowed, wait_time = backend.check_and_consume("test")
            assert allowed is True, f"Request {i+1} should be allowed"
            assert wait_time == 0.0
    
    def test_reject_requests_exceeding_limit(self):
        """Test requests exceeding limit are rejected"""
        backend = SlidingWindowBackend(max_requests=3, window_seconds=60.0)
        
        # Use all requests
        for _ in range(3):
            backend.check_and_consume("test")
        
        # Next request should be rejected
        allowed, wait_time = backend.check_and_consume("test")
        assert allowed is False
        assert wait_time > 0
    
    def test_window_expiration(self):
        """Test requests are allowed after window expires"""
        backend = SlidingWindowBackend(max_requests=2, window_seconds=0.1)  # 100ms window
        
        # Use all requests
        for _ in range(2):
            backend.check_and_consume("test")
        
        # Wait for window to expire
        time.sleep(0.15)
        
        # Should be allowed again
        allowed, _ = backend.check_and_consume("test")
        assert allowed is True
    
    def test_separate_keys(self):
        """Test different keys have separate windows"""
        backend = SlidingWindowBackend(max_requests=2, window_seconds=60.0)
        
        # Use all requests for key1
        for _ in range(2):
            backend.check_and_consume("key1")
        
        # key2 should still be allowed
        allowed, _ = backend.check_and_consume("key2")
        assert allowed is True
    
    def test_get_remaining(self):
        """Test get_remaining returns correct count"""
        backend = SlidingWindowBackend(max_requests=5, window_seconds=60.0)
        
        assert backend.get_remaining("test") == 5
        
        backend.check_and_consume("test")
        backend.check_and_consume("test")
        
        assert backend.get_remaining("test") == 3
    
    def test_reset(self):
        """Test reset clears the window"""
        backend = SlidingWindowBackend(max_requests=5, window_seconds=60.0)
        
        # Use some requests
        for _ in range(3):
            backend.check_and_consume("test")
        
        # Reset
        backend.reset("test")
        
        # Should have full capacity
        assert backend.get_remaining("test") == 5


class TestRateLimiter:
    """Tests for RateLimiter manager"""
    
    def test_disabled_limiter(self):
        """Test disabled rate limiter always allows"""
        config = RateLimiterConfig(enabled=False)
        limiter = RateLimiter(config)
        
        assert limiter.enabled is False
        
        allowed, wait_time = limiter.acquire()
        assert allowed is True
        assert wait_time == 0.0
    
    def test_token_bucket_backend(self):
        """Test with token bucket backend"""
        config = RateLimiterConfig(
            enabled=True,
            backend="token_bucket",
            requests_per_minute=60,
            burst_size=5
        )
        limiter = RateLimiter(config)
        
        assert limiter.enabled is True
        
        stats = limiter.get_stats()
        assert stats['backend'] == 'token_bucket'
    
    def test_sliding_window_backend(self):
        """Test with sliding window backend"""
        config = RateLimiterConfig(
            enabled=True,
            backend="sliding_window",
            requests_per_minute=60
        )
        limiter = RateLimiter(config)
        
        assert limiter.enabled is True
        
        stats = limiter.get_stats()
        assert stats['backend'] == 'sliding_window'
    
    def test_invalid_backend(self):
        """Test invalid backend raises error"""
        config = RateLimiterConfig(
            enabled=True,
            backend="invalid_backend"
        )
        
        with pytest.raises(ValueError, match="Unsupported rate limiter backend"):
            RateLimiter(config)
    
    def test_wait_and_acquire(self):
        """Test wait_and_acquire blocks until allowed"""
        config = RateLimiterConfig(
            enabled=True,
            backend="token_bucket",
            requests_per_minute=60,  # 1 per second
            burst_size=1,
            wait_timeout=5.0
        )
        limiter = RateLimiter(config)
        
        # First request should be immediate
        start = time.time()
        result = limiter.wait_and_acquire()
        assert result is True
        assert time.time() - start < 0.1
        
        # Second request should wait
        start = time.time()
        result = limiter.wait_and_acquire(timeout=2.0)
        elapsed = time.time() - start
        assert result is True
        assert elapsed >= 0.5  # Should have waited
    
    def test_wait_and_acquire_timeout(self):
        """Test wait_and_acquire returns False on timeout"""
        config = RateLimiterConfig(
            enabled=True,
            backend="sliding_window",
            requests_per_minute=1,  # Very restrictive
            wait_timeout=0.1
        )
        limiter = RateLimiter(config)
        
        # First request consumes the quota
        limiter.acquire()
        
        # Second request should timeout
        result = limiter.wait_and_acquire(timeout=0.05)
        assert result is False
    
    def test_get_remaining(self):
        """Test get_remaining returns correct count"""
        config = RateLimiterConfig(
            enabled=True,
            backend="sliding_window",
            requests_per_minute=5
        )
        limiter = RateLimiter(config)
        
        assert limiter.get_remaining() == 5
        
        limiter.acquire()
        limiter.acquire()
        
        assert limiter.get_remaining() == 3
    
    def test_get_remaining_disabled(self):
        """Test get_remaining returns -1 when disabled"""
        config = RateLimiterConfig(enabled=False)
        limiter = RateLimiter(config)
        
        assert limiter.get_remaining() == -1


class TestRateLimitContext:
    """Tests for RateLimitContext context manager"""
    
    def test_sync_context_manager(self):
        """Test sync context manager"""
        config = RateLimiterConfig(
            enabled=True,
            backend="token_bucket",
            requests_per_minute=60,
            burst_size=5
        )
        limiter = RateLimiter(config)
        
        with RateLimitContext(limiter, key="test") as acquired:
            assert acquired is True
    
    def test_sync_context_manager_no_wait(self):
        """Test sync context manager without waiting"""
        config = RateLimiterConfig(
            enabled=True,
            backend="token_bucket",
            requests_per_minute=60,
            burst_size=1
        )
        limiter = RateLimiter(config)
        
        # First should succeed
        with RateLimitContext(limiter, wait=False) as acquired:
            assert acquired is True
        
        # Second should fail without waiting
        with RateLimitContext(limiter, wait=False) as acquired:
            assert acquired is False


@pytest.mark.asyncio
class TestRateLimiterAsync:
    """Async tests for RateLimiter"""
    
    async def test_wait_and_acquire_async(self):
        """Test async wait_and_acquire"""
        config = RateLimiterConfig(
            enabled=True,
            backend="token_bucket",
            requests_per_minute=60,
            burst_size=2
        )
        limiter = RateLimiter(config)
        
        result = await limiter.wait_and_acquire_async()
        assert result is True
    
    async def test_async_context_manager(self):
        """Test async context manager"""
        config = RateLimiterConfig(
            enabled=True,
            backend="token_bucket",
            requests_per_minute=60,
            burst_size=5
        )
        limiter = RateLimiter(config)
        
        async with RateLimitContext(limiter, key="test") as acquired:
            assert acquired is True
