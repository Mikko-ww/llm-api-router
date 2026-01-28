"""Tests for retry module."""
import pytest
import time
import asyncio
from unittest.mock import Mock
import httpx

from llm_api_router.retry import (
    calculate_backoff_delay,
    should_retry,
    with_retry,
    with_retry_async,
)
from llm_api_router.types import RetryConfig
from llm_api_router.exceptions import (
    RetryExhaustedError,
    RateLimitError,
    ProviderError,
    TimeoutError,
    NetworkError,
    AuthenticationError,
)


class TestCalculateBackoffDelay:
    """Tests for calculate_backoff_delay function."""

    def test_initial_delay(self):
        """Test initial delay calculation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0)
        delay = calculate_backoff_delay(0, config)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Test exponential growth of delay."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0)
        
        assert calculate_backoff_delay(0, config) == 1.0
        assert calculate_backoff_delay(1, config) == 2.0
        assert calculate_backoff_delay(2, config) == 4.0
        assert calculate_backoff_delay(3, config) == 8.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=5.0)
        
        assert calculate_backoff_delay(10, config) == 5.0

    def test_custom_exponential_base(self):
        """Test with custom exponential base."""
        config = RetryConfig(initial_delay=1.0, exponential_base=3.0)
        
        assert calculate_backoff_delay(0, config) == 1.0
        assert calculate_backoff_delay(1, config) == 3.0
        assert calculate_backoff_delay(2, config) == 9.0


class TestShouldRetry:
    """Tests for should_retry function."""

    def test_retry_on_configured_status_codes(self):
        """Test retry on configured status codes."""
        config = RetryConfig(retry_on_status_codes=(429, 500, 502, 503, 504))
        
        error = ProviderError("Server error", status_code=500)
        assert should_retry(error, 500, config) is True
        
        error = RateLimitError("Rate limit", status_code=429)
        assert should_retry(error, 429, config) is True

    def test_no_retry_on_client_errors(self):
        """Test no retry on client errors."""
        config = RetryConfig()
        
        error = AuthenticationError("Auth failed", status_code=401)
        assert should_retry(error, 401, config) is False

    def test_retry_on_network_errors(self):
        """Test retry on network errors."""
        config = RetryConfig()
        
        assert should_retry(NetworkError("Network error"), None, config) is True
        assert should_retry(TimeoutError("Timeout"), None, config) is True
        assert should_retry(httpx.TimeoutException("Timeout"), None, config) is True
        assert should_retry(httpx.NetworkError("Network"), None, config) is True

    def test_retry_on_rate_limit(self):
        """Test retry on rate limit errors."""
        config = RetryConfig()
        
        error = RateLimitError("Too many requests")
        assert should_retry(error, None, config) is True

    def test_retry_on_server_errors(self):
        """Test retry on server errors (5xx)."""
        config = RetryConfig()
        
        error = ProviderError("Server error", status_code=503)
        assert should_retry(error, 503, config) is True


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_successful_on_first_try(self):
        """Test successful execution on first try."""
        call_count = 0
        
        @with_retry()
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Test retry on transient errors."""
        call_count = 0
        
        @with_retry(RetryConfig(max_retries=2, initial_delay=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limit", status_code=429)
            return "success"
        
        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test retry exhausted after max attempts."""
        call_count = 0
        
        @with_retry(RetryConfig(max_retries=2, initial_delay=0.01))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Rate limit", status_code=429)
        
        with pytest.raises(RetryExhaustedError) as exc_info:
            always_fails()
        
        assert call_count == 3  # Initial + 2 retries
        assert "已重试 2 次" in str(exc_info.value)

    def test_no_retry_on_auth_error(self):
        """Test no retry on authentication errors."""
        call_count = 0
        
        @with_retry(RetryConfig(max_retries=2, initial_delay=0.01))
        def auth_fails():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Auth failed", status_code=401)
        
        with pytest.raises(AuthenticationError):
            auth_fails()
        
        assert call_count == 1  # No retries

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff delays are applied."""
        call_count = 0
        call_times = []
        
        @with_retry(RetryConfig(max_retries=2, initial_delay=0.1, exponential_base=2.0))
        def timed_func():
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())
            if call_count < 3:
                raise RateLimitError("Rate limit", status_code=429)
            return "success"
        
        start = time.time()
        result = timed_func()
        
        assert result == "success"
        assert call_count == 3
        
        # Check that delays are approximately correct
        # First delay: ~0.1s, second delay: ~0.2s
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            
            assert 0.08 < delay1 < 0.15  # ~0.1s with some tolerance
            assert 0.15 < delay2 < 0.25  # ~0.2s with some tolerance


class TestWithRetryAsync:
    """Tests for with_retry_async decorator."""

    @pytest.mark.asyncio
    async def test_successful_on_first_try(self):
        """Test successful execution on first try."""
        call_count = 0
        
        @with_retry_async()
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Test retry on transient errors."""
        call_count = 0
        
        @with_retry_async(RetryConfig(max_retries=2, initial_delay=0.01))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limit", status_code=429)
            return "success"
        
        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry exhausted after max attempts."""
        call_count = 0
        
        @with_retry_async(RetryConfig(max_retries=2, initial_delay=0.01))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Network error")
        
        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_fails()
        
        assert call_count == 3  # Initial + 2 retries
        assert "已重试 2 次" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        """Test no retry on authentication errors."""
        call_count = 0
        
        @with_retry_async(RetryConfig(max_retries=2, initial_delay=0.01))
        async def auth_fails():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Auth failed", status_code=401)
        
        with pytest.raises(AuthenticationError):
            await auth_fails()
        
        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that exponential backoff delays are applied."""
        call_count = 0
        call_times = []
        
        @with_retry_async(RetryConfig(max_retries=2, initial_delay=0.1, exponential_base=2.0))
        async def timed_func():
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())
            if call_count < 3:
                raise TimeoutError("Timeout")
            return "success"
        
        start = time.time()
        result = await timed_func()
        
        assert result == "success"
        assert call_count == 3
        
        # Check that delays are approximately correct
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            
            assert 0.08 < delay1 < 0.15  # ~0.1s with some tolerance
            assert 0.15 < delay2 < 0.25  # ~0.2s with some tolerance
