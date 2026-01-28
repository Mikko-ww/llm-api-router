"""Unit tests for retry logic."""
import pytest
import time
from unittest.mock import Mock, patch
import httpx
from llm_api_router.types import RetryConfig
from llm_api_router.retry import with_retry, with_retry_async
from llm_api_router.exceptions import MaxRetriesExceededError, RequestTimeoutError


class TestRetryConfig:
    """Test RetryConfig class."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert 429 in config.retry_on_status_codes
        assert 500 in config.retry_on_status_codes
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )
        
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
    
    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False
        )
        
        # First retry: 1.0 * 2^0 = 1.0
        assert config.calculate_delay(0) == 1.0
        # Second retry: 1.0 * 2^1 = 2.0
        assert config.calculate_delay(1) == 2.0
        # Third retry: 1.0 * 2^2 = 4.0
        assert config.calculate_delay(2) == 4.0
    
    def test_calculate_delay_max_limit(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,
            jitter=False
        )
        
        # Even though 2^10 = 1024, delay should be capped at 5.0
        assert config.calculate_delay(10) == 5.0
    
    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=True
        )
        
        delay = config.calculate_delay(0)
        # With jitter, delay should be between 0.5 and 1.5
        assert 0.5 <= delay <= 1.5


class TestWithRetry:
    """Test with_retry decorator."""
    
    def test_success_no_retry(self):
        """Test that successful function doesn't retry."""
        config = RetryConfig(max_retries=3)
        call_count = 0
        
        @with_retry(config)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_transient_error(self):
        """Test retry on transient HTTP error."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        call_count = 0
        
        @with_retry(config)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = Mock()
                response.status_code = 500
                raise httpx.HTTPStatusError("Server error", request=Mock(), response=response)
            return "success"
        
        result = failing_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test that MaxRetriesExceededError is raised after max retries."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        call_count = 0
        
        @with_retry(config)
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            response = Mock()
            response.status_code = 500
            raise httpx.HTTPStatusError("Server error", request=Mock(), response=response)
        
        with pytest.raises(MaxRetriesExceededError):
            always_failing_func()
        
        # Should have called: 1 initial + 2 retries = 3 times
        assert call_count == 3
    
    def test_no_retry_on_non_retryable_status(self):
        """Test that non-retryable status codes don't trigger retry."""
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        call_count = 0
        
        @with_retry(config)
        def non_retryable_error():
            nonlocal call_count
            call_count += 1
            response = Mock()
            response.status_code = 400  # Not in retry_on_status_codes
            raise httpx.HTTPStatusError("Bad request", request=Mock(), response=response)
        
        with pytest.raises(httpx.HTTPStatusError):
            non_retryable_error()
        
        # Should only be called once (no retries)
        assert call_count == 1
    
    def test_timeout_error_retry(self):
        """Test retry on timeout error."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        call_count = 0
        
        @with_retry(config)
        def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Request timeout")
            return "success"
        
        result = timeout_func()
        
        assert result == "success"
        assert call_count == 3


class TestWithRetryAsync:
    """Test with_retry_async decorator."""
    
    @pytest.mark.asyncio
    async def test_success_no_retry_async(self):
        """Test that successful async function doesn't retry."""
        config = RetryConfig(max_retries=3)
        call_count = 0
        
        @with_retry_async(config)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_error_async(self):
        """Test retry on transient HTTP error (async)."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        call_count = 0
        
        @with_retry_async(config)
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = Mock()
                response.status_code = 500
                raise httpx.HTTPStatusError("Server error", request=Mock(), response=response)
            return "success"
        
        result = await failing_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded_async(self):
        """Test MaxRetriesExceededError in async function."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        call_count = 0
        
        @with_retry_async(config)
        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            response = Mock()
            response.status_code = 500
            raise httpx.HTTPStatusError("Server error", request=Mock(), response=response)
        
        with pytest.raises(MaxRetriesExceededError):
            await always_failing_func()
        
        assert call_count == 3
