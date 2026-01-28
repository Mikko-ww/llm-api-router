"""Unit tests for exceptions module."""
import pytest
from llm_api_router.exceptions import (
    LLMRouterError,
    AuthenticationError,
    RateLimitError,
    ProviderError,
    StreamError
)


class TestLLMRouterError:
    """Test LLMRouterError base exception."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = LLMRouterError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.provider is None
        assert error.status_code is None
        assert error.details == {}
    
    def test_error_with_all_params(self):
        """Test error with all parameters."""
        error = LLMRouterError(
            message="API error",
            provider="openai",
            status_code=500,
            details={"request_id": "123"}
        )
        assert error.message == "API error"
        assert error.provider == "openai"
        assert error.status_code == 500
        assert error.details == {"request_id": "123"}


class TestAuthenticationError:
    """Test AuthenticationError exception."""
    
    def test_authentication_error(self):
        """Test authentication error creation."""
        error = AuthenticationError(
            message="Invalid API key",
            provider="openai",
            status_code=401
        )
        assert isinstance(error, LLMRouterError)
        assert error.message == "Invalid API key"
        assert error.status_code == 401


class TestRateLimitError:
    """Test RateLimitError exception."""
    
    def test_rate_limit_error(self):
        """Test rate limit error creation."""
        error = RateLimitError(
            message="Rate limit exceeded",
            provider="anthropic",
            status_code=429
        )
        assert isinstance(error, LLMRouterError)
        assert error.message == "Rate limit exceeded"
        assert error.status_code == 429


class TestProviderError:
    """Test ProviderError exception."""
    
    def test_provider_error(self):
        """Test provider error creation."""
        error = ProviderError(
            message="Service unavailable",
            provider="gemini",
            status_code=503
        )
        assert isinstance(error, LLMRouterError)
        assert error.message == "Service unavailable"
        assert error.status_code == 503


class TestStreamError:
    """Test StreamError exception."""
    
    def test_stream_error(self):
        """Test stream error creation."""
        error = StreamError(
            message="Stream interrupted",
            provider="openai",
            details={"chunk_index": 5}
        )
        assert isinstance(error, LLMRouterError)
        assert error.message == "Stream interrupted"
        assert error.details["chunk_index"] == 5
