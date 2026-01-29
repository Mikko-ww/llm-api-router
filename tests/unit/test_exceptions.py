"""Unit tests for exceptions module."""
import pytest
from llm_api_router.exceptions import (
    LLMRouterError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RateLimitError,
    BadRequestError,
    ProviderError,
    TimeoutError,
    NetworkError,
    StreamError,
    RetryExhaustedError,
)


class TestLLMRouterError:
    """Test LLMRouterError base exception."""

    def test_basic_error(self):
        """Test creating a basic LLMRouterError."""
        error = LLMRouterError("Test error")
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.provider is None
        assert error.status_code is None
        assert error.details == {}

    def test_error_with_provider(self):
        """Test error with provider information."""
        error = LLMRouterError("API error", provider="openai")
        
        assert error.message == "API error"
        assert error.provider == "openai"

    def test_error_with_status_code(self):
        """Test error with HTTP status code."""
        error = LLMRouterError("Server error", status_code=500)
        
        assert error.status_code == 500

    def test_error_with_details(self):
        """Test error with additional details."""
        details = {"error_type": "timeout", "retry_after": 60}
        error = LLMRouterError("Error", details=details)
        
        assert error.details == details

    def test_error_with_all_parameters(self):
        """Test error with all parameters."""
        details = {"error_type": "rate_limit"}
        error = LLMRouterError(
            "Rate limit exceeded",
            provider="anthropic",
            status_code=429,
            details=details
        )
        
        assert error.message == "Rate limit exceeded"
        assert error.provider == "anthropic"
        assert error.status_code == 429
        assert error.details == details


class TestAuthenticationError:
    """Test AuthenticationError exception."""

    def test_authentication_error(self):
        """Test creating an AuthenticationError."""
        error = AuthenticationError("Invalid API key", provider="openai", status_code=401)
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Invalid API key"
        assert error.provider == "openai"
        assert error.status_code == 401

    def test_authentication_error_inheritance(self):
        """Test that AuthenticationError inherits from LLMRouterError."""
        error = AuthenticationError("Auth failed")
        
        assert isinstance(error, AuthenticationError)
        assert isinstance(error, LLMRouterError)
        assert isinstance(error, Exception)


class TestRateLimitError:
    """Test RateLimitError exception."""

    def test_rate_limit_error(self):
        """Test creating a RateLimitError."""
        error = RateLimitError("Too many requests", status_code=429)
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Too many requests"
        assert error.status_code == 429

    def test_rate_limit_error_with_retry_info(self):
        """Test RateLimitError with retry information."""
        details = {"retry_after": 60}
        error = RateLimitError("Rate limit", details=details)
        
        assert error.details["retry_after"] == 60


class TestProviderError:
    """Test ProviderError exception."""

    def test_provider_error(self):
        """Test creating a ProviderError."""
        error = ProviderError("Service unavailable", status_code=503)
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Service unavailable"
        assert error.status_code == 503

    def test_provider_error_with_provider_info(self):
        """Test ProviderError with provider information."""
        error = ProviderError(
            "API error",
            provider="gemini",
            details={"error_code": "RESOURCE_EXHAUSTED"}
        )
        
        assert error.provider == "gemini"
        assert error.details["error_code"] == "RESOURCE_EXHAUSTED"


class TestStreamError:
    """Test StreamError exception."""

    def test_stream_error(self):
        """Test creating a StreamError."""
        error = StreamError("Stream parsing failed")
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Stream parsing failed"

    def test_stream_error_with_details(self):
        """Test StreamError with parsing details."""
        details = {"chunk": "invalid data", "position": 42}
        error = StreamError("Invalid chunk", details=details)
        
        assert error.details["chunk"] == "invalid data"
        assert error.details["position"] == 42


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""

    def test_raise_and_catch_llm_router_error(self):
        """Test raising and catching LLMRouterError."""
        with pytest.raises(LLMRouterError) as exc_info:
            raise LLMRouterError("Test error")
        
        assert exc_info.value.message == "Test error"

    def test_catch_specific_error_as_base_error(self):
        """Test that specific errors can be caught as base error."""
        with pytest.raises(LLMRouterError):
            raise AuthenticationError("Auth failed")

    def test_catch_multiple_error_types(self):
        """Test catching multiple error types."""
        error_classes = [
            AuthenticationError,
            PermissionError,
            NotFoundError,
            RateLimitError,
            BadRequestError,
            ProviderError,
            TimeoutError,
            NetworkError,
            StreamError,
            RetryExhaustedError,
        ]
        for error_class in error_classes:
            with pytest.raises(LLMRouterError):
                raise error_class("Error message")


class TestPermissionError:
    """Test PermissionError exception."""

    def test_permission_error(self):
        """Test creating a PermissionError."""
        error = PermissionError("Access denied", provider="openai", status_code=403)
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Access denied"
        assert error.provider == "openai"
        assert error.status_code == 403


class TestNotFoundError:
    """Test NotFoundError exception."""

    def test_not_found_error(self):
        """Test creating a NotFoundError."""
        error = NotFoundError("Model not found", status_code=404)
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Model not found"
        assert error.status_code == 404


class TestBadRequestError:
    """Test BadRequestError exception."""

    def test_bad_request_error(self):
        """Test creating a BadRequestError."""
        error = BadRequestError("Invalid request", status_code=400)
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Invalid request"
        assert error.status_code == 400


class TestTimeoutError:
    """Test TimeoutError exception."""

    def test_timeout_error(self):
        """Test creating a TimeoutError."""
        error = TimeoutError("Request timeout")
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Request timeout"


class TestNetworkError:
    """Test NetworkError exception."""

    def test_network_error(self):
        """Test creating a NetworkError."""
        error = NetworkError("Connection failed")
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "Connection failed"


class TestRetryExhaustedError:
    """Test RetryExhaustedError exception."""

    def test_retry_exhausted_error(self):
        """Test creating a RetryExhaustedError."""
        details = {"max_retries": 3, "original_error": "Rate limit"}
        error = RetryExhaustedError("All retries failed", details=details)
        
        assert isinstance(error, LLMRouterError)
        assert error.message == "All retries failed"
        assert error.details["max_retries"] == 3

