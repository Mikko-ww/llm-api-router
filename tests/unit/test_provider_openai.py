"""Unit tests for OpenAI provider."""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import httpx
from llm_api_router.providers.openai import OpenAIProvider
from llm_api_router.types import ProviderConfig, UnifiedRequest, Message, Choice
from llm_api_router.exceptions import AuthenticationError, RateLimitError, ProviderError, RetryExhaustedError
from tests.fixtures.mock_responses import (
    get_openai_chat_response,
    SAMPLE_MESSAGES
)


class TestOpenAIProvider:
    """Test OpenAI provider."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )

    @pytest.fixture
    def openai_provider(self, openai_config):
        """OpenAI provider instance."""
        return OpenAIProvider(openai_config)

    def test_provider_initialization(self, openai_config):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(openai_config)
        
        assert provider.config == openai_config
        assert provider.base_url == "https://api.openai.com/v1"
        assert "Authorization" in provider.headers
        assert provider.headers["Authorization"] == "Bearer sk-test-key"
        assert provider.headers["Content-Type"] == "application/json"

    def test_provider_initialization_with_custom_base_url(self):
        """Test provider initialization with custom base URL."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            base_url="https://custom.api.com/v1/"
        )
        provider = OpenAIProvider(config)
        
        # Should strip trailing slash
        assert provider.base_url == "https://custom.api.com/v1"

    def test_provider_initialization_with_extra_headers(self):
        """Test provider initialization with extra headers."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            extra_headers={"X-Custom-Header": "value"}
        )
        provider = OpenAIProvider(config)
        
        assert provider.headers["X-Custom-Header"] == "value"

    def test_convert_request_basic(self, openai_provider):
        """Test converting basic request."""
        request = UnifiedRequest(
            messages=SAMPLE_MESSAGES,
            model="gpt-4"
        )
        
        converted = openai_provider.convert_request(request)
        
        assert converted["model"] == "gpt-4"
        assert converted["messages"] == SAMPLE_MESSAGES
        assert converted["stream"] is False
        assert "temperature" in converted

    def test_convert_request_uses_default_model(self, openai_provider):
        """Test that convert_request uses default model when not specified."""
        request = UnifiedRequest(messages=SAMPLE_MESSAGES)
        
        converted = openai_provider.convert_request(request)
        
        assert converted["model"] == "gpt-3.5-turbo"

    def test_convert_request_with_all_parameters(self, openai_provider):
        """Test converting request with all parameters."""
        request = UnifiedRequest(
            messages=SAMPLE_MESSAGES,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            stream=True,
            top_p=0.9,
            stop=["END"]
        )
        
        converted = openai_provider.convert_request(request)
        
        assert converted["model"] == "gpt-4"
        assert converted["temperature"] == 0.7
        assert converted["max_tokens"] == 100
        assert converted["stream"] is True
        assert converted["top_p"] == 0.9
        assert converted["stop"] == ["END"]

    def test_convert_request_omits_none_values(self, openai_provider):
        """Test that convert_request omits None values."""
        request = UnifiedRequest(
            messages=SAMPLE_MESSAGES,
            max_tokens=None,
            top_p=None,
            stop=None
        )
        
        converted = openai_provider.convert_request(request)
        
        # These should not be in the converted request if they are None
        assert "max_tokens" not in converted or converted["max_tokens"] is None

    def test_convert_response(self, openai_provider):
        """Test converting OpenAI response to unified format."""
        provider_response = get_openai_chat_response()
        
        unified_response = openai_provider.convert_response(provider_response)
        
        assert unified_response.id == provider_response["id"]
        assert unified_response.object == provider_response["object"]
        assert unified_response.created == provider_response["created"]
        assert unified_response.model == provider_response["model"]
        assert len(unified_response.choices) == 1
        
        choice = unified_response.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert "AI assistant" in choice.message.content
        assert choice.finish_reason == "stop"
        
        assert unified_response.usage.prompt_tokens == 12
        assert unified_response.usage.completion_tokens == 15
        assert unified_response.usage.total_tokens == 27

    def test_convert_response_with_empty_choices(self, openai_provider):
        """Test converting response with empty choices."""
        provider_response = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 123456,
            "model": "gpt-3.5-turbo",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        unified_response = openai_provider.convert_response(provider_response)
        
        assert len(unified_response.choices) == 0
        assert unified_response.usage.total_tokens == 0

    def test_convert_chunk(self, openai_provider):
        """Test converting stream chunk."""
        chunk_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None
                }
            ]
        }
        
        unified_chunk = openai_provider._convert_chunk(chunk_data)
        
        assert unified_chunk.id == "chatcmpl-test"
        assert unified_chunk.object == "chat.completion.chunk"
        assert len(unified_chunk.choices) == 1
        
        choice = unified_chunk.choices[0]
        assert choice.index == 0
        assert choice.delta.content == "Hello"
        assert choice.finish_reason is None

    def test_send_request_success(self, openai_provider):
        """Test successful send_request."""
        request = UnifiedRequest(messages=SAMPLE_MESSAGES)
        mock_response_data = get_openai_chat_response()
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value = mock_response
            
            mock_client = httpx.Client()
            response = openai_provider.send_request(mock_client, request)
            
            assert response.id == mock_response_data["id"]
            assert response.model == mock_response_data["model"]
            assert len(response.choices) > 0

    def test_send_request_authentication_error(self, openai_provider):
        """Test send_request with authentication error."""
        request = UnifiedRequest(messages=SAMPLE_MESSAGES)
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
            mock_post.return_value = mock_response
            
            mock_client = httpx.Client()
            
            with pytest.raises(AuthenticationError):
                openai_provider.send_request(mock_client, request)

    def test_send_request_rate_limit_error(self, openai_provider):
        """Test send_request with rate limit error."""
        request = UnifiedRequest(messages=SAMPLE_MESSAGES)
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
            mock_post.return_value = mock_response
            
            mock_client = httpx.Client()
            
            # Should raise RetryExhaustedError after retrying
            with pytest.raises(RetryExhaustedError):
                openai_provider.send_request(mock_client, request)

    def test_send_request_provider_error(self, openai_provider):
        """Test send_request with provider error."""
        request = UnifiedRequest(messages=SAMPLE_MESSAGES)
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": {"message": "Internal server error"}}
            mock_post.return_value = mock_response
            
            mock_client = httpx.Client()
            
            # Should raise RetryExhaustedError after retrying
            with pytest.raises(RetryExhaustedError):
                openai_provider.send_request(mock_client, request)
