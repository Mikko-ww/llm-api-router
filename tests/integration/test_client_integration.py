"""Integration tests for llm-api-router with mocked HTTP responses."""
import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock
from llm_api_router.client import Client, AsyncClient
from llm_api_router.types import ProviderConfig
from tests.fixtures.mock_responses import (
    get_openai_chat_response,
    get_anthropic_chat_response,
    get_gemini_chat_response,
    get_deepseek_chat_response,
    SAMPLE_MESSAGES
)


class TestClientIntegration:
    """Integration tests for synchronous Client."""

    def test_openai_chat_completion_non_streaming(self):
        """Test OpenAI chat completion without streaming."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = get_openai_chat_response()
            mock_post.return_value = mock_response
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=SAMPLE_MESSAGES,
                    temperature=0.7
                )
                
                assert response is not None
                assert response.model == "gpt-3.5-turbo-0125"
                assert len(response.choices) == 1
                assert response.choices[0].message.role == "assistant"
                assert "AI assistant" in response.choices[0].message.content
                assert response.usage.total_tokens > 0

    def test_deepseek_chat_completion(self):
        """Test DeepSeek chat completion."""
        config = ProviderConfig(
            provider_type="deepseek",
            api_key="sk-test-key",
            default_model="deepseek-chat"
        )
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = get_deepseek_chat_response()
            mock_post.return_value = mock_response
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=SAMPLE_MESSAGES
                )
                
                assert response is not None
                assert "deepseek" in response.model.lower()
                assert len(response.choices) == 1

    def test_client_with_different_models(self):
        """Test client with different model specifications."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            
            # Simulate different responses for different models
            def json_response():
                response_data = get_openai_chat_response()
                response_data["model"] = "gpt-4"
                return response_data
            
            mock_response.json.side_effect = json_response
            mock_post.return_value = mock_response
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=SAMPLE_MESSAGES,
                    model="gpt-4"
                )
                
                # Verify that model override worked
                call_args = mock_post.call_args
                request_data = call_args[1]['json']
                assert request_data['model'] == 'gpt-4'

    def test_client_error_handling(self):
        """Test client error handling for various HTTP errors."""
        from llm_api_router import RetryConfig
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-invalid-key",
            default_model="gpt-3.5-turbo",
            retry_config=RetryConfig(max_retries=0)  # Disable retries to immediately get error
        )
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {
                "error": {"message": "Invalid authentication"}
            }
            mock_post.return_value = mock_response
            
            from llm_api_router.exceptions import AuthenticationError
            
            with Client(config) as client:
                with pytest.raises(AuthenticationError):
                    client.chat.completions.create(messages=SAMPLE_MESSAGES)


class TestAsyncClientIntegration:
    """Integration tests for asynchronous AsyncClient."""

    @pytest.mark.asyncio
    async def test_async_openai_chat_completion(self):
        """Test async OpenAI chat completion."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            # For async, json() should return the value directly, not a coroutine
            mock_response.json.return_value = get_openai_chat_response()
            mock_post.return_value = mock_response
            
            async with AsyncClient(config) as client:
                response = await client.chat.completions.create(
                    messages=SAMPLE_MESSAGES,
                    temperature=0.5
                )
                
                assert response is not None
                assert response.model == "gpt-3.5-turbo-0125"
                assert len(response.choices) == 1
                assert response.choices[0].message.role == "assistant"

    @pytest.mark.asyncio
    async def test_async_client_with_max_tokens(self):
        """Test async client with max_tokens parameter."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = get_openai_chat_response()
            mock_post.return_value = mock_response
            
            async with AsyncClient(config) as client:
                response = await client.chat.completions.create(
                    messages=SAMPLE_MESSAGES,
                    max_tokens=100
                )
                
                # Verify max_tokens was passed
                call_args = mock_post.call_args
                request_data = call_args[1]['json']
                assert request_data['max_tokens'] == 100

    @pytest.mark.asyncio
    async def test_async_client_error_handling(self):
        """Test async client error handling."""
        from llm_api_router import RetryConfig
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-invalid-key",
            default_model="gpt-3.5-turbo",
            retry_config=RetryConfig(max_retries=0)  # Disable retries to immediately get error
        )
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {
                "error": {"message": "Rate limit exceeded"}
            }
            mock_post.return_value = mock_response
            
            from llm_api_router.exceptions import RateLimitError
            
            async with AsyncClient(config) as client:
                with pytest.raises(RateLimitError):
                    await client.chat.completions.create(messages=SAMPLE_MESSAGES)


class TestMultiProviderIntegration:
    """Integration tests for multiple providers."""

    @pytest.mark.parametrize("provider_type,mock_response,expected_model", [
        ("openai", get_openai_chat_response(), "gpt-3.5-turbo-0125"),
        ("deepseek", get_deepseek_chat_response(), "deepseek-chat"),
    ])
    def test_multiple_providers(self, provider_type, mock_response, expected_model):
        """Test that different providers work correctly."""
        config = ProviderConfig(
            provider_type=provider_type,
            api_key="sk-test-key"
        )
        
        with patch('httpx.Client.post') as mock_post:
            mock_http_response = Mock()
            mock_http_response.status_code = 200
            mock_http_response.json.return_value = mock_response
            mock_post.return_value = mock_http_response
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=SAMPLE_MESSAGES
                )
                
                assert response is not None
                assert response.model == expected_model
                assert len(response.choices) > 0


class TestClientConfiguration:
    """Integration tests for client configuration."""

    def test_client_with_custom_base_url(self):
        """Test client with custom base URL."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            base_url="https://custom.api.com/v1",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = get_openai_chat_response()
            mock_post.return_value = mock_response
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=SAMPLE_MESSAGES
                )
                
                # Verify custom base URL was used
                call_args = mock_post.call_args
                url = call_args[0][0]
                assert "custom.api.com" in url

    def test_client_with_extra_headers(self):
        """Test client with extra headers."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo",
            extra_headers={"X-Custom-Header": "custom-value"}
        )
        
        with patch('httpx.Client.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = get_openai_chat_response()
            mock_post.return_value = mock_response
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=SAMPLE_MESSAGES
                )
                
                # Verify extra headers were included
                call_args = mock_post.call_args
                headers = call_args[1]['headers']
                assert "X-Custom-Header" in headers
                assert headers["X-Custom-Header"] == "custom-value"
