"""Integration tests for the complete flow."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx
from llm_api_router import Client, AsyncClient, ProviderConfig
from llm_api_router.types import UnifiedResponse, Message, Choice, Usage
from tests.fixtures.mock_responses import OPENAI_MOCK_RESPONSE


class TestIntegrationClient:
    """Integration tests for Client."""
    
    @patch('httpx.Client')
    def test_complete_flow_openai(self, mock_http_client):
        """Test complete flow with OpenAI provider."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.json.return_value = OPENAI_MOCK_RESPONSE
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.close = Mock()
        mock_http_client.return_value = mock_client_instance
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with Client(config) as client:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Verify response structure
            assert response is not None
            assert hasattr(response, 'choices')
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], 'message')
    
    @patch('httpx.Client')
    def test_error_handling(self, mock_http_client):
        """Test error handling in integration flow."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error"
            }
        }
        
        def raise_for_status():
            raise httpx.HTTPStatusError(
                "401 Unauthorized",
                request=Mock(),
                response=mock_response
            )
        
        mock_response.raise_for_status = raise_for_status
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.close = Mock()
        mock_http_client.return_value = mock_client_instance
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="invalid-key",
            default_model="gpt-3.5-turbo"
        )
        
        with Client(config) as client:
            # Should raise an error due to authentication failure
            with pytest.raises(Exception):  # Specific error depends on provider implementation
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}]
                )


class TestIntegrationAsyncClient:
    """Integration tests for AsyncClient."""
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_complete_flow_openai_async(self, mock_http_client):
        """Test complete async flow with OpenAI provider."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.json.return_value = OPENAI_MOCK_RESPONSE
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        
        # Create async mock
        from unittest.mock import AsyncMock
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_http_client.return_value = mock_client_instance
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        async with AsyncClient(config) as client:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Verify response structure
            assert response is not None
            assert hasattr(response, 'choices')


class TestMultiProviderIntegration:
    """Test switching between different providers."""
    
    @patch('httpx.Client')
    def test_switch_providers(self, mock_http_client):
        """Test that switching providers works correctly."""
        mock_response = Mock()
        mock_response.json.return_value = OPENAI_MOCK_RESPONSE
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.close = Mock()
        mock_http_client.return_value = mock_client_instance
        
        providers = ["openai", "deepseek", "openrouter"]
        
        for provider_type in providers:
            config = ProviderConfig(
                provider_type=provider_type,
                api_key="test-key",
                default_model="test-model"
            )
            
            with Client(config) as client:
                # Should be able to create client for each provider
                assert client is not None
                assert client.config.provider_type == provider_type
