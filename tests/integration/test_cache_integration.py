"""
Integration tests for cache functionality with Client
"""
import pytest
from unittest.mock import Mock, patch
from llm_api_router import Client, AsyncClient, ProviderConfig
from llm_api_router.cache import CacheConfig
from llm_api_router.types import UnifiedResponse, Choice, Message, Usage


@pytest.fixture
def mock_response():
    """Create a mock UnifiedResponse"""
    return UnifiedResponse(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Hello! How can I help you?"),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18
        )
    )


@pytest.fixture
def provider_config_with_cache():
    """Create a ProviderConfig with cache enabled"""
    cache_config = CacheConfig(
        enabled=True,
        backend="memory",
        ttl=60,
        max_size=100
    )
    
    return ProviderConfig(
        provider_type="openai",
        api_key="test-key",
        cache_config=cache_config
    )


@pytest.fixture
def provider_config_without_cache():
    """Create a ProviderConfig without cache"""
    return ProviderConfig(
        provider_type="openai",
        api_key="test-key"
    )


class TestClientCaching:
    """Test caching functionality with Client"""
    
    def test_cache_disabled_by_default(self, provider_config_without_cache):
        """Test that cache is disabled by default"""
        client = Client(provider_config_without_cache)
        
        stats = client.get_cache_stats()
        assert stats['enabled'] is False
    
    def test_cache_enabled(self, provider_config_with_cache):
        """Test that cache can be enabled"""
        client = Client(provider_config_with_cache)
        
        stats = client.get_cache_stats()
        assert stats['enabled'] is True
        assert stats['backend'] == 'memory'
    
    def test_cache_hit(self, provider_config_with_cache, mock_response):
        """Test cache hit on repeated request"""
        client = Client(provider_config_with_cache)
        
        # Mock the provider's send_request method
        with patch.object(client._provider, 'send_request', return_value=mock_response) as mock_send:
            # First call - should hit the API
            response1 = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            assert mock_send.call_count == 1
            assert response1.choices[0].message.content == "Hello! How can I help you?"
            
            # Second call with same parameters - should hit cache
            response2 = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Should still be 1 because second call used cache
            assert mock_send.call_count == 1
            assert response2.choices[0].message.content == "Hello! How can I help you?"
            
            # Check cache stats
            stats = client.get_cache_stats()
            assert stats['hits'] >= 1
    
    def test_cache_miss_on_different_request(self, provider_config_with_cache, mock_response):
        """Test cache miss on different request"""
        client = Client(provider_config_with_cache)
        
        with patch.object(client._provider, 'send_request', return_value=mock_response) as mock_send:
            # First call
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Second call with different content - should not hit cache
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-3.5-turbo"
            )
            
            # Should be 2 calls since content is different
            assert mock_send.call_count == 2
    
    def test_streaming_not_cached(self, provider_config_with_cache):
        """Test that streaming requests are not cached"""
        client = Client(provider_config_with_cache)
        
        # Mock the stream_request method
        mock_stream = Mock(return_value=iter([]))
        
        with patch.object(client._provider, 'stream_request', mock_stream):
            # Call with stream=True
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo",
                stream=True
            )
            
            # Consume the iterator
            list(result)
            
            # Should have called stream_request
            assert mock_stream.call_count == 1
            
            # Second call should also call stream_request (not cached)
            result = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo",
                stream=True
            )
            list(result)
            
            assert mock_stream.call_count == 2
    
    def test_clear_cache(self, provider_config_with_cache, mock_response):
        """Test clearing cache"""
        client = Client(provider_config_with_cache)
        
        with patch.object(client._provider, 'send_request', return_value=mock_response) as mock_send:
            # First call
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Clear cache
            client.clear_cache()
            
            # Second call with same parameters - should hit API again
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Should be 2 calls since cache was cleared
            assert mock_send.call_count == 2
    
    def test_cache_with_different_models(self, provider_config_with_cache, mock_response):
        """Test that different models produce different cache keys"""
        client = Client(provider_config_with_cache)
        
        with patch.object(client._provider, 'send_request', return_value=mock_response) as mock_send:
            # First call with gpt-3.5-turbo
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Second call with gpt-4 - should not hit cache
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4"
            )
            
            # Should be 2 calls since models are different
            assert mock_send.call_count == 2


@pytest.mark.asyncio
class TestAsyncClientCaching:
    """Test caching functionality with AsyncClient"""
    
    async def test_cache_enabled(self, provider_config_with_cache):
        """Test that cache can be enabled for async client"""
        client = AsyncClient(provider_config_with_cache)
        
        stats = client.get_cache_stats()
        assert stats['enabled'] is True
        assert stats['backend'] == 'memory'
        
        await client.close()
    
    async def test_cache_hit(self, provider_config_with_cache, mock_response):
        """Test cache hit on repeated async request"""
        client = AsyncClient(provider_config_with_cache)
        
        # Mock the provider's send_request_async method
        with patch.object(client._provider, 'send_request_async', return_value=mock_response) as mock_send:
            # First call - should hit the API
            response1 = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            assert mock_send.call_count == 1
            assert response1.choices[0].message.content == "Hello! How can I help you?"
            
            # Second call with same parameters - should hit cache
            response2 = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Should still be 1 because second call used cache
            assert mock_send.call_count == 1
            assert response2.choices[0].message.content == "Hello! How can I help you?"
            
            # Check cache stats
            stats = client.get_cache_stats()
            assert stats['hits'] >= 1
        
        await client.close()
    
    async def test_clear_cache(self, provider_config_with_cache, mock_response):
        """Test clearing cache in async client"""
        client = AsyncClient(provider_config_with_cache)
        
        with patch.object(client._provider, 'send_request_async', return_value=mock_response) as mock_send:
            # First call
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Clear cache
            client.clear_cache()
            
            # Second call with same parameters - should hit API again
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            # Should be 2 calls since cache was cleared
            assert mock_send.call_count == 2
        
        await client.close()
