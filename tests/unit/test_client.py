"""Unit tests for Client and AsyncClient."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_api_router.client import Client, AsyncClient, Chat, AsyncChat, Completions, AsyncCompletions
from llm_api_router.types import ProviderConfig, UnifiedResponse, UnifiedChunk, Message, Choice, Usage


class TestCompletions:
    """Test Completions class."""
    
    def test_create_non_streaming(self):
        """Test non-streaming completion creation."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_response = UnifiedResponse(
                id="test-123",
                object="chat.completion",
                created=1234567890,
                model="gpt-3.5-turbo",
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop"
                )],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            )
            mock_provider.send_request.return_value = mock_response
            mock_factory.return_value = mock_provider
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}]
                )
                
                assert isinstance(response, UnifiedResponse)
                assert response.choices[0].message.content == "Hello!"
                mock_provider.send_request.assert_called_once()
    
    def test_create_streaming(self):
        """Test streaming completion creation."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            
            # Mock streaming response
            def mock_stream(*args, **kwargs):
                chunks = [
                    UnifiedChunk(
                        id="test-123",
                        object="chat.completion.chunk",
                        created=1234567890,
                        model="gpt-3.5-turbo",
                        choices=[Mock(index=0, delta=Message(role="assistant", content="Hello"))]
                    )
                ]
                for chunk in chunks:
                    yield chunk
            
            mock_provider.stream_request.return_value = mock_stream()
            mock_factory.return_value = mock_provider
            
            with Client(config) as client:
                stream = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True
                )
                
                chunks = list(stream)
                assert len(chunks) == 1
                mock_provider.stream_request.assert_called_once()


class TestChat:
    """Test Chat class."""
    
    def test_chat_has_completions(self):
        """Test that Chat has completions attribute."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            with Client(config) as client:
                assert hasattr(client.chat, 'completions')
                assert isinstance(client.chat.completions, Completions)


class TestClient:
    """Test Client class."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            with Client(config) as client:
                assert client.config == config
                assert hasattr(client, 'chat')
                assert isinstance(client.chat, Chat)
    
    def test_client_context_manager(self):
        """Test client as context manager."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            with Client(config) as client:
                assert client is not None
            # Client should be closed after exiting context


class TestAsyncCompletions:
    """Test AsyncCompletions class."""
    
    @pytest.mark.asyncio
    async def test_create_non_streaming(self):
        """Test non-streaming async completion creation."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        from unittest.mock import AsyncMock
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_response = UnifiedResponse(
                id="test-123",
                object="chat.completion",
                created=1234567890,
                model="gpt-3.5-turbo",
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop"
                )],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            )
            # Make send_request_async return a coroutine
            async def async_return():
                return mock_response
            mock_provider.send_request_async = AsyncMock(return_value=mock_response)
            mock_factory.return_value = mock_provider
            
            async with AsyncClient(config) as client:
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}]
                )
                
                assert isinstance(response, UnifiedResponse)
                assert response.choices[0].message.content == "Hello!"


class TestAsyncClient:
    """Test AsyncClient class."""
    
    @pytest.mark.asyncio
    async def test_async_client_initialization(self):
        """Test async client initialization."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            async with AsyncClient(config) as client:
                assert client.config == config
                assert hasattr(client, 'chat')
                assert isinstance(client.chat, AsyncChat)
    
    @pytest.mark.asyncio
    async def test_async_client_context_manager(self):
        """Test async client as context manager."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            async with AsyncClient(config) as client:
                assert client is not None
            # Client should be closed after exiting context
