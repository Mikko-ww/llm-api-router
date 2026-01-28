"""Unit tests for Client and AsyncClient."""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from llm_api_router.client import Client, AsyncClient, Chat, AsyncChat, Completions, AsyncCompletions
from llm_api_router.types import ProviderConfig, UnifiedResponse, Message, Choice, Usage


class TestClient:
    """Test synchronous Client class."""

    def test_client_initialization(self, sample_provider_config):
        """Test Client initialization."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            
            assert client.config == sample_provider_config
            assert client._provider == mock_provider
            assert isinstance(client.chat, Chat)
            mock_factory.assert_called_once_with(sample_provider_config)

    def test_client_context_manager(self, sample_provider_config):
        """Test Client as context manager."""
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            with Client(sample_provider_config) as client:
                assert client is not None
                assert hasattr(client, 'close')

    def test_client_close(self, sample_provider_config):
        """Test Client close method."""
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            client = Client(sample_provider_config)
            
            # Mock the http client close method
            client._http_client.close = Mock()
            
            client.close()
            client._http_client.close.assert_called_once()

    def test_client_enter_exit(self, sample_provider_config):
        """Test Client __enter__ and __exit__ methods."""
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            client = Client(sample_provider_config)
            client._http_client.close = Mock()
            
            # Test __enter__
            context_client = client.__enter__()
            assert context_client is client
            
            # Test __exit__
            client.__exit__(None, None, None)
            client._http_client.close.assert_called_once()


class TestChat:
    """Test Chat class."""

    def test_chat_initialization(self, sample_provider_config):
        """Test Chat initialization."""
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            client = Client(sample_provider_config)
            chat = Chat(client)
            
            assert isinstance(chat.completions, Completions)
            assert chat.completions._client is client


class TestCompletions:
    """Test Completions class."""

    def test_completions_create_non_streaming(self, sample_provider_config, sample_messages, sample_unified_response):
        """Test non-streaming completions.create."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request = Mock(return_value=sample_unified_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            response = client.chat.completions.create(messages=sample_messages)
            
            assert response == sample_unified_response
            mock_provider.send_request.assert_called_once()

    def test_completions_create_with_model(self, sample_provider_config, sample_messages):
        """Test completions.create with custom model."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_response = Mock()
            mock_provider.send_request = Mock(return_value=mock_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            response = client.chat.completions.create(
                messages=sample_messages,
                model="gpt-4"
            )
            
            # Verify the request object passed to send_request
            call_args = mock_provider.send_request.call_args
            request = call_args[0][1]  # Second argument is the request
            assert request.model == "gpt-4"

    def test_completions_create_with_temperature(self, sample_provider_config, sample_messages):
        """Test completions.create with custom temperature."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_response = Mock()
            mock_provider.send_request = Mock(return_value=mock_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            client.chat.completions.create(
                messages=sample_messages,
                temperature=0.7
            )
            
            call_args = mock_provider.send_request.call_args
            request = call_args[0][1]
            assert request.temperature == 0.7

    def test_completions_create_streaming(self, sample_provider_config, sample_messages):
        """Test streaming completions.create."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_stream = iter([Mock(), Mock()])
            mock_provider.stream_request = Mock(return_value=mock_stream)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            response = client.chat.completions.create(
                messages=sample_messages,
                stream=True
            )
            
            assert response == mock_stream
            mock_provider.stream_request.assert_called_once()


class TestAsyncClient:
    """Test asynchronous AsyncClient class."""

    @pytest.mark.asyncio
    async def test_async_client_initialization(self, sample_provider_config):
        """Test AsyncClient initialization."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            
            assert client.config == sample_provider_config
            assert client._provider == mock_provider
            assert isinstance(client.chat, AsyncChat)

    @pytest.mark.asyncio
    async def test_async_client_context_manager(self, sample_provider_config):
        """Test AsyncClient as async context manager."""
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            async with AsyncClient(sample_provider_config) as client:
                assert client is not None
                assert hasattr(client, 'close')

    @pytest.mark.asyncio
    async def test_async_client_close(self, sample_provider_config):
        """Test AsyncClient close method."""
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            client = AsyncClient(sample_provider_config)
            
            # Mock the async http client aclose method
            client._http_client.aclose = AsyncMock()
            
            await client.close()
            client._http_client.aclose.assert_called_once()


class TestAsyncChat:
    """Test AsyncChat class."""

    def test_async_chat_initialization(self, sample_provider_config):
        """Test AsyncChat initialization."""
        with patch('llm_api_router.client.ProviderFactory.get_provider'):
            client = AsyncClient(sample_provider_config)
            chat = AsyncChat(client)
            
            assert isinstance(chat.completions, AsyncCompletions)
            assert chat.completions._client is client


class TestAsyncCompletions:
    """Test AsyncCompletions class."""

    @pytest.mark.asyncio
    async def test_async_completions_create_non_streaming(
        self, sample_provider_config, sample_messages, sample_unified_response
    ):
        """Test non-streaming async completions.create."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request_async = AsyncMock(return_value=sample_unified_response)
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            response = await client.chat.completions.create(messages=sample_messages)
            
            assert response == sample_unified_response
            mock_provider.send_request_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_completions_create_with_parameters(
        self, sample_provider_config, sample_messages
    ):
        """Test async completions.create with custom parameters."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_response = Mock()
            mock_provider.send_request_async = AsyncMock(return_value=mock_response)
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            await client.chat.completions.create(
                messages=sample_messages,
                model="gpt-4",
                temperature=0.5,
                max_tokens=100
            )
            
            call_args = mock_provider.send_request_async.call_args
            request = call_args[0][1]
            assert request.model == "gpt-4"
            assert request.temperature == 0.5
            assert request.max_tokens == 100

    @pytest.mark.asyncio
    async def test_async_completions_create_streaming(
        self, sample_provider_config, sample_messages
    ):
        """Test streaming async completions.create."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            
            async def mock_async_generator():
                yield Mock()
                yield Mock()
            
            mock_stream = mock_async_generator()
            mock_provider.stream_request_async = Mock(return_value=mock_stream)
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            response = await client.chat.completions.create(
                messages=sample_messages,
                stream=True
            )
            
            assert response == mock_stream
            mock_provider.stream_request_async.assert_called_once()
