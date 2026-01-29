"""Unit tests for Embeddings API."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from llm_api_router.client import Client, AsyncClient, Embeddings, AsyncEmbeddings
from llm_api_router.types import (
    ProviderConfig, EmbeddingRequest, EmbeddingResponse, Embedding, EmbeddingUsage
)


@pytest.fixture
def sample_embedding_response() -> EmbeddingResponse:
    """Sample embedding response for testing."""
    return EmbeddingResponse(
        data=[
            Embedding(index=0, embedding=[0.1, 0.2, 0.3], object="embedding"),
            Embedding(index=1, embedding=[0.4, 0.5, 0.6], object="embedding"),
        ],
        model="text-embedding-3-small",
        usage=EmbeddingUsage(prompt_tokens=10, total_tokens=10),
        object="list"
    )


@pytest.fixture
def mock_openai_embedding_response() -> dict:
    """Mock OpenAI embedding API response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                "object": "embedding",
                "index": 1,
                "embedding": [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8
        }
    }


class TestEmbeddings:
    """Test synchronous Embeddings class."""

    def test_embeddings_initialization(self, sample_provider_config):
        """Test Embeddings initialization."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            embeddings = Embeddings(client)
            
            assert embeddings._client is client

    def test_embeddings_create_single_text(self, sample_provider_config, sample_embedding_response):
        """Test creating embeddings with a single text string."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings = Mock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            response = client.embeddings.create(input="Hello, world!")
            
            assert response == sample_embedding_response
            mock_provider.create_embeddings.assert_called_once()
            
            # Verify the request was converted to a list
            call_args = mock_provider.create_embeddings.call_args
            request = call_args[0][1]
            assert isinstance(request, EmbeddingRequest)
            assert request.input == ["Hello, world!"]

    def test_embeddings_create_multiple_texts(self, sample_provider_config, sample_embedding_response):
        """Test creating embeddings with multiple texts."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings = Mock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            texts = ["Hello", "World", "Test"]
            response = client.embeddings.create(input=texts)
            
            assert response == sample_embedding_response
            
            call_args = mock_provider.create_embeddings.call_args
            request = call_args[0][1]
            assert request.input == texts

    def test_embeddings_create_with_model(self, sample_provider_config, sample_embedding_response):
        """Test creating embeddings with a specific model."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings = Mock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            response = client.embeddings.create(
                input="test",
                model="text-embedding-ada-002"
            )
            
            call_args = mock_provider.create_embeddings.call_args
            request = call_args[0][1]
            assert request.model == "text-embedding-ada-002"

    def test_embeddings_create_with_dimensions(self, sample_provider_config, sample_embedding_response):
        """Test creating embeddings with specific dimensions."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings = Mock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            response = client.embeddings.create(
                input="test",
                dimensions=512
            )
            
            call_args = mock_provider.create_embeddings.call_args
            request = call_args[0][1]
            assert request.dimensions == 512

    def test_embeddings_create_with_encoding_format(self, sample_provider_config, sample_embedding_response):
        """Test creating embeddings with specific encoding format."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings = Mock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = Client(sample_provider_config)
            response = client.embeddings.create(
                input="test",
                encoding_format="base64"
            )
            
            call_args = mock_provider.create_embeddings.call_args
            request = call_args[0][1]
            assert request.encoding_format == "base64"


class TestAsyncEmbeddings:
    """Test asynchronous AsyncEmbeddings class."""

    @pytest.mark.asyncio
    async def test_async_embeddings_initialization(self, sample_provider_config):
        """Test AsyncEmbeddings initialization."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            embeddings = AsyncEmbeddings(client)
            
            assert embeddings._client is client

    @pytest.mark.asyncio
    async def test_async_embeddings_create_single_text(self, sample_provider_config, sample_embedding_response):
        """Test creating async embeddings with a single text string."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings_async = AsyncMock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            response = await client.embeddings.create(input="Hello, world!")
            
            assert response == sample_embedding_response
            mock_provider.create_embeddings_async.assert_called_once()
            
            call_args = mock_provider.create_embeddings_async.call_args
            request = call_args[0][1]
            assert isinstance(request, EmbeddingRequest)
            assert request.input == ["Hello, world!"]

    @pytest.mark.asyncio
    async def test_async_embeddings_create_multiple_texts(self, sample_provider_config, sample_embedding_response):
        """Test creating async embeddings with multiple texts."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings_async = AsyncMock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            texts = ["Hello", "World", "Test"]
            response = await client.embeddings.create(input=texts)
            
            assert response == sample_embedding_response
            
            call_args = mock_provider.create_embeddings_async.call_args
            request = call_args[0][1]
            assert request.input == texts

    @pytest.mark.asyncio
    async def test_async_embeddings_create_with_parameters(self, sample_provider_config, sample_embedding_response):
        """Test creating async embeddings with all parameters."""
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings_async = AsyncMock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            client = AsyncClient(sample_provider_config)
            response = await client.embeddings.create(
                input="test",
                model="text-embedding-3-large",
                encoding_format="float",
                dimensions=256
            )
            
            call_args = mock_provider.create_embeddings_async.call_args
            request = call_args[0][1]
            assert request.model == "text-embedding-3-large"
            assert request.encoding_format == "float"
            assert request.dimensions == 256


class TestEmbeddingTypes:
    """Test embedding data types."""

    def test_embedding_request_creation(self):
        """Test EmbeddingRequest dataclass creation."""
        request = EmbeddingRequest(
            input=["test1", "test2"],
            model="text-embedding-3-small",
            encoding_format="float",
            dimensions=512
        )
        
        assert request.input == ["test1", "test2"]
        assert request.model == "text-embedding-3-small"
        assert request.encoding_format == "float"
        assert request.dimensions == 512

    def test_embedding_request_defaults(self):
        """Test EmbeddingRequest default values."""
        request = EmbeddingRequest(input=["test"])
        
        assert request.input == ["test"]
        assert request.model is None
        assert request.encoding_format is None
        assert request.dimensions is None

    def test_embedding_creation(self):
        """Test Embedding dataclass creation."""
        embedding = Embedding(
            index=0,
            embedding=[0.1, 0.2, 0.3],
            object="embedding"
        )
        
        assert embedding.index == 0
        assert embedding.embedding == [0.1, 0.2, 0.3]
        assert embedding.object == "embedding"

    def test_embedding_usage_creation(self):
        """Test EmbeddingUsage dataclass creation."""
        usage = EmbeddingUsage(
            prompt_tokens=10,
            total_tokens=10
        )
        
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 10

    def test_embedding_response_creation(self):
        """Test EmbeddingResponse dataclass creation."""
        response = EmbeddingResponse(
            data=[
                Embedding(index=0, embedding=[0.1, 0.2], object="embedding")
            ],
            model="test-model",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
            object="list"
        )
        
        assert len(response.data) == 1
        assert response.model == "test-model"
        assert response.usage.prompt_tokens == 5
        assert response.object == "list"


class TestOpenAIEmbeddings:
    """Test OpenAI provider embeddings implementation."""

    def test_openai_supports_embeddings(self, sample_provider_config):
        """Test that OpenAI provider supports embeddings."""
        from llm_api_router.providers.openai import OpenAIProvider
        
        provider = OpenAIProvider(sample_provider_config)
        assert provider.supports_embeddings() is True

    def test_openai_convert_embedding_request(self, sample_provider_config):
        """Test OpenAI embedding request conversion."""
        from llm_api_router.providers.openai import OpenAIProvider
        
        provider = OpenAIProvider(sample_provider_config)
        request = EmbeddingRequest(
            input=["test1", "test2"],
            model="text-embedding-3-small",
            dimensions=512
        )
        
        converted = provider._convert_embedding_request(request)
        
        assert converted["input"] == ["test1", "test2"]
        assert converted["model"] == "text-embedding-3-small"
        assert converted["dimensions"] == 512

    def test_openai_convert_embedding_response(self, sample_provider_config, mock_openai_embedding_response):
        """Test OpenAI embedding response conversion."""
        from llm_api_router.providers.openai import OpenAIProvider
        
        provider = OpenAIProvider(sample_provider_config)
        response = provider._convert_embedding_response(mock_openai_embedding_response)
        
        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 2
        assert response.data[0].index == 0
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert response.model == "text-embedding-3-small"
        assert response.usage.prompt_tokens == 8
        assert response.usage.total_tokens == 8


class TestProviderEmbeddingsSupport:
    """Test embeddings support across different providers."""

    def test_base_provider_not_support_embeddings(self, sample_provider_config):
        """Test that base provider doesn't support embeddings by default."""
        from llm_api_router.providers.base import BaseProvider
        
        # Create a minimal concrete implementation
        class TestProvider(BaseProvider):
            def convert_request(self, request): pass
            def convert_response(self, response): pass
            def send_request(self, client, request): pass
            async def send_request_async(self, client, request): pass
            def stream_request(self, client, request): pass
            async def stream_request_async(self, client, request): pass
        
        provider = TestProvider(sample_provider_config)
        assert provider.supports_embeddings() is False

    def test_base_provider_embeddings_raises_not_implemented(self, sample_provider_config):
        """Test that base provider raises NotImplementedError for embeddings."""
        from llm_api_router.providers.base import BaseProvider
        
        class TestProvider(BaseProvider):
            def convert_request(self, request): pass
            def convert_response(self, response): pass
            def send_request(self, client, request): pass
            async def send_request_async(self, client, request): pass
            def stream_request(self, client, request): pass
            async def stream_request_async(self, client, request): pass
        
        provider = TestProvider(sample_provider_config)
        request = EmbeddingRequest(input=["test"])
        
        with pytest.raises(NotImplementedError):
            provider.create_embeddings(Mock(), request)

    def test_gemini_supports_embeddings(self):
        """Test that Gemini provider supports embeddings."""
        from llm_api_router.providers.gemini import GeminiProvider
        
        config = ProviderConfig(
            provider_type="gemini",
            api_key="test-key"
        )
        provider = GeminiProvider(config)
        assert provider.supports_embeddings() is True

    def test_zhipu_supports_embeddings(self):
        """Test that Zhipu provider supports embeddings."""
        from llm_api_router.providers.zhipu import ZhipuProvider
        
        config = ProviderConfig(
            provider_type="zhipu",
            api_key="test-id.test-secret"
        )
        provider = ZhipuProvider(config)
        assert provider.supports_embeddings() is True

    def test_aliyun_supports_embeddings(self):
        """Test that Aliyun provider supports embeddings."""
        from llm_api_router.providers.aliyun import AliyunProvider
        
        config = ProviderConfig(
            provider_type="aliyun",
            api_key="test-key"
        )
        provider = AliyunProvider(config)
        assert provider.supports_embeddings() is True
