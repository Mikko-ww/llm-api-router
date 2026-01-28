"""Unit tests for types module."""
import pytest
from llm_api_router.types import (
    ProviderConfig,
    Message,
    UnifiedRequest,
    UnifiedResponse,
    Usage,
    Choice,
    UnifiedChunk,
    ChunkChoice
)


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_provider_config_creation(self):
        """Test creating a ProviderConfig instance."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            default_model="gpt-3.5-turbo"
        )
        
        assert config.provider_type == "openai"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.default_model == "gpt-3.5-turbo"

    def test_provider_config_with_extra_headers(self):
        """Test ProviderConfig with extra headers."""
        config = ProviderConfig(
            provider_type="custom",
            api_key="test-key",
            extra_headers={"X-Custom-Header": "value"}
        )
        
        assert config.extra_headers == {"X-Custom-Header": "value"}

    def test_provider_config_defaults(self):
        """Test ProviderConfig default values."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key"
        )
        
        assert config.base_url is None
        assert config.default_model is None
        assert config.extra_headers == {}
        assert config.api_version is None


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self):
        """Test creating a Message instance."""
        message = Message(role="user", content="Hello")
        
        assert message.role == "user"
        assert message.content == "Hello"

    def test_message_assistant_role(self):
        """Test Message with assistant role."""
        message = Message(role="assistant", content="Hi there!")
        
        assert message.role == "assistant"
        assert message.content == "Hi there!"


class TestUnifiedRequest:
    """Test UnifiedRequest dataclass."""

    def test_unified_request_basic(self):
        """Test basic UnifiedRequest creation."""
        messages = [{"role": "user", "content": "Hello"}]
        request = UnifiedRequest(messages=messages)
        
        assert request.messages == messages
        assert request.model is None
        assert request.temperature == 1.0
        assert request.max_tokens is None
        assert request.stream is False

    def test_unified_request_with_parameters(self):
        """Test UnifiedRequest with all parameters."""
        messages = [{"role": "user", "content": "Hello"}]
        request = UnifiedRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            stream=True,
            top_p=0.9,
            stop=["END"]
        )
        
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.stream is True
        assert request.top_p == 0.9
        assert request.stop == ["END"]


class TestUsage:
    """Test Usage dataclass."""

    def test_usage_creation(self):
        """Test creating a Usage instance."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestChoice:
    """Test Choice dataclass."""

    def test_choice_creation(self):
        """Test creating a Choice instance."""
        message = Message(role="assistant", content="Hello")
        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        
        assert choice.index == 0
        assert choice.message == message
        assert choice.finish_reason == "stop"


class TestUnifiedResponse:
    """Test UnifiedResponse dataclass."""

    def test_unified_response_creation(self):
        """Test creating a UnifiedResponse instance."""
        message = Message(role="assistant", content="Hello")
        choice = Choice(index=0, message=message, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        response = UnifiedResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[choice],
            usage=usage
        )
        
        assert response.id == "test-id"
        assert response.object == "chat.completion"
        assert response.created == 1234567890
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        assert response.usage == usage


class TestChunkChoice:
    """Test ChunkChoice dataclass."""

    def test_chunk_choice_creation(self):
        """Test creating a ChunkChoice instance."""
        delta = Message(role="assistant", content="Hello")
        choice = ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=None
        )
        
        assert choice.index == 0
        assert choice.delta == delta
        assert choice.finish_reason is None


class TestUnifiedChunk:
    """Test UnifiedChunk dataclass."""

    def test_unified_chunk_creation(self):
        """Test creating a UnifiedChunk instance."""
        delta = Message(role="assistant", content="Hello")
        choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
        
        chunk = UnifiedChunk(
            id="chunk-id",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[choice]
        )
        
        assert chunk.id == "chunk-id"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.created == 1234567890
        assert chunk.model == "gpt-3.5-turbo"
        assert len(chunk.choices) == 1
        assert chunk.choices[0] == choice
