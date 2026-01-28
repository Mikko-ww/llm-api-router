"""Unit tests for types module."""
import pytest
from llm_api_router.types import (
    ProviderConfig,
    Message,
    UnifiedRequest,
    Usage,
    Choice,
    UnifiedResponse,
    ChunkChoice,
    UnifiedChunk
)


class TestProviderConfig:
    """Test ProviderConfig dataclass."""
    
    def test_basic_config(self):
        """Test basic provider configuration."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        assert config.provider_type == "openai"
        assert config.api_key == "test-key"
        assert config.default_model == "gpt-3.5-turbo"
        assert config.base_url is None
        assert config.extra_headers == {}
    
    def test_config_with_optional_fields(self):
        """Test configuration with optional fields."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            base_url="https://custom.api.com",
            default_model="gpt-4",
            extra_headers={"X-Custom": "value"},
            api_version="2023-01-01"
        )
        assert config.base_url == "https://custom.api.com"
        assert config.extra_headers == {"X-Custom": "value"}
        assert config.api_version == "2023-01-01"


class TestMessage:
    """Test Message dataclass."""
    
    def test_message_creation(self):
        """Test message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestUnifiedRequest:
    """Test UnifiedRequest dataclass."""
    
    def test_basic_request(self):
        """Test basic unified request."""
        messages = [{"role": "user", "content": "Hello"}]
        request = UnifiedRequest(messages=messages, model="gpt-3.5-turbo")
        assert request.messages == messages
        assert request.model == "gpt-3.5-turbo"
        assert request.temperature == 1.0
        assert request.stream is False
    
    def test_request_with_all_params(self):
        """Test request with all parameters."""
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
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.stream is True
        assert request.top_p == 0.9
        assert request.stop == ["END"]


class TestUsage:
    """Test Usage dataclass."""
    
    def test_usage_creation(self):
        """Test usage creation."""
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
        """Test choice creation."""
        message = Message(role="assistant", content="Hello")
        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        assert choice.index == 0
        assert choice.message.content == "Hello"
        assert choice.finish_reason == "stop"


class TestUnifiedResponse:
    """Test UnifiedResponse dataclass."""
    
    def test_response_creation(self):
        """Test unified response creation."""
        message = Message(role="assistant", content="Hello")
        choice = Choice(index=0, message=message, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        
        response = UnifiedResponse(
            id="test-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[choice],
            usage=usage
        )
        
        assert response.id == "test-123"
        assert response.object == "chat.completion"
        assert response.created == 1234567890
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello"
        assert response.usage.total_tokens == 30


class TestChunkChoice:
    """Test ChunkChoice dataclass."""
    
    def test_chunk_choice_creation(self):
        """Test chunk choice creation."""
        delta = Message(role="assistant", content="Hi")
        chunk_choice = ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=None
        )
        assert chunk_choice.index == 0
        assert chunk_choice.delta.content == "Hi"
        assert chunk_choice.finish_reason is None


class TestUnifiedChunk:
    """Test UnifiedChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test unified chunk creation."""
        delta = Message(role="assistant", content="Hi")
        chunk_choice = ChunkChoice(index=0, delta=delta)
        
        chunk = UnifiedChunk(
            id="test-123",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[chunk_choice]
        )
        
        assert chunk.id == "test-123"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.created == 1234567890
        assert chunk.model == "gpt-3.5-turbo"
        assert len(chunk.choices) == 1
        assert chunk.choices[0].delta.content == "Hi"
