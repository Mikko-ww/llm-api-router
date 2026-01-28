"""Pytest configuration and shared fixtures."""
import pytest
from typing import Dict, Any
from llm_api_router.types import ProviderConfig, UnifiedResponse, Message, Choice, Usage


@pytest.fixture
def sample_provider_config() -> ProviderConfig:
    """Sample provider configuration for testing."""
    return ProviderConfig(
        provider_type="openai",
        api_key="test-api-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-3.5-turbo"
    )


@pytest.fixture
def sample_messages() -> list[Dict[str, str]]:
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_unified_response() -> UnifiedResponse:
    """Sample unified response for testing."""
    return UnifiedResponse(
        id="chatcmpl-test123",
        object="chat.completion",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="I'm doing well, thank you!"),
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
def mock_http_response_success() -> Dict[str, Any]:
    """Mock successful HTTP response data."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }


@pytest.fixture
def mock_stream_chunks() -> list[str]:
    """Mock SSE stream chunks."""
    return [
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        'data: [DONE]\n\n'
    ]
