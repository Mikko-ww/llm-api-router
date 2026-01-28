"""Pytest configuration and shared fixtures."""
import pytest
from llm_api_router.types import ProviderConfig


@pytest.fixture
def openai_config():
    """OpenAI provider configuration fixture."""
    return ProviderConfig(
        provider_type="openai",
        api_key="sk-test-key",
        default_model="gpt-3.5-turbo"
    )


@pytest.fixture
def anthropic_config():
    """Anthropic provider configuration fixture."""
    return ProviderConfig(
        provider_type="anthropic",
        api_key="sk-ant-test-key",
        default_model="claude-3-5-sonnet-20240620"
    )


@pytest.fixture
def gemini_config():
    """Gemini provider configuration fixture."""
    return ProviderConfig(
        provider_type="gemini",
        api_key="test-key",
        default_model="gemini-1.5-flash"
    )


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_multi_turn_messages():
    """Sample multi-turn conversation for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"}
    ]
