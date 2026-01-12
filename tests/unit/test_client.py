import pytest
from llm_api_router.client import Client
from llm_api_router.types import ProviderConfig
from llm_api_router.providers.openai import OpenAIProvider

def test_client_init_openai():
    config = ProviderConfig(provider_type="openai", api_key="sk-test")
    client = Client(config)
    assert isinstance(client._provider, OpenAIProvider)
    assert client._provider.config.api_key == "sk-test"

def test_client_init_unknown_provider():
    config = ProviderConfig(provider_type="unknown", api_key="sk-test")
    with pytest.raises(ValueError, match="不支持的提供商类型"):
        Client(config)

def test_client_context_manager():
    config = ProviderConfig(provider_type="openai", api_key="sk-test")
    with Client(config) as client:
        assert isinstance(client._provider, OpenAIProvider)
