"""Unit tests for ProviderFactory."""
import pytest
from unittest.mock import Mock, patch
from llm_api_router.factory import ProviderFactory
from llm_api_router.types import ProviderConfig
from llm_api_router.providers.base import BaseProvider


class TestProviderFactory:
    """Test ProviderFactory class."""
    
    def test_get_provider_openai(self):
        """Test getting OpenAI provider."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )
        
        provider = ProviderFactory.get_provider(config)
        assert provider is not None
        assert isinstance(provider, BaseProvider)
    
    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider."""
        config = ProviderConfig(
            provider_type="anthropic",
            api_key="test-key",
            default_model="claude-3-5-sonnet-20240620"
        )
        
        provider = ProviderFactory.get_provider(config)
        assert provider is not None
        assert isinstance(provider, BaseProvider)
    
    def test_get_provider_gemini(self):
        """Test getting Gemini provider."""
        config = ProviderConfig(
            provider_type="gemini",
            api_key="test-key",
            default_model="gemini-1.5-flash"
        )
        
        provider = ProviderFactory.get_provider(config)
        assert provider is not None
        assert isinstance(provider, BaseProvider)
    
    def test_get_provider_unsupported(self):
        """Test getting unsupported provider raises error."""
        config = ProviderConfig(
            provider_type="unsupported",
            api_key="test-key",
            default_model="test-model"
        )
        
        with pytest.raises(ValueError, match="不支持的提供商类型"):
            ProviderFactory.get_provider(config)
    
    def test_register_new_provider(self):
        """Test registering a new provider."""
        # Register a test provider
        ProviderFactory.register(
            "test_provider",
            "llm_api_router.providers.openai",
            "OpenAIProvider"
        )
        
        # Check it's registered
        assert "test_provider" in ProviderFactory._PROVIDER_MAPPING
        
        # Clean up
        del ProviderFactory._PROVIDER_MAPPING["test_provider"]
    
    def test_get_all_supported_providers(self):
        """Test that all mapped providers are supported."""
        expected_providers = [
            "openai", "openrouter", "deepseek", "anthropic",
            "gemini", "zhipu", "aliyun", "xai", "ollama"
        ]
        
        for provider_type in expected_providers:
            assert provider_type in ProviderFactory._PROVIDER_MAPPING
