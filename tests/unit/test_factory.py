"""Unit tests for ProviderFactory."""
import pytest
from unittest.mock import Mock, patch
from llm_api_router.factory import ProviderFactory
from llm_api_router.types import ProviderConfig


class TestProviderFactory:
    """Test ProviderFactory class."""

    def test_factory_has_provider_mapping(self):
        """Test that factory has provider mapping."""
        assert hasattr(ProviderFactory, '_PROVIDER_MAPPING')
        assert isinstance(ProviderFactory._PROVIDER_MAPPING, dict)
        assert len(ProviderFactory._PROVIDER_MAPPING) > 0

    def test_factory_supports_common_providers(self):
        """Test that factory supports common providers."""
        common_providers = ["openai", "anthropic", "gemini", "deepseek"]
        
        for provider in common_providers:
            assert provider in ProviderFactory._PROVIDER_MAPPING

    def test_get_provider_with_unknown_type(self):
        """Test get_provider with unknown provider type."""
        config = ProviderConfig(
            provider_type="unknown_provider",
            api_key="test-key"
        )
        
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.get_provider(config)
        
        assert "不支持的提供商类型" in str(exc_info.value)

    def test_get_provider_with_valid_type(self):
        """Test get_provider with valid provider type."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key"
        )
        
        with patch('llm_api_router.factory.importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_provider_class = Mock()
            mock_provider_instance = Mock()
            
            mock_provider_class.return_value = mock_provider_instance
            mock_module.OpenAIProvider = mock_provider_class
            mock_import.return_value = mock_module
            
            provider = ProviderFactory.get_provider(config)
            
            assert provider == mock_provider_instance
            mock_import.assert_called_once_with("llm_api_router.providers.openai")
            mock_provider_class.assert_called_once_with(config)

    def test_get_provider_import_error(self):
        """Test get_provider with import error."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key"
        )
        
        with patch('llm_api_router.factory.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            with pytest.raises(ValueError) as exc_info:
                ProviderFactory.get_provider(config)
            
            assert "无法加载提供商模块" in str(exc_info.value)

    def test_get_provider_attribute_error(self):
        """Test get_provider with missing class in module."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key"
        )
        
        with patch('llm_api_router.factory.importlib.import_module') as mock_import:
            mock_module = Mock(spec=[])  # Empty spec, so getattr will raise AttributeError
            mock_import.return_value = mock_module
            
            with pytest.raises(ValueError) as exc_info:
                ProviderFactory.get_provider(config)
            
            assert "未在模块" in str(exc_info.value)

    def test_register_new_provider(self):
        """Test registering a new provider."""
        original_mapping = ProviderFactory._PROVIDER_MAPPING.copy()
        
        try:
            ProviderFactory.register(
                "custom_provider",
                "custom.module.path",
                "CustomProvider"
            )
            
            assert "custom_provider" in ProviderFactory._PROVIDER_MAPPING
            module_path, class_name = ProviderFactory._PROVIDER_MAPPING["custom_provider"]
            assert module_path == "custom.module.path"
            assert class_name == "CustomProvider"
        finally:
            # Restore original mapping
            ProviderFactory._PROVIDER_MAPPING = original_mapping

    def test_register_override_existing_provider(self):
        """Test that registering can override existing provider."""
        original_mapping = ProviderFactory._PROVIDER_MAPPING.copy()
        
        try:
            # Get original openai mapping
            original_openai = ProviderFactory._PROVIDER_MAPPING["openai"]
            
            # Override it
            ProviderFactory.register(
                "openai",
                "new.module.path",
                "NewProvider"
            )
            
            # Verify override
            module_path, class_name = ProviderFactory._PROVIDER_MAPPING["openai"]
            assert module_path == "new.module.path"
            assert class_name == "NewProvider"
            assert ProviderFactory._PROVIDER_MAPPING["openai"] != original_openai
        finally:
            # Restore original mapping
            ProviderFactory._PROVIDER_MAPPING = original_mapping

    def test_all_registered_providers_have_correct_format(self):
        """Test that all registered providers have correct tuple format."""
        for provider_type, mapping in ProviderFactory._PROVIDER_MAPPING.items():
            assert isinstance(mapping, tuple)
            assert len(mapping) == 2
            
            module_path, class_name = mapping
            assert isinstance(module_path, str)
            assert isinstance(class_name, str)
            assert len(module_path) > 0
            assert len(class_name) > 0
