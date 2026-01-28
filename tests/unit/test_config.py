"""Unit tests for configuration management."""
import pytest
import os
import json
import yaml
import tempfile
from pathlib import Path
from llm_api_router.types import ProviderConfig


class TestProviderConfigFromEnv:
    """Test ProviderConfig.from_env() method."""
    
    def test_from_env_with_defaults(self, monkeypatch):
        """Test loading from environment with default variable names."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        config = ProviderConfig.from_env(
            provider_type="openai",
            default_model="gpt-3.5-turbo"
        )
        
        assert config.provider_type == "openai"
        assert config.api_key == "sk-test-key"
        assert config.default_model == "gpt-3.5-turbo"
    
    def test_from_env_with_custom_var(self, monkeypatch):
        """Test loading from environment with custom variable name."""
        monkeypatch.setenv("MY_API_KEY", "sk-custom-key")
        
        config = ProviderConfig.from_env(
            provider_type="openai",
            api_key_env="MY_API_KEY",
            default_model="gpt-4"
        )
        
        assert config.api_key == "sk-custom-key"
        assert config.default_model == "gpt-4"
    
    def test_from_env_with_base_url(self, monkeypatch):
        """Test loading from environment with base URL."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("MY_BASE_URL", "https://custom.api.com")
        
        config = ProviderConfig.from_env(
            provider_type="openai",
            base_url_env="MY_BASE_URL",
            default_model="gpt-3.5-turbo"
        )
        
        assert config.base_url == "https://custom.api.com"
    
    def test_from_env_missing_key(self):
        """Test error when API key environment variable is missing."""
        with pytest.raises(ValueError, match="Environment variable.*not found"):
            ProviderConfig.from_env(
                provider_type="openai",
                api_key_env="NONEXISTENT_KEY"
            )
    
    def test_from_env_with_extra_params(self, monkeypatch):
        """Test loading from environment with extra parameters."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        config = ProviderConfig.from_env(
            provider_type="openai",
            default_model="gpt-3.5-turbo",
            extra_headers={"X-Custom": "value"},
            api_version="2023-01-01"
        )
        
        assert config.extra_headers == {"X-Custom": "value"}
        assert config.api_version == "2023-01-01"


class TestProviderConfigFromDict:
    """Test ProviderConfig.from_dict() method."""
    
    def test_from_dict_minimal(self):
        """Test loading from dict with minimal fields."""
        data = {
            "provider_type": "openai",
            "api_key": "sk-test-key"
        }
        
        config = ProviderConfig.from_dict(data)
        
        assert config.provider_type == "openai"
        assert config.api_key == "sk-test-key"
        assert config.base_url is None
        assert config.default_model is None
    
    def test_from_dict_complete(self):
        """Test loading from dict with all fields."""
        data = {
            "provider_type": "openai",
            "api_key": "sk-test-key",
            "base_url": "https://api.openai.com",
            "default_model": "gpt-4",
            "extra_headers": {"X-Custom": "value"},
            "api_version": "2023-01-01"
        }
        
        config = ProviderConfig.from_dict(data)
        
        assert config.provider_type == "openai"
        assert config.api_key == "sk-test-key"
        assert config.base_url == "https://api.openai.com"
        assert config.default_model == "gpt-4"
        assert config.extra_headers == {"X-Custom": "value"}
        assert config.api_version == "2023-01-01"
    
    def test_from_dict_missing_required(self):
        """Test error when required fields are missing."""
        data = {"provider_type": "openai"}
        
        with pytest.raises(ValueError, match="Missing required field"):
            ProviderConfig.from_dict(data)


class TestProviderConfigFromFile:
    """Test ProviderConfig.from_file() method."""
    
    def test_from_json_file(self):
        """Test loading configuration from JSON file."""
        data = {
            "provider_type": "openai",
            "api_key": "sk-test-key",
            "default_model": "gpt-3.5-turbo"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            config = ProviderConfig.from_file(temp_path)
            
            assert config.provider_type == "openai"
            assert config.api_key == "sk-test-key"
            assert config.default_model == "gpt-3.5-turbo"
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        data = {
            "provider_type": "anthropic",
            "api_key": "sk-ant-test-key",
            "default_model": "claude-3-5-sonnet-20240620"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name
        
        try:
            config = ProviderConfig.from_file(temp_path)
            
            assert config.provider_type == "anthropic"
            assert config.api_key == "sk-ant-test-key"
            assert config.default_model == "claude-3-5-sonnet-20240620"
        finally:
            os.unlink(temp_path)
    
    def test_from_yml_file(self):
        """Test loading configuration from .yml file."""
        data = {
            "provider_type": "gemini",
            "api_key": "test-key",
            "default_model": "gemini-1.5-flash"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name
        
        try:
            config = ProviderConfig.from_file(temp_path)
            
            assert config.provider_type == "gemini"
            assert config.api_key == "test-key"
        finally:
            os.unlink(temp_path)
    
    def test_from_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ProviderConfig.from_file("/nonexistent/file.json")
    
    def test_from_file_unsupported_format(self):
        """Test error when file format is unsupported."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                ProviderConfig.from_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestProviderConfigMerge:
    """Test ProviderConfig.merge() method."""
    
    def test_merge_basic(self):
        """Test basic config merge."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        new_config = config.merge({"default_model": "gpt-4"})
        
        # Original config should be unchanged
        assert config.default_model == "gpt-3.5-turbo"
        # New config should have merged values
        assert new_config.default_model == "gpt-4"
        assert new_config.api_key == "sk-test-key"
    
    def test_merge_multiple_fields(self):
        """Test merging multiple fields."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        new_config = config.merge({
            "default_model": "gpt-4",
            "base_url": "https://custom.api.com",
            "api_version": "2023-01-01"
        })
        
        assert new_config.default_model == "gpt-4"
        assert new_config.base_url == "https://custom.api.com"
        assert new_config.api_version == "2023-01-01"
    
    def test_merge_preserves_original(self):
        """Test that merge doesn't modify original config."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            extra_headers={"X-Original": "value"}
        )
        
        new_config = config.merge({"extra_headers": {"X-New": "value"}})
        
        # Original should be unchanged
        assert config.extra_headers == {"X-Original": "value"}
        # New config has new value
        assert new_config.extra_headers == {"X-New": "value"}


class TestProviderConfigValidate:
    """Test ProviderConfig.validate() method."""
    
    def test_validate_success(self):
        """Test validation of valid config."""
        config = ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-3.5-turbo"
        )
        
        assert config.validate() is True
    
    def test_validate_empty_provider_type(self):
        """Test validation fails with empty provider_type."""
        config = ProviderConfig(
            provider_type="",
            api_key="sk-test-key"
        )
        
        with pytest.raises(ValueError, match="provider_type cannot be empty"):
            config.validate()
    
    def test_validate_empty_api_key(self):
        """Test validation fails with empty api_key."""
        config = ProviderConfig(
            provider_type="openai",
            api_key=""
        )
        
        with pytest.raises(ValueError, match="api_key cannot be empty"):
            config.validate()
    
    def test_validate_azure_missing_version(self):
        """Test validation fails for Azure without api_version."""
        config = ProviderConfig(
            provider_type="azure",
            api_key="sk-test-key"
        )
        
        with pytest.raises(ValueError, match="api_version is required for Azure"):
            config.validate()
    
    def test_validate_azure_with_version(self):
        """Test validation succeeds for Azure with api_version."""
        config = ProviderConfig(
            provider_type="azure",
            api_key="sk-test-key",
            api_version="2023-01-01"
        )
        
        assert config.validate() is True
