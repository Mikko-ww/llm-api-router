"""Example: Configuration management using environment variables.

WARNING: This example hardcodes API keys for demonstration purposes only.
NEVER hardcode API keys in production code. Instead, use:
- Environment variables
- Secure configuration files (not committed to version control)
- Secret management services (AWS Secrets Manager, HashiCorp Vault, etc.)
- Operating system credential stores
"""
import os
from llm_api_router import Client, ProviderConfig

# Example 1: Load from environment variables with default naming
# Set environment variable: OPENAI_API_KEY=sk-...
os.environ['OPENAI_API_KEY'] = 'sk-test-key-from-env'

config = ProviderConfig.from_env(
    provider_type="openai",
    default_model="gpt-3.5-turbo"
)

print("Example 1: Default environment variable naming")
print(f"Provider: {config.provider_type}")
print(f"API Key: {config.api_key[:10]}...")
print(f"Model: {config.default_model}")
print()

# Example 2: Load from custom environment variable names
os.environ['MY_CUSTOM_API_KEY'] = 'sk-custom-key'
os.environ['MY_CUSTOM_BASE_URL'] = 'https://custom.api.com'

config2 = ProviderConfig.from_env(
    provider_type="openai",
    api_key_env="MY_CUSTOM_API_KEY",
    base_url_env="MY_CUSTOM_BASE_URL",
    default_model="gpt-4"
)

print("Example 2: Custom environment variable names")
print(f"Provider: {config2.provider_type}")
print(f"API Key: {config2.api_key[:10]}...")
print(f"Base URL: {config2.base_url}")
print(f"Model: {config2.default_model}")
print()

# Example 3: Load from dictionary
config_dict = {
    "provider_type": "anthropic",
    "api_key": "sk-ant-test-key",
    "default_model": "claude-3-5-sonnet-20240620",
    "extra_headers": {"X-Custom": "value"}
}

config3 = ProviderConfig.from_dict(config_dict)

print("Example 3: Load from dictionary")
print(f"Provider: {config3.provider_type}")
print(f"API Key: {config3.api_key[:10]}...")
print(f"Model: {config3.default_model}")
print(f"Extra headers: {config3.extra_headers}")
print()

# Example 4: Merge configurations
base_config = ProviderConfig(
    provider_type="openai",
    api_key="sk-base-key",
    default_model="gpt-3.5-turbo"
)

# Override model and add base_url
merged_config = base_config.merge({
    "default_model": "gpt-4",
    "base_url": "https://custom.api.com"
})

print("Example 4: Merge configurations")
print(f"Original model: {base_config.default_model}")
print(f"Merged model: {merged_config.default_model}")
print(f"Merged base URL: {merged_config.base_url}")
print()

# Example 5: Validate configuration
try:
    valid_config = ProviderConfig(
        provider_type="openai",
        api_key="sk-test-key",
        default_model="gpt-3.5-turbo"
    )
    valid_config.validate()
    print("Example 5: Configuration is valid ✓")
except ValueError as e:
    print(f"Configuration error: {e}")

# Example 6: Invalid configuration (empty api_key)
try:
    invalid_config = ProviderConfig(
        provider_type="openai",
        api_key=""
    )
    invalid_config.validate()
except ValueError as e:
    print(f"Example 6: Validation caught error: {e}")
