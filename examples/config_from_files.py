"""Example: Load configuration from JSON and YAML files.

WARNING: This example hardcodes API keys for demonstration purposes only.
NEVER hardcode API keys in production code. Instead, use:
- Environment variables
- Secure configuration files (not committed to version control)
- Secret management services (AWS Secrets Manager, HashiCorp Vault, etc.)
- Operating system credential stores
"""
import json
import yaml
import tempfile
import os
from llm_api_router import ProviderConfig

# Example 1: Create and load from JSON file
print("Example 1: JSON Configuration File")
print("-" * 50)

json_config = {
    "provider_type": "openai",
    "api_key": "sk-test-key",
    "default_model": "gpt-4",
    "base_url": "https://api.openai.com/v1",
    "extra_headers": {
        "X-Organization": "my-org"
    }
}

# Create temporary JSON file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(json_config, f, indent=2)
    json_path = f.name

print(f"Created JSON config file: {json_path}")
print(f"Contents:")
with open(json_path, 'r') as f:
    print(f.read())

# Load configuration from JSON file
config_from_json = ProviderConfig.from_file(json_path)
print(f"\nLoaded configuration:")
print(f"  Provider: {config_from_json.provider_type}")
print(f"  Model: {config_from_json.default_model}")
print(f"  Base URL: {config_from_json.base_url}")
print(f"  Extra headers: {config_from_json.extra_headers}")

# Clean up
os.unlink(json_path)
print()

# Example 2: Create and load from YAML file
print("Example 2: YAML Configuration File")
print("-" * 50)

yaml_config = {
    "provider_type": "anthropic",
    "api_key": "sk-ant-test-key",
    "default_model": "claude-3-5-sonnet-20240620",
    "api_version": "2023-06-01"
}

# Create temporary YAML file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(yaml_config, f, default_flow_style=False)
    yaml_path = f.name

print(f"Created YAML config file: {yaml_path}")
print(f"Contents:")
with open(yaml_path, 'r') as f:
    print(f.read())

# Load configuration from YAML file
config_from_yaml = ProviderConfig.from_file(yaml_path)
print(f"Loaded configuration:")
print(f"  Provider: {config_from_yaml.provider_type}")
print(f"  Model: {config_from_yaml.default_model}")
print(f"  API Version: {config_from_yaml.api_version}")

# Clean up
os.unlink(yaml_path)
print()

# Example 3: Multi-provider configuration
print("Example 3: Multi-Provider Configuration")
print("-" * 50)

multi_provider_configs = {
    "openai": {
        "provider_type": "openai",
        "api_key": "sk-openai-key",
        "default_model": "gpt-4"
    },
    "anthropic": {
        "provider_type": "anthropic",
        "api_key": "sk-ant-key",
        "default_model": "claude-3-5-sonnet-20240620"
    },
    "gemini": {
        "provider_type": "gemini",
        "api_key": "gemini-key",
        "default_model": "gemini-1.5-flash"
    }
}

# Save as JSON
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(multi_provider_configs, f, indent=2)
    multi_config_path = f.name

print(f"Created multi-provider config file: {multi_config_path}")

# Load and use different providers
with open(multi_config_path, 'r') as f:
    all_configs = json.load(f)

for provider_name, provider_data in all_configs.items():
    config = ProviderConfig.from_dict(provider_data)
    print(f"\n{provider_name.upper()}:")
    print(f"  Type: {config.provider_type}")
    print(f"  Model: {config.default_model}")
    print(f"  API Key: {config.api_key[:10]}...")

# Clean up
os.unlink(multi_config_path)
