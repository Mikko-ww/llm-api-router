"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Check if CLI dependencies are available
try:
    from typer.testing import CliRunner
    from llm_api_router.cli.main import app
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


pytestmark = pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI dependencies not installed")


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestVersion:
    """Tests for version command."""
    
    def test_version_flag(self, runner):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "llm-api-router" in result.stdout
    
    def test_version_short_flag(self, runner):
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0


class TestValidate:
    """Tests for validate command."""
    
    def test_validate_valid_json(self, runner):
        config = {
            "provider_type": "openai",
            "api_key": "sk-test",
            "default_model": "gpt-3.5-turbo"
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            
            result = runner.invoke(app, ["validate", f.name])
            assert result.exit_code == 0
            assert "valid" in result.stdout.lower()
    
    def test_validate_missing_provider_type(self, runner):
        config = {"api_key": "sk-test"}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            
            result = runner.invoke(app, ["validate", f.name])
            assert result.exit_code == 1
            assert "provider_type" in result.stdout
    
    def test_validate_file_not_found(self, runner):
        result = runner.invoke(app, ["validate", "/nonexistent/file.json"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
    
    def test_validate_invalid_json(self, runner):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            f.flush()
            
            result = runner.invoke(app, ["validate", f.name])
            assert result.exit_code == 1
    
    def test_validate_verbose(self, runner):
        config = {"provider_type": "openai", "api_key": "sk-test"}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            
            result = runner.invoke(app, ["validate", f.name, "--verbose"])
            assert result.exit_code == 0
            # Should show config contents
            assert "provider_type" in result.stdout


class TestModels:
    """Tests for models command."""
    
    def test_models_openai(self, runner):
        result = runner.invoke(app, ["models", "openai"])
        assert result.exit_code == 0
        assert "gpt-4" in result.stdout
        assert "gpt-3.5-turbo" in result.stdout
    
    def test_models_anthropic(self, runner):
        result = runner.invoke(app, ["models", "anthropic"])
        assert result.exit_code == 0
        assert "claude" in result.stdout.lower()
    
    def test_models_unknown_provider(self, runner):
        result = runner.invoke(app, ["models", "unknown_provider"])
        assert result.exit_code == 1


class TestTest:
    """Tests for test command."""
    
    @patch("llm_api_router.Client")
    def test_test_success(self, mock_client_class, runner):
        # Mock the client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(app, ["test", "openai", "--api-key", "sk-test"])
        assert result.exit_code == 0
        assert "successful" in result.stdout.lower()
    
    def test_test_no_api_key(self, runner):
        # Clear any env vars
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["test", "openai"])
            assert result.exit_code == 1
            assert "api key" in result.stdout.lower()


class TestHelp:
    """Tests for help output."""
    
    def test_main_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "test" in result.stdout
        assert "validate" in result.stdout
        assert "benchmark" in result.stdout
        assert "models" in result.stdout
        assert "chat" in result.stdout
    
    def test_test_help(self, runner):
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        assert "provider" in result.stdout.lower()
    
    def test_benchmark_help(self, runner):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "requests" in result.stdout.lower()
