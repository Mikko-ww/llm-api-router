import pytest
from unittest.mock import Mock, MagicMock
import httpx
from llm_api_router.types import ProviderConfig, UnifiedRequest, Message
from llm_api_router.providers.openai import OpenAIProvider
from llm_api_router.exceptions import AuthenticationError

class TestOpenAIProvider:
    @pytest.fixture
    def config(self):
        return ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            default_model="gpt-3.5-turbo"
        )

    @pytest.fixture
    def provider(self, config):
        return OpenAIProvider(config)

    def test_convert_request(self, provider):
        req = UnifiedRequest(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7
        )
        data = provider.convert_request(req)
        assert data["model"] == "gpt-3.5-turbo"
        assert data["messages"] == [{"role": "user", "content": "hello"}]
        assert data["temperature"] == 0.7

    def test_convert_response(self, provider):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0613",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!",
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        resp = provider.convert_response(openai_resp)
        assert resp.id == "chatcmpl-123"
        assert resp.choices[0].message.content == "Hello there!"
        assert resp.usage.total_tokens == 21

    def test_send_request_success(self, provider):
        client = MagicMock(spec=httpx.Client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {}
        }
        client.post.return_value = mock_response

        req = UnifiedRequest(messages=[{"role": "user", "content": "hi"}])
        resp = provider.send_request(client, req)
        
        assert resp.choices[0].message.content == "hi"
        client.post.assert_called_once()

    def test_send_request_auth_error(self, provider):
        client = MagicMock(spec=httpx.Client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.text = "Invalid API Key"
        mock_response.json.side_effect = Exception("No JSON") # Simulate non-JSON error body
        client.post.return_value = mock_response

        req = UnifiedRequest(messages=[{"role": "user", "content": "hi"}])
        
        with pytest.raises(AuthenticationError):
            provider.send_request(client, req)
