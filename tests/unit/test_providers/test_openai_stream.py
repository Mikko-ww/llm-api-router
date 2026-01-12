import pytest
from unittest.mock import Mock, MagicMock
import httpx
from llm_api_router.types import ProviderConfig, UnifiedRequest
from llm_api_router.providers.openai import OpenAIProvider

class TestOpenAIStream:
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

    def test_stream_request_success(self, provider):
        client = MagicMock(spec=httpx.Client)
        
        # Mocking stream context manager
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        
        # SSE data lines
        lines = [
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":169,"model":"gpt-3.5","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":169,"model":"gpt-3.5","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}',
            'data: [DONE]'
        ]
        mock_response.iter_lines.return_value = lines
        
        # client.stream() returns a context manager that yields mock_response
        client.stream.return_value.__enter__.return_value = mock_response

        req = UnifiedRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
        chunks = list(provider.stream_request(client, req))
        
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hel"
        assert chunks[1].choices[0].delta.content == "lo"
