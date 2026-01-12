import pytest
from unittest.mock import Mock, MagicMock
import httpx
from llm_api_router.client import AsyncClient
from llm_api_router.types import ProviderConfig, UnifiedResponse, UnifiedChunk, Choice, Message, Usage, ChunkChoice

@pytest.fixture
def config():
    return ProviderConfig(
        provider_type="openai",
        api_key="test-key",
        default_model="gpt-3.5-turbo"
    )

@pytest.mark.asyncio
async def test_async_client_create_success(config):
    # Mocking httpx.AsyncClient behavior inside OpenAIProvider
    
    # We need to patch the provider's send_request_async, 
    # but since the client initializes the provider internally, it's easier to mock at the httpx level if we want integration-style unit tests,
    # OR we can just verify the AsyncClient delegates correctly.
    
    # Let's mock the httpx.AsyncClient response to test the full flow
    
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test",
        "object": "chat.completion",
        "created": 123,
        "model": "gpt-3.5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "async hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    }
    
    async with AsyncClient(config) as client:
        # We need to mock the internal _http_client's post method
        client._http_client.post = MagicMock()
        client._http_client.post.return_value = mock_response

        # Need to fix await on mock:
        async def async_return_value(*args, **kwargs):
            return mock_response
        client._http_client.post.side_effect = async_return_value

        resp = await client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}]
        )
        
        assert isinstance(resp, UnifiedResponse)
        assert resp.choices[0].message.content == "async hi"

@pytest.mark.asyncio
async def test_async_client_stream_success(config):
    # Mock streaming response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    
    lines = [
        'data: {"id":"1","choices":[{"index":0,"delta":{"content":"A"}}]}',
        'data: {"id":"1","choices":[{"index":0,"delta":{"content":"B"}}]}',
        'data: [DONE]'
    ]
    
    async def async_gen():
        for line in lines:
            yield line
            
    mock_response.aiter_lines.return_value = async_gen()
    
    async with AsyncClient(config) as client:
        # Mock stream context manager
        # client.stream("POST", ...) returns an async context manager
        
        class MockStreamContext:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, exc_type, exc, tb):
                pass
        
        client._http_client.stream = MagicMock(return_value=MockStreamContext())
        
        chunks = []
        stream_resp = await client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            stream=True
        )
        
        async for chunk in stream_resp:
            chunks.append(chunk)
            
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "A"
        assert chunks[1].choices[0].delta.content == "B"
