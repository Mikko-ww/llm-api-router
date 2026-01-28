from typing import List, Dict, Optional, Union, Iterator, AsyncIterator
import httpx
from .types import ProviderConfig, UnifiedRequest, UnifiedResponse, UnifiedChunk

from .exceptions import LLMRouterError
from .factory import ProviderFactory

# --- Synchronous Classes ---

class Completions:
    def __init__(self, client: "Client"):
        self._client = client

    def create(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Union[UnifiedResponse, Iterator[UnifiedChunk]]:
        """
        创建聊天补全
        """
        request = UnifiedRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            top_p=top_p,
            stop=stop
        )
        
        if stream:
            return self._client._provider.stream_request(self._client._http_client, request)
        else:
            return self._client._provider.send_request(self._client._http_client, request)

class Chat:
    def __init__(self, client: "Client"):
        self.completions = Completions(client)

class Client:
    """同步客户端"""
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._http_client = httpx.Client(timeout=config.timeout)
        self._provider = self._get_provider(config)
        self.chat = Chat(self)

    def _get_provider(self, config: ProviderConfig):
        return ProviderFactory.get_provider(config)

    def close(self):
        self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# --- Asynchronous Classes ---

class AsyncCompletions:
    def __init__(self, client: "AsyncClient"):
        self._client = client

    async def create(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Union[UnifiedResponse, AsyncIterator[UnifiedChunk]]:
        """
        创建聊天补全 (异步)
        """
        request = UnifiedRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            top_p=top_p,
            stop=stop
        )
        
        if stream:
            return self._client._provider.stream_request_async(self._client._http_client, request)
        else:
            return await self._client._provider.send_request_async(self._client._http_client, request)

class AsyncChat:
    def __init__(self, client: "AsyncClient"):
        self.completions = AsyncCompletions(client)

class AsyncClient:
    """异步客户端"""
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._http_client = httpx.AsyncClient(timeout=config.timeout)
        self._provider = self._get_provider(config)
        self.chat = AsyncChat(self)

    def _get_provider(self, config: ProviderConfig):
        return ProviderFactory.get_provider(config)

    async def close(self):
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
