from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, AsyncIterator
import httpx
from ..types import UnifiedRequest, UnifiedResponse, UnifiedChunk

class BaseProvider(ABC):
    """所有提供商适配器必须实现的抽象基类"""
    
    @abstractmethod
    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        """将统一请求转换为提供商特定的请求 payload"""
        pass

    @abstractmethod
    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        """将提供商特定的响应转换为统一响应"""
        pass
    
    @abstractmethod
    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        """执行同步请求"""
        pass

    @abstractmethod
    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        """执行异步请求"""
        pass

    @abstractmethod
    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        """执行同步流式请求"""
        pass

    @abstractmethod
    def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        """执行异步流式请求"""
        pass