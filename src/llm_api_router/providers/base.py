from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, AsyncIterator, Optional
import httpx
from ..types import (
    UnifiedRequest, UnifiedResponse, UnifiedChunk, ProviderConfig, RetryConfig,
    EmbeddingRequest, EmbeddingResponse
)
from ..exceptions import (
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RateLimitError,
    BadRequestError,
    ProviderError,
    TimeoutError,
    NetworkError,
)

class BaseProvider(ABC):
    """所有提供商适配器必须实现的抽象基类"""
    
    def __init__(self, config: ProviderConfig):
        """初始化provider"""
        self.config = config
        self.retry_config = config.retry_config or RetryConfig()
    
    @abstractmethod
    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        """将统一请求转换为提供商特定的请求 payload"""
        pass

    @abstractmethod
    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        """将提供商特定的响应转换为统一响应"""
        pass
    
    def handle_error_response(self, response: httpx.Response, provider_name: str) -> None:
        """
        统一处理错误响应，根据状态码抛出相应的异常
        
        Args:
            response: HTTP响应对象
            provider_name: 提供商名称
            
        Raises:
            相应的异常类型
        """
        # 尝试解析错误消息
        try:
            error_data = response.json()
            error_msg = self._extract_error_message(error_data)
        except Exception:
            error_msg = response.text or f"HTTP {response.status_code}"
        
        status_code = response.status_code
        
        # 根据状态码抛出不同的异常
        if status_code == 400:
            raise BadRequestError(
                f"{provider_name} Bad Request: {error_msg}",
                provider=provider_name,
                status_code=status_code
            )
        elif status_code == 401:
            raise AuthenticationError(
                f"{provider_name} Authentication Failed: {error_msg}",
                provider=provider_name,
                status_code=status_code
            )
        elif status_code == 403:
            raise PermissionError(
                f"{provider_name} Permission Denied: {error_msg}",
                provider=provider_name,
                status_code=status_code
            )
        elif status_code == 404:
            raise NotFoundError(
                f"{provider_name} Resource Not Found: {error_msg}",
                provider=provider_name,
                status_code=status_code
            )
        elif status_code == 429:
            raise RateLimitError(
                f"{provider_name} Rate Limit Exceeded: {error_msg}",
                provider=provider_name,
                status_code=status_code
            )
        elif status_code >= 500:
            raise ProviderError(
                f"{provider_name} Server Error: {error_msg}",
                provider=provider_name,
                status_code=status_code
            )
        else:
            raise ProviderError(
                f"{provider_name} Error {status_code}: {error_msg}",
                provider=provider_name,
                status_code=status_code
            )
    
    def _extract_error_message(self, error_data: Dict[str, Any]) -> str:
        """
        从错误数据中提取错误消息
        子类可以重写此方法以适配不同的错误格式
        
        Args:
            error_data: 错误响应的JSON数据
            
        Returns:
            错误消息字符串
        """
        # 常见的错误消息字段
        if isinstance(error_data, dict):
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            if "message" in error_data:
                return error_data["message"]
            if "detail" in error_data:
                return error_data["detail"]
        return str(error_data)
    
    def handle_request_error(self, error: Exception, provider_name: str) -> None:
        """
        处理请求异常（如网络错误、超时等）
        
        Args:
            error: 捕获的异常
            provider_name: 提供商名称
            
        Raises:
            相应的异常类型
        """
        if isinstance(error, httpx.TimeoutException):
            raise TimeoutError(
                f"{provider_name} Request Timeout: {str(error)}",
                provider=provider_name
            )
        elif isinstance(error, (httpx.NetworkError, httpx.ConnectError)):
            raise NetworkError(
                f"{provider_name} Network Error: {str(error)}",
                provider=provider_name
            )
        elif isinstance(error, httpx.RequestError):
            raise NetworkError(
                f"{provider_name} Request Error: {str(error)}",
                provider=provider_name
            )
        else:
            # 如果是已知的LLM Router异常，直接重新抛出
            raise
    
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

    # --- Embeddings Methods ---
    
    def supports_embeddings(self) -> bool:
        """检查此 provider 是否支持 embeddings API"""
        return False
    
    def create_embeddings(self, client: httpx.Client, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        创建文本嵌入（同步）
        
        Args:
            client: HTTP客户端
            request: 嵌入请求
            
        Returns:
            嵌入响应
            
        Raises:
            NotImplementedError: 如果 provider 不支持 embeddings
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")
    
    async def create_embeddings_async(self, client: httpx.AsyncClient, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        创建文本嵌入（异步）
        
        Args:
            client: 异步HTTP客户端
            request: 嵌入请求
            
        Returns:
            嵌入响应
            
        Raises:
            NotImplementedError: 如果 provider 不支持 embeddings
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")