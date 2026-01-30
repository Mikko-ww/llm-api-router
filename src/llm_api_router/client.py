from typing import List, Dict, Optional, Union, Iterator, AsyncIterator
import httpx
import logging
from .types import (
    ProviderConfig, UnifiedRequest, UnifiedResponse, UnifiedChunk,
    EmbeddingRequest, EmbeddingResponse
)

from .exceptions import LLMRouterError
from .factory import ProviderFactory
from .logging_config import setup_logging, generate_request_id, get_logger

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
            stop=stop,
            request_id=generate_request_id()
        )
        
        if stream:
            return self._client._provider.stream_request(self._client._http_client, request)
        else:
            return self._client._provider.send_request(self._client._http_client, request)

class Chat:
    def __init__(self, client: "Client"):
        self.completions = Completions(client)


class Embeddings:
    """Embeddings API 包装类"""
    
    def __init__(self, client: "Client"):
        self._client = client
    
    def create(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> EmbeddingResponse:
        """
        创建文本嵌入
        
        Args:
            input: 要嵌入的文本，可以是单个字符串或字符串列表
            model: 使用的模型，如果不指定则使用 provider 默认模型
            encoding_format: 编码格式 ("float" 或 "base64")，默认为 "float"
            dimensions: 输出向量维度（仅部分模型支持）
            
        Returns:
            EmbeddingResponse: 包含嵌入向量的响应
            
        Raises:
            NotImplementedError: 如果当前 provider 不支持 embeddings
        """
        # 统一转换为列表
        if isinstance(input, str):
            input = [input]
        
        request = EmbeddingRequest(
            input=input,
            model=model,
            encoding_format=encoding_format,
            dimensions=dimensions
        )
        
        return self._client._provider.create_embeddings(self._client._http_client, request)


class Client:
    """同步客户端"""
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._http_client = httpx.Client(timeout=config.timeout)
        self._provider = self._get_provider(config)
        self.chat = Chat(self)
        self.embeddings = Embeddings(self)
        
        # Initialize logging only if custom config is provided or not yet configured
        if config.log_config or not logging.getLogger("llm_api_router").handlers:
            self._logger = setup_logging(config.log_config)

    def _get_provider(self, config: ProviderConfig):
        return ProviderFactory.get_provider(config)
    
    def get_metrics_collector(self):
        """
        Get the metrics collector instance used by this client
        
        Returns:
            MetricsCollector instance or None if metrics are disabled
        """
        if hasattr(self._provider, '_metrics_collector'):
            return self._provider._metrics_collector
        return None
    
    def get_metrics(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Get raw metrics, optionally filtered by provider and/or model
        
        Args:
            provider: Filter by provider (optional)
            model: Filter by model (optional)
            
        Returns:
            List of RequestMetrics or empty list if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.get_metrics(provider, model)
        return []
    
    def get_aggregated_metrics(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Get aggregated metrics grouped by provider and model
        
        Args:
            provider: Filter by provider (optional)
            model: Filter by model (optional)
            
        Returns:
            List of AggregatedMetrics or empty list if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.get_aggregated_metrics(provider, model)
        return []
    
    def export_metrics_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format
        
        Returns:
            Prometheus-formatted metrics string or empty string if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.export_prometheus()
        return ""
    
    def compare_providers(self):
        """
        Compare performance across providers
        
        Returns:
            List of provider comparison data or empty list if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.compare_providers()
        return []

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


class AsyncEmbeddings:
    """异步 Embeddings API 包装类"""
    
    def __init__(self, client: "AsyncClient"):
        self._client = client
    
    async def create(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> EmbeddingResponse:
        """
        创建文本嵌入 (异步)
        
        Args:
            input: 要嵌入的文本，可以是单个字符串或字符串列表
            model: 使用的模型，如果不指定则使用 provider 默认模型
            encoding_format: 编码格式 ("float" 或 "base64")，默认为 "float"
            dimensions: 输出向量维度（仅部分模型支持）
            
        Returns:
            EmbeddingResponse: 包含嵌入向量的响应
            
        Raises:
            NotImplementedError: 如果当前 provider 不支持 embeddings
        """
        # 统一转换为列表
        if isinstance(input, str):
            input = [input]
        
        request = EmbeddingRequest(
            input=input,
            model=model,
            encoding_format=encoding_format,
            dimensions=dimensions
        )
        
        return await self._client._provider.create_embeddings_async(self._client._http_client, request)


class AsyncClient:
    """异步客户端"""
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._http_client = httpx.AsyncClient(timeout=config.timeout)
        self._provider = self._get_provider(config)
        self.chat = AsyncChat(self)
        self.embeddings = AsyncEmbeddings(self)
        
        # Initialize logging only if custom config is provided or not yet configured
        if config.log_config or not logging.getLogger("llm_api_router").handlers:
            self._logger = setup_logging(config.log_config)

    def _get_provider(self, config: ProviderConfig):
        return ProviderFactory.get_provider(config)
    
    def get_metrics_collector(self):
        """
        Get the metrics collector instance used by this client
        
        Returns:
            MetricsCollector instance or None if metrics are disabled
        """
        if hasattr(self._provider, '_metrics_collector'):
            return self._provider._metrics_collector
        return None
    
    def get_metrics(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Get raw metrics, optionally filtered by provider and/or model
        
        Args:
            provider: Filter by provider (optional)
            model: Filter by model (optional)
            
        Returns:
            List of RequestMetrics or empty list if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.get_metrics(provider, model)
        return []
    
    def get_aggregated_metrics(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Get aggregated metrics grouped by provider and model
        
        Args:
            provider: Filter by provider (optional)
            model: Filter by model (optional)
            
        Returns:
            List of AggregatedMetrics or empty list if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.get_aggregated_metrics(provider, model)
        return []
    
    def export_metrics_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format
        
        Returns:
            Prometheus-formatted metrics string or empty string if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.export_prometheus()
        return ""
    
    def compare_providers(self):
        """
        Compare performance across providers
        
        Returns:
            List of provider comparison data or empty list if metrics are disabled
        """
        collector = self.get_metrics_collector()
        if collector:
            return collector.compare_providers()
        return []

    async def close(self):
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
