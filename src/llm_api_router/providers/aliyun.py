from typing import Dict, Any, Iterator, AsyncIterator, List, Optional
import httpx
import json
from ..types import (
    UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, 
    ProviderConfig, ChunkChoice, EmbeddingRequest, EmbeddingResponse,
    Embedding, EmbeddingUsage
)
from ..exceptions import StreamError
from ..retry import with_retry, with_retry_async
from .base import BaseProvider

class AliyunProvider(BaseProvider):
    """Alibaba Cloud (DashScope/Qwen) Provider Adapter"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation").rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **config.extra_headers
        }

    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        # DashScope Structure: 
        # { 
        #   "model": "...", 
        #   "input": { "messages": [...] }, 
        #   "parameters": { "result_format": "message", ... } 
        # }
        
        parameters = {
            "result_format": "message"
        }
        
        if request.stream:
             parameters["incremental_output"] = True # Enable delta streaming
             
        if request.temperature is not None:
            parameters["temperature"] = request.temperature
        if request.max_tokens is not None:
             parameters["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            parameters["top_p"] = request.top_p
        if request.stop is not None:
            parameters["stop"] = request.stop

        data: Dict[str, Any] = {
            "model": request.model or self.config.default_model or "qwen-turbo",
            "input": {
                "messages": request.messages
            },
            "parameters": parameters
        }
            
        return data

    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        output = provider_response.get("output", {})
        choices = []
        for c in output.get("choices", []):
            msg_data = c.get("message", {})
            message = Message(role=msg_data.get("role", ""), content=msg_data.get("content", ""))
            choices.append(Choice(
                index=c.get("index", 0),
                message=message,
                finish_reason=c.get("finish_reason", "")
            ))
        
        usage_data = provider_response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

        return UnifiedResponse(
            id=provider_response.get("request_id", ""),
            object="chat.completion",
            created=0,
            model=provider_response.get("model", ""), # Dashscope sometimes returns request_id at top
            choices=choices,
            usage=usage
        )
    
    def _convert_chunk(self, chunk_data: Dict[str, Any]) -> UnifiedChunk:
        output = chunk_data.get("output", {})
        choices = []
        for c in output.get("choices", []):
            msg_data = c.get("message", {})
            # When incremental_output=True, 'content' is the delta
            delta = Message(role=msg_data.get("role", ""), content=msg_data.get("content", ""))
            choices.append(ChunkChoice(
                index=c.get("index", 0),
                delta=delta,
                finish_reason=c.get("finish_reason")
            ))
            
        return UnifiedChunk(
            id=chunk_data.get("request_id", ""),
            object="chat.completion.chunk",
            created=0,
            model="",
            choices=choices
        )

    @with_retry()
    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        # DashScope is strictly POST
        url = self.base_url # base_url is full path
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Aliyun")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Aliyun")

    @with_retry_async()
    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = self.base_url
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Aliyun")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Aliyun")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = self.base_url
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "Aliyun")
                
                for line in response.iter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        # DashScope doesn't use [DONE] typically? It just stops.
                        # But standard SSE might.
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="Aliyun")
        except httpx.RequestError as e:
            self.handle_request_error(e, "Aliyun")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = self.base_url
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "Aliyun")
                
                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="Aliyun")
        except httpx.RequestError as e:
            self.handle_request_error(e, "Aliyun")

    # --- Embeddings Implementation ---
    
    def supports_embeddings(self) -> bool:
        """Aliyun (DashScope) 支持 embeddings API"""
        return True
    
    def _convert_embedding_request(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """将嵌入请求转换为 DashScope 特定的请求格式"""
        # DashScope embedding format
        # {
        #   "model": "text-embedding-v2",
        #   "input": { "texts": ["text1", "text2"] },
        #   "parameters": { "dimension": 512 }  # optional
        # }
        data: Dict[str, Any] = {
            "model": request.model or "text-embedding-v2",
            "input": {
                "texts": request.input
            }
        }
        if request.dimensions is not None:
            data["parameters"] = {"dimension": request.dimensions}
        return data
    
    def _convert_embedding_response(self, provider_response: Dict[str, Any]) -> EmbeddingResponse:
        """将 DashScope 嵌入响应转换为统一格式"""
        output = provider_response.get("output", {})
        embeddings_data = output.get("embeddings", [])
        
        embeddings = []
        for item in embeddings_data:
            embeddings.append(Embedding(
                index=item.get("text_index", 0),
                embedding=item.get("embedding", []),
                object="embedding"
            ))
        
        usage_data = provider_response.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=usage_data.get("total_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return EmbeddingResponse(
            data=embeddings,
            model=provider_response.get("model", ""),
            usage=usage,
            object="list"
        )
    
    @with_retry()
    def create_embeddings(self, client: httpx.Client, request: EmbeddingRequest) -> EmbeddingResponse:
        """创建文本嵌入（同步）"""
        payload = self._convert_embedding_request(request)
        # DashScope embedding endpoint is different
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Aliyun")
            return self._convert_embedding_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Aliyun")
    
    @with_retry_async()
    async def create_embeddings_async(self, client: httpx.AsyncClient, request: EmbeddingRequest) -> EmbeddingResponse:
        """创建文本嵌入（异步）"""
        payload = self._convert_embedding_request(request)
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Aliyun")
            return self._convert_embedding_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Aliyun")

