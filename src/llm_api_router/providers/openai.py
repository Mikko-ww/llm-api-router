from typing import Dict, Any, Iterator, AsyncIterator, List
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

class OpenAIProvider(BaseProvider):
    """OpenAI 提供商适配器"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **config.extra_headers
        }

    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": request.messages,
            "stream": request.stream
        }
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stop is not None:
            data["stop"] = request.stop
        return data

    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        choices = []
        for c in provider_response.get("choices", []):
            msg_data = c.get("message", {})
            message = Message(role=msg_data.get("role", ""), content=msg_data.get("content", ""))
            choices.append(Choice(
                index=c.get("index", 0),
                message=message,
                finish_reason=c.get("finish_reason", "")
            ))
        
        usage_data = provider_response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

        return UnifiedResponse(
            id=provider_response.get("id", ""),
            object=provider_response.get("object", "chat.completion"),
            created=provider_response.get("created", 0),
            model=provider_response.get("model", ""),
            choices=choices,
            usage=usage
        )
    
    def _convert_chunk(self, chunk_data: Dict[str, Any]) -> UnifiedChunk:
        choices = []
        for c in chunk_data.get("choices", []):
            delta_data = c.get("delta", {})
            delta = Message(role=delta_data.get("role", ""), content=delta_data.get("content", ""))
            choices.append(ChunkChoice(
                index=c.get("index", 0),
                delta=delta,
                finish_reason=c.get("finish_reason")
            ))
            
        return UnifiedChunk(
            id=chunk_data.get("id", ""),
            object=chunk_data.get("object", "chat.completion.chunk"),
            created=chunk_data.get("created", 0),
            model=chunk_data.get("model", ""),
            choices=choices
        )

    @with_retry()
    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "OpenAI")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "OpenAI")

    @with_retry_async()
    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "OpenAI")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "OpenAI")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "OpenAI")
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="OpenAI")
        except httpx.RequestError as e:
            self.handle_request_error(e, "OpenAI")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "OpenAI")
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="OpenAI")
        except httpx.RequestError as e:
            self.handle_request_error(e, "OpenAI")

    # --- Embeddings Implementation ---
    
    def supports_embeddings(self) -> bool:
        """OpenAI 支持 embeddings API"""
        return True
    
    def _convert_embedding_request(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """将嵌入请求转换为 OpenAI 特定的请求格式"""
        data: Dict[str, Any] = {
            "model": request.model or self.config.default_model or "text-embedding-3-small",
            "input": request.input
        }
        if request.encoding_format is not None:
            data["encoding_format"] = request.encoding_format
        if request.dimensions is not None:
            data["dimensions"] = request.dimensions
        return data
    
    def _convert_embedding_response(self, provider_response: Dict[str, Any]) -> EmbeddingResponse:
        """将 OpenAI 嵌入响应转换为统一格式"""
        embeddings = []
        for item in provider_response.get("data", []):
            embeddings.append(Embedding(
                index=item.get("index", 0),
                embedding=item.get("embedding", []),
                object=item.get("object", "embedding")
            ))
        
        usage_data = provider_response.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return EmbeddingResponse(
            data=embeddings,
            model=provider_response.get("model", ""),
            usage=usage,
            object=provider_response.get("object", "list")
        )
    
    @with_retry()
    def create_embeddings(self, client: httpx.Client, request: EmbeddingRequest) -> EmbeddingResponse:
        """创建文本嵌入（同步）"""
        payload = self._convert_embedding_request(request)
        url = f"{self.base_url}/embeddings"
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "OpenAI")
            return self._convert_embedding_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "OpenAI")
    
    @with_retry_async()
    async def create_embeddings_async(self, client: httpx.AsyncClient, request: EmbeddingRequest) -> EmbeddingResponse:
        """创建文本嵌入（异步）"""
        payload = self._convert_embedding_request(request)
        url = f"{self.base_url}/embeddings"
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "OpenAI")
            return self._convert_embedding_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "OpenAI")