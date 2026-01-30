from typing import Dict, Any, Iterator, AsyncIterator, List
import httpx
import json
import time
from ..types import (
    UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, 
    ProviderConfig, ChunkChoice, EmbeddingRequest, EmbeddingResponse, 
    Embedding, EmbeddingUsage, Tool, ToolCall, FunctionCall, FunctionDefinition
)
from ..exceptions import StreamError
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
        
        # Add tools support
        if request.tools is not None:
            data["tools"] = self._convert_tools_to_openai(request.tools)
        if request.tool_choice is not None:
            data["tool_choice"] = request.tool_choice
        
        return data
    
    def _convert_tools_to_openai(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert Tool objects to OpenAI format"""
        result = []
        for tool in tools:
            tool_dict: Dict[str, Any] = {
                "type": tool.type
            }
            if tool.function:
                function_dict: Dict[str, Any] = {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters
                }
                if tool.function.strict is not None:
                    function_dict["strict"] = tool.function.strict
                tool_dict["function"] = function_dict
            result.append(tool_dict)
        return result

    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        choices = []
        for c in provider_response.get("choices", []):
            msg_data = c.get("message", {})
            
            # Parse tool_calls if present
            tool_calls = None
            if "tool_calls" in msg_data and msg_data["tool_calls"]:
                tool_calls = self._parse_tool_calls(msg_data["tool_calls"])
            
            message = Message(
                role=msg_data.get("role", ""),
                content=msg_data.get("content"),
                tool_calls=tool_calls
            )
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
    
    def _parse_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[ToolCall]:
        """Parse tool_calls from OpenAI response format"""
        result = []
        for tc in tool_calls_data:
            func_data = tc.get("function", {})
            tool_call = ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=FunctionCall(
                    name=func_data.get("name", ""),
                    arguments=func_data.get("arguments", "{}")
                )
            )
            result.append(tool_call)
        return result
    
    def _convert_chunk(self, chunk_data: Dict[str, Any]) -> UnifiedChunk:
        choices = []
        for c in chunk_data.get("choices", []):
            delta_data = c.get("delta", {})
            
            # Parse tool_calls if present in delta
            tool_calls = None
            if "tool_calls" in delta_data and delta_data["tool_calls"]:
                tool_calls = self._parse_tool_calls(delta_data["tool_calls"])
            
            delta = Message(
                role=delta_data.get("role", ""),
                content=delta_data.get("content"),
                tool_calls=tool_calls
            )
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

    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        return self._with_retry(self._send_request_impl)(client, request)
    
    def _send_request_impl(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        start_time = time.time()
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        
        # Log request
        log_extra = {
            "provider": "OpenAI",
            "model": request.model or self.config.default_model,
            "message_count": len(request.messages),
        }
        if request.request_id:
            log_extra["request_id"] = request.request_id
        
        self.logger.info("Sending chat completion request", extra=log_extra)
        
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                # Record failure metrics
                self._record_metrics(
                    model=request.model or self.config.default_model,
                    latency_ms=latency_ms,
                    success=False,
                    status_code=response.status_code,
                    error_type=f"HTTP_{response.status_code}",
                    stream=False,
                    request_id=request.request_id,
                )
                self.handle_error_response(response, "OpenAI", request.request_id)
            
            result = self.convert_response(response.json())
            
            # Record success metrics
            self._record_metrics(
                model=result.model,
                latency_ms=latency_ms,
                success=True,
                status_code=200,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                stream=False,
                request_id=request.request_id,
            )
            
            # Log successful response
            log_extra["latency_ms"] = int(latency_ms)
            log_extra["tokens"] = result.usage.total_tokens
            self.logger.info("Chat completion successful", extra=log_extra)
            
            return result
        except httpx.RequestError as e:
            latency_ms = (time.time() - start_time) * 1000
            # Record failure metrics
            self._record_metrics(
                model=request.model or self.config.default_model,
                latency_ms=latency_ms,
                success=False,
                error_type=type(e).__name__,
                stream=False,
                request_id=request.request_id,
            )
            self.handle_request_error(e, "OpenAI", request.request_id)

    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        return await self._with_retry_async(self._send_request_async_impl)(client, request)
    
    async def _send_request_async_impl(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        start_time = time.time()
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        
        # Log request
        log_extra = {
            "provider": "OpenAI",
            "model": request.model or self.config.default_model,
            "message_count": len(request.messages),
        }
        if request.request_id:
            log_extra["request_id"] = request.request_id
        
        self.logger.info("Sending async chat completion request", extra=log_extra)
        
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                # Record failure metrics
                self._record_metrics(
                    model=request.model or self.config.default_model,
                    latency_ms=latency_ms,
                    success=False,
                    status_code=response.status_code,
                    error_type=f"HTTP_{response.status_code}",
                    stream=False,
                    request_id=request.request_id,
                )
                self.handle_error_response(response, "OpenAI", request.request_id)
            
            result = self.convert_response(response.json())
            
            # Record success metrics
            self._record_metrics(
                model=result.model,
                latency_ms=latency_ms,
                success=True,
                status_code=200,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                stream=False,
                request_id=request.request_id,
            )
            
            # Log successful response
            log_extra["latency_ms"] = int(latency_ms)
            log_extra["tokens"] = result.usage.total_tokens
            self.logger.info("Async chat completion successful", extra=log_extra)
            
            return result
        except httpx.RequestError as e:
            latency_ms = (time.time() - start_time) * 1000
            # Record failure metrics
            self._record_metrics(
                model=request.model or self.config.default_model,
                latency_ms=latency_ms,
                success=False,
                error_type=type(e).__name__,
                stream=False,
                request_id=request.request_id,
            )
            self.handle_request_error(e, "OpenAI", request.request_id)

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        
        # Log request
        log_extra = {
            "provider": "OpenAI",
            "model": request.model or self.config.default_model,
            "message_count": len(request.messages),
            "stream": True,
        }
        if request.request_id:
            log_extra["request_id"] = request.request_id
        
        self.logger.info("Starting stream request", extra=log_extra)
        
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "OpenAI", request.request_id)
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            self.logger.debug("Stream completed", extra=log_extra)
                            break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="OpenAI")
        except httpx.RequestError as e:
            self.handle_request_error(e, "OpenAI", request.request_id)

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
    
    def create_embeddings(self, client: httpx.Client, request: EmbeddingRequest) -> EmbeddingResponse:
        return self._with_retry(self._create_embeddings_impl)(client, request)
    
    def _create_embeddings_impl(self, client: httpx.Client, request: EmbeddingRequest) -> EmbeddingResponse:
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
    
    async def create_embeddings_async(self, client: httpx.AsyncClient, request: EmbeddingRequest) -> EmbeddingResponse:
        return await self._with_retry_async(self._create_embeddings_async_impl)(client, request)
    
    async def _create_embeddings_async_impl(self, client: httpx.AsyncClient, request: EmbeddingRequest) -> EmbeddingResponse:
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