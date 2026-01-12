from typing import Dict, Any, Iterator, AsyncIterator, List, Optional
import httpx
import json
from ..types import UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, ProviderConfig, ChunkChoice
from ..exceptions import AuthenticationError, RateLimitError, ProviderError
from .base import BaseProvider

class AliyunProvider(BaseProvider):
    """Alibaba Cloud (DashScope/Qwen) Provider Adapter"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
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

    def _handle_error(self, response: httpx.Response):
        try:
            error_data = response.json()
            code = error_data.get("code", "")
            msg = error_data.get("message", response.text)
        except Exception:
            code = ""
            msg = response.text

        if response.status_code == 401 or code == "InvalidApiKey":
            raise AuthenticationError(f"Aliyun Auth Failed: {msg}", provider="aliyun", status_code=401)
        elif response.status_code == 429 or code == "Throttling.RateQuota":
            raise RateLimitError(f"Aliyun Rate Limit: {msg}", provider="aliyun", status_code=429)
        else:
            raise ProviderError(f"Aliyun Error {response.status_code} ({code}): {msg}", provider="aliyun", status_code=response.status_code)

    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        # DashScope is strictly POST
        url = self.base_url # base_url is full path
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"Network error: {str(e)}", provider="aliyun")

    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = self.base_url
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"Network error: {str(e)}", provider="aliyun")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = self.base_url
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                for line in response.iter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        # DashScope doesn't use [DONE] typically? It just stops.
                        # But standard SSE might.
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
             raise ProviderError(f"Network error during stream: {str(e)}", provider="aliyun")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = self.base_url
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
             raise ProviderError(f"Network error during stream: {str(e)}", provider="aliyun")
