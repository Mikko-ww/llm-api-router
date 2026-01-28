from typing import Dict, Any, Iterator, AsyncIterator
import httpx
import json
from ..types import UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, ProviderConfig, ChunkChoice
from ..exceptions import StreamError
from ..retry import with_retry, with_retry_async
from .base import BaseProvider

class XAIProvider(BaseProvider):
    """xAI (Grok) Provider Adapter"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or "https://api.x.ai/v1").rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **config.extra_headers
        }

    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": request.model or self.config.default_model or "grok-beta",
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
                self.handle_error_response(response, "xAI")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "xAI")

    @with_retry_async()
    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "xAI")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "xAI")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "xAI")
                
                for line in response.iter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="xAI")
        except httpx.RequestError as e:
            self.handle_request_error(e, "xAI")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "xAI")
                
                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="xAI")
        except httpx.RequestError as e:
            self.handle_request_error(e, "xAI")
