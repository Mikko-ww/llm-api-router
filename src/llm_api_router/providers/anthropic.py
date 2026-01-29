from typing import Dict, Any, Iterator, AsyncIterator, List, Optional
import httpx
import json
from ..types import UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, ProviderConfig, ChunkChoice
from ..exceptions import StreamError
from ..retry import with_retry, with_retry_async
from .base import BaseProvider

class AnthropicProvider(BaseProvider):
    """Anthropic Provider Adapter"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **config.extra_headers
        }

    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        # Handle System Message extraction
        messages = []
        system_prompt = None
        
        for msg in request.messages:
            if msg.get("role") == "system":
                # Concatenate multiple system messages if present
                if system_prompt is None:
                    system_prompt = msg.get("content", "")
                else:
                    system_prompt += "\n" + msg.get("content", "")
            else:
                messages.append(msg)

        data: Dict[str, Any] = {
            "model": request.model or self.config.default_model or "claude-3-5-sonnet-20240620",
            "messages": messages,
            "stream": request.stream,
            "max_tokens": request.max_tokens or 1024  # Anthropic usually requires this
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stop is not None:
            data["stop_sequences"] = request.stop
            
        return data

    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        content_blocks = provider_response.get("content", [])
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        usage_data = provider_response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
        )

        return UnifiedResponse(
            id=provider_response.get("id", ""),
            object="chat.completion",
            created=0, # Anthropic doesn't return timestamp in top level usually
            model=provider_response.get("model", ""),
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=text_content),
                finish_reason=provider_response.get("stop_reason") or "stop"
            )],
            usage=usage
        )
    
    def _convert_chunk_event(self, event_type: str, data: Dict[str, Any]) -> Optional[UnifiedChunk]:
        # Anthropic SSE sequence:
        # message_start -> content_block_start -> content_block_delta -> ... -> content_block_stop -> message_delta -> message_stop
        
        if event_type == "content_block_delta":
            delta_data = data.get("delta", {})
            if delta_data.get("type") == "text_delta":
                text = delta_data.get("text", "")
                return UnifiedChunk(
                    id="",
                    object="chat.completion.chunk",
                    created=0,
                    model="",
                    choices=[ChunkChoice(
                        index=data.get("index", 0),
                        delta=Message(role="assistant", content=text),
                        finish_reason=None
                    )]
                )
        elif event_type == "message_delta":
            delta = data.get("delta", {})
             # Captured stop reason
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                 return UnifiedChunk(
                    id="",
                    object="chat.completion.chunk",
                    created=0,
                    model="",
                    choices=[ChunkChoice(
                        index=0,
                        delta=Message(role="", content=""),
                        finish_reason=stop_reason
                    )]
                )
        # We ignore other events like message_start, ping for simple text streaming for now
        return None

    @with_retry()
    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = f"{self.base_url}/messages"
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Anthropic")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Anthropic")

    @with_retry_async()
    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = f"{self.base_url}/messages"
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Anthropic")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Anthropic")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/messages"
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "Anthropic")
                
                # Custom SSE parsing for Anthropic
                current_event_type = None
                
                for line in response.iter_lines():
                    if not line: continue
                    line = line.strip()
                    if line.startswith("event: "):
                        current_event_type = line[7:].strip()
                    elif line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not current_event_type: continue
                        
                        try:
                            data = json.loads(data_str)
                            chunk = self._convert_chunk_event(current_event_type, data)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="Anthropic")
        except httpx.RequestError as e:
            self.handle_request_error(e, "Anthropic")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/messages"
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "Anthropic")
                
                current_event_type = None

                async for line in response.aiter_lines():
                    if not line: continue
                    line = line.strip()
                    if line.startswith("event: "):
                        current_event_type = line[7:].strip()
                    elif line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not current_event_type: continue
                        
                        try:
                            data = json.loads(data_str)
                            chunk = self._convert_chunk_event(current_event_type, data)
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="Anthropic")
        except httpx.RequestError as e:
            self.handle_request_error(e, "Anthropic")
