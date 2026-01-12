from typing import Dict, Any, Iterator, AsyncIterator, List
import httpx
import json
from ..types import UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, ProviderConfig, ChunkChoice
from ..exceptions import AuthenticationError, RateLimitError, ProviderError, LLMRouterError
from .base import BaseProvider

class OpenRouterProvider(BaseProvider):
    """OpenRouter 提供商适配器"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = (config.base_url or "https://openrouter.ai/api/v1").rstrip("/")
        # OpenRouter 推荐必须的头部
        default_headers = {
            "HTTP-Referer": "https://github.com/your-repo/llm-api-router", # 默认引用页
            "X-Title": "LLM API Router", # 默认应用名称
        }
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **default_headers,
            **config.extra_headers
        }

    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": request.messages,
            "stream": request.stream
        }
        # OpenRouter 支持所有 OpenAI 参数，甚至更多模型特定参数
        # 这里保持与 UnifiedRequest 一致
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stop is not None:
            data["stop"] = request.stop
        
        # 处理 OpenRouter 特有的 transforms 或其他参数，如果 UnifiedRequest 支持扩展的话
        # 目前保持基础兼容
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

    def _handle_error(self, response: httpx.Response):
        try:
            error_data = response.json()
            # OpenRouter 错误格式通常包含 error 字段
            msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            msg = response.text

        if response.status_code == 401:
            raise AuthenticationError(f"OpenRouter Auth Failed: {msg}", provider="openrouter", status_code=401)
        elif response.status_code == 402:
            raise AuthenticationError(f"OpenRouter Insufficient Credits: {msg}", provider="openrouter", status_code=402)
        elif response.status_code == 429:
            raise RateLimitError(f"OpenRouter Rate Limit: {msg}", provider="openrouter", status_code=429)
        elif response.status_code >= 500:
            raise ProviderError(f"OpenRouter Server Error: {msg}", provider="openrouter", status_code=response.status_code)
        else:
            raise ProviderError(f"OpenRouter Error {response.status_code}: {msg}", provider="openrouter", status_code=response.status_code)

    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"Network error: {str(e)}", provider="openrouter")

    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"Network error: {str(e)}", provider="openrouter")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            # 某些 OpenRouter 模型可能返回 :keepalive:
                            if data_str.strip() == ":keepalive:":
                                continue
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
             raise ProviderError(f"Network error during stream: {str(e)}", provider="openrouter")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        payload = self.convert_request(request)
        url = f"{self.base_url}/chat/completions"
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            if data_str.strip() == ":keepalive:":
                                continue
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
             raise ProviderError(f"Network error during stream: {str(e)}", provider="openrouter")
