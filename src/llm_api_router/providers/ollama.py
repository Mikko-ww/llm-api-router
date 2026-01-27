from typing import Dict, Any, Iterator, AsyncIterator
import httpx
import json
from datetime import datetime
from ..types import UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, ProviderConfig, ChunkChoice
from ..exceptions import AuthenticationError, RateLimitError, ProviderError
from .base import BaseProvider

class OllamaProvider(BaseProvider):
    """Ollama 提供商适配器
    
    Ollama 提供本地运行的开源模型 API，支持 LLaMA、Mistral 等模型。
    默认运行在 http://localhost:11434
    """
    
    # Ollama 模型通常需要更长的响应时间，因为是本地运算
    DEFAULT_TIMEOUT = 120.0
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        # Ollama 默认运行在本地 11434 端口
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")
        # Ollama API 不需要认证密钥（本地服务），但保留 headers 以防用户有自定义配置
        self.headers = {
            "Content-Type": "application/json",
            **config.extra_headers
        }

    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        """将统一请求转换为 Ollama API 格式"""
        data: Dict[str, Any] = {
            "model": request.model or self.config.default_model,
            "messages": request.messages,
            "stream": request.stream
        }
        
        # Ollama 使用 options 对象来配置生成参数
        options: Dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop is not None:
            options["stop"] = request.stop
        
        # Ollama 没有直接的 max_tokens 参数，使用 num_predict
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
            
        if options:
            data["options"] = options
            
        return data

    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        """将 Ollama API 响应转换为统一格式
        
        Ollama 响应格式:
        {
            "model": "llama3.2",
            "created_at": "2025-01-09T06:47:26.589285Z",
            "message": {
                "role": "assistant",
                "content": "..."
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 88,
            "eval_count": 53
        }
        """
        msg_data = provider_response.get("message", {})
        message = Message(
            role=msg_data.get("role", "assistant"),
            content=msg_data.get("content", "")
        )
        
        choices = [Choice(
            index=0,
            message=message,
            finish_reason=provider_response.get("done_reason", "stop")
        )]
        
        # Ollama 提供的 token 计数
        prompt_tokens = provider_response.get("prompt_eval_count", 0)
        completion_tokens = provider_response.get("eval_count", 0)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )

        # Ollama 使用 ISO 时间戳，我们需要转换为 Unix 时间戳
        created_at = provider_response.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            created = int(dt.timestamp())
        except Exception:
            created = 0

        return UnifiedResponse(
            id=f"ollama-{created}",
            object="chat.completion",
            created=created,
            model=provider_response.get("model", ""),
            choices=choices,
            usage=usage
        )
    
    def _convert_chunk(self, chunk_data: Dict[str, Any]) -> UnifiedChunk:
        """将 Ollama 流式响应块转换为统一格式
        
        Ollama 流式响应格式:
        {
            "model": "llama3.2",
            "created_at": "2025-01-09T06:47:26.589285Z",
            "message": {
                "role": "assistant",
                "content": "word"
            },
            "done": false
        }
        """
        msg_data = chunk_data.get("message", {})
        delta = Message(
            role=msg_data.get("role", ""),
            content=msg_data.get("content", "")
        )
        
        finish_reason = None
        if chunk_data.get("done"):
            finish_reason = chunk_data.get("done_reason", "stop")
        
        choices = [ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )]
        
        # 处理时间戳
        created_at = chunk_data.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            created = int(dt.timestamp())
        except Exception:
            created = 0
        
        return UnifiedChunk(
            id=f"ollama-{created}",
            object="chat.completion.chunk",
            created=created,
            model=chunk_data.get("model", ""),
            choices=choices
        )

    def _handle_error(self, response: httpx.Response):
        """处理 Ollama API 错误响应"""
        try:
            error_data = response.json()
            msg = error_data.get("error", response.text)
        except Exception:
            msg = response.text

        if response.status_code == 404:
            raise ProviderError(f"Ollama 模型未找到或服务未运行: {msg}", provider="ollama", status_code=404)
        elif response.status_code >= 500:
            raise ProviderError(f"Ollama 服务器错误: {msg}", provider="ollama", status_code=response.status_code)
        else:
            raise ProviderError(f"Ollama 错误 {response.status_code}: {msg}", provider="ollama", status_code=response.status_code)

    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        """执行同步请求到 Ollama"""
        payload = self.convert_request(request)
        url = f"{self.base_url}/api/chat"
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.DEFAULT_TIMEOUT)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"网络错误: {str(e)}", provider="ollama")

    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        """执行异步请求到 Ollama"""
        payload = self.convert_request(request)
        url = f"{self.base_url}/api/chat"
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.DEFAULT_TIMEOUT)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"网络错误: {str(e)}", provider="ollama")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        """执行同步流式请求到 Ollama
        
        Ollama 使用 newline-delimited JSON (NDJSON) 格式而非 SSE
        """
        payload = self.convert_request(request)
        url = f"{self.base_url}/api/chat"
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.DEFAULT_TIMEOUT) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                # Ollama 使用 NDJSON 格式：每行一个 JSON 对象
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield self._convert_chunk(data)
                        # 如果收到 done=true，停止迭代
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.RequestError as e:
            raise ProviderError(f"流式请求网络错误: {str(e)}", provider="ollama")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        """执行异步流式请求到 Ollama
        
        Ollama 使用 newline-delimited JSON (NDJSON) 格式而非 SSE
        """
        payload = self.convert_request(request)
        url = f"{self.base_url}/api/chat"
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.DEFAULT_TIMEOUT) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                # Ollama 使用 NDJSON 格式：每行一个 JSON 对象
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield self._convert_chunk(data)
                        # 如果收到 done=true，停止迭代
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.RequestError as e:
            raise ProviderError(f"流式请求网络错误: {str(e)}", provider="ollama")
