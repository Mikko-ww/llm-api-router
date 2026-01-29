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

class GeminiProvider(BaseProvider):
    """Google Gemini Provider Adapter"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.api_key = config.api_key
        # Gemini uses a query parameter 'key' commonly, but we can try header 'x-goog-api-key' for cleanliness if supported.
        # Fallback to query param if needed, but official docs mention x-goog-api-key for REST.
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            **config.extra_headers
        }

    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        contents = []
        system_instruction = None

        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                # Concatenate system messages
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += "\n" + content
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })

        model = request.model or self.config.default_model or "gemini-1.5-flash"
        
        # Adjust model name: Gemini API usually expects 'models/gemini-pro' in URL or just 'gemini-pro'
        # The URL structure is .../models/{model}:generateContent
        # We will handle the URL construction in send_request based on this model.
        
        data: Dict[str, Any] = {
            "contents": contents
        }
        
        if system_instruction:
            data["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        generation_config = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.stop is not None:
             generation_config["stopSequences"] = request.stop

        if generation_config:
            data["generationConfig"] = generation_config
            
        return data

    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        content = ""
        finish_reason = "stop"
        
        candidates = provider_response.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            # Capture finish reason mapping
            # Gemini reasons: STOP, MAX_TOKENS, SAFETY, RECITATION, ...
            finish_reason = candidate.get("finishReason", "stop").lower()
            
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                content += part.get("text", "")

        usage_metadata = provider_response.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0),
            completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
            total_tokens=usage_metadata.get("totalTokenCount", 0)
        )

        return UnifiedResponse(
            id="", # Gemini doesn't return a stable session ID in simple rest calls
            object="chat.completion",
            created=0,
            model="", # Not explicitly echoed in all responses
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason=finish_reason
            )],
            usage=usage
        )
    
    def _convert_chunk(self, chunk_data: Dict[str, Any]) -> UnifiedChunk:
        content = ""
        candidates = chunk_data.get("candidates", [])
        finish_reason = None
        
        if candidates:
            candidate = candidates[0]
            chunk_finish = candidate.get("finishReason") 
            # In Gemini stream, intermediate chunks might not have finishReason, or it is None/unspecified
            if chunk_finish and chunk_finish != "STOP": # Normal stop might just be end
                 finish_reason = chunk_finish.lower()

            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                content += part.get("text", "")
                
        return UnifiedChunk(
            id="",
            object="chat.completion.chunk",
            created=0,
            model="",
            choices=[ChunkChoice(
                index=0,
                delta=Message(role="assistant", content=content),
                finish_reason=finish_reason
            )]
        )

    def _get_url(self, model: str, stream: bool = False) -> str:
        # If model doesn't start with 'models/', prepend it? No, standard is often just 'gemini-1.5-flash'
        # But API expects `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
        # Safe URL encoding just in case
        if not model.startswith("models/"):
             model_path = f"models/{model}"
        else:
             model_path = model
        
        action = "streamGenerateContent" if stream else "generateContent"
        return f"{self.base_url}/{model_path}:{action}"

    @with_retry()
    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        model = request.model or self.config.default_model or "gemini-1.5-flash"
        payload = self.convert_request(request)
        url = self._get_url(model, stream=False)
        
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Gemini")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Gemini")

    @with_retry_async()
    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        model = request.model or self.config.default_model or "gemini-1.5-flash"
        payload = self.convert_request(request)
        url = self._get_url(model, stream=False)
        
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Gemini")
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Gemini")

    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        model = request.model or self.config.default_model or "gemini-1.5-flash"
        payload = self.convert_request(request)
        url = self._get_url(model, stream=True)
        # Gemini Stream also accepts params like ?alt=sse usually? or just returns a JSON list?
        # Actually standard REST streamGenerateContent returns a JSON list bracket by bracket if not using alt=sse.
        # But commonly we use alt=sse for easier parsing if supported.
        # Let's try adding ?alt=sse query param?
        # Official docs say: Content-Type: application/json. Response is a stream of JSON objects.
        # It's NOT SSE by default in the raw REST API, it's a long-lived JSON array usually.
        # HOWEVER, let's treat it as line-delimited JSON or check docs.
        # "The response is a chunked response...".
        # Let's assume standard behavior: we might get a JSON array `[` then objects `, { ... }`.
        # This is hard to parse with standard `iter_lines`.
        # BUT: simple solution -> usually `alt=sse` enables SSE.
        
        # Let's try appending `?alt=sse` to URL for easier parsing.
        url += "?alt=sse"
        
        try:
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "Gemini")
                
                for line in response.iter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="Gemini")
        except httpx.RequestError as e:
            self.handle_request_error(e, "Gemini")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        model = request.model or self.config.default_model or "gemini-1.5-flash"
        payload = self.convert_request(request)
        url = self._get_url(model, stream=True)
        url += "?alt=sse"
        
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=self.config.timeout) as response:
                if response.status_code != 200:
                    self.handle_error_response(response, "Gemini")
                
                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError as e:
                            raise StreamError(f"Failed to parse stream data: {str(e)}", provider="Gemini")
        except httpx.RequestError as e:
            self.handle_request_error(e, "Gemini")

    # --- Embeddings Implementation ---
    
    def supports_embeddings(self) -> bool:
        """Gemini 支持 embeddings API"""
        return True
    
    def _convert_embedding_request(self, request: EmbeddingRequest, text: str) -> Dict[str, Any]:
        """将单个文本转换为 Gemini embedContent 请求格式"""
        # Gemini batchEmbedContents format:
        # { "requests": [{ "model": "models/embedding-001", "content": { "parts": [{"text": "..."}] } }] }
        # For single: embedContent
        data: Dict[str, Any] = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        if request.dimensions is not None:
            data["outputDimensionality"] = request.dimensions
        return data
    
    def _convert_batch_embedding_request(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """将批量请求转换为 Gemini batchEmbedContents 格式"""
        model = request.model or "embedding-001"
        if not model.startswith("models/"):
            model = f"models/{model}"
            
        requests = []
        for text in request.input:
            req: Dict[str, Any] = {
                "model": model,
                "content": {
                    "parts": [{"text": text}]
                }
            }
            if request.dimensions is not None:
                req["outputDimensionality"] = request.dimensions
            requests.append(req)
        
        return {"requests": requests}
    
    def _convert_embedding_response(self, provider_response: Dict[str, Any]) -> EmbeddingResponse:
        """将 Gemini batchEmbedContents 响应转换为统一格式"""
        embeddings_data = provider_response.get("embeddings", [])
        
        embeddings = []
        for idx, item in enumerate(embeddings_data):
            embeddings.append(Embedding(
                index=idx,
                embedding=item.get("values", []),
                object="embedding"
            ))
        
        # Gemini doesn't return token usage for embeddings
        usage = EmbeddingUsage(
            prompt_tokens=0,
            total_tokens=0
        )
        
        return EmbeddingResponse(
            data=embeddings,
            model="",  # Gemini doesn't echo model in response
            usage=usage,
            object="list"
        )
    
    def _get_embedding_url(self, model: str) -> str:
        """获取 embedding API URL"""
        if not model.startswith("models/"):
            model = f"models/{model}"
        return f"{self.base_url}/{model}:batchEmbedContents"
    
    @with_retry()
    def create_embeddings(self, client: httpx.Client, request: EmbeddingRequest) -> EmbeddingResponse:
        """创建文本嵌入（同步）"""
        model = request.model or "embedding-001"
        payload = self._convert_batch_embedding_request(request)
        url = self._get_embedding_url(model)
        
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Gemini")
            return self._convert_embedding_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Gemini")
    
    @with_retry_async()
    async def create_embeddings_async(self, client: httpx.AsyncClient, request: EmbeddingRequest) -> EmbeddingResponse:
        """创建文本嵌入（异步）"""
        model = request.model or "embedding-001"
        payload = self._convert_batch_embedding_request(request)
        url = self._get_embedding_url(model)
        
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=self.config.timeout)
            if response.status_code != 200:
                self.handle_error_response(response, "Gemini")
            return self._convert_embedding_response(response.json())
        except httpx.RequestError as e:
            self.handle_request_error(e, "Gemini")
