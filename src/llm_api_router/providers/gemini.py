from typing import Dict, Any, Iterator, AsyncIterator, List, Optional
import httpx
import json
from ..types import UnifiedRequest, UnifiedResponse, UnifiedChunk, Message, Choice, Usage, ProviderConfig, ChunkChoice
from ..exceptions import AuthenticationError, RateLimitError, ProviderError
from .base import BaseProvider

class GeminiProvider(BaseProvider):
    """Google Gemini Provider Adapter"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
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

    def _handle_error(self, response: httpx.Response):
        try:
            error_data = response.json()
            error = error_data.get("error", {})
            msg = error.get("message", response.text)
            status = error.get("code", response.status_code)
        except Exception:
            msg = response.text
            status = response.status_code

        if status == 400: # INVALID_ARGUMENT
             raise ProviderError(f"Gemini Invalid Argument: {msg}", provider="gemini", status_code=400)
        elif status == 401 or response.status_code == 401:
            raise AuthenticationError(f"Gemini Auth Failed: {msg}", provider="gemini", status_code=401)
        elif status == 429 or response.status_code == 429:
            raise RateLimitError(f"Gemini Rate Limit: {msg}", provider="gemini", status_code=429)
        else:
            raise ProviderError(f"Gemini Error {status}: {msg}", provider="gemini", status_code=response.status_code)

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

    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        model = request.model or self.config.default_model or "gemini-1.5-flash"
        payload = self.convert_request(request)
        url = self._get_url(model, stream=False)
        
        try:
            response = client.post(url, headers=self.headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"Network error: {str(e)}", provider="gemini")

    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        model = request.model or self.config.default_model or "gemini-1.5-flash"
        payload = self.convert_request(request)
        url = self._get_url(model, stream=False)
        
        try:
            response = await client.post(url, headers=self.headers, json=payload, timeout=60.0)
            if response.status_code != 200:
                self._handle_error(response)
            return self.convert_response(response.json())
        except httpx.RequestError as e:
            raise ProviderError(f"Network error: {str(e)}", provider="gemini")

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
            with client.stream("POST", url, headers=self.headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                for line in response.iter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
             raise ProviderError(f"Network error during stream: {str(e)}", provider="gemini")

    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        model = request.model or self.config.default_model or "gemini-1.5-flash"
        payload = self.convert_request(request)
        url = self._get_url(model, stream=True)
        url += "?alt=sse"
        
        try:
            async with client.stream("POST", url, headers=self.headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    self._handle_error(response)
                
                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            yield self._convert_chunk(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.RequestError as e:
             raise ProviderError(f"Network error during stream: {str(e)}", provider="gemini")
