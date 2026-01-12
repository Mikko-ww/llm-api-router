from typing import Optional, Dict, Any

class LLMRouterError(Exception):
    """LLM API 路由器的基础异常类"""
    def __init__(self, message: str, provider: Optional[str] = None, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.details = details or {}

class AuthenticationError(LLMRouterError):
    """鉴权失败 (HTTP 401)"""
    pass

class RateLimitError(LLMRouterError):
    """速率限制 (HTTP 429)"""
    pass

class ProviderError(LLMRouterError):
    """提供商服务错误 (HTTP 5xx or 400)"""
    pass

class StreamError(LLMRouterError):
    """流式处理错误"""
    pass