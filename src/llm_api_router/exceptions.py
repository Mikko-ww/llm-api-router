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

class PermissionError(LLMRouterError):
    """权限不足 (HTTP 403)"""
    pass

class NotFoundError(LLMRouterError):
    """资源未找到 (HTTP 404)"""
    pass

class RateLimitError(LLMRouterError):
    """速率限制 (HTTP 429)"""
    pass

class BadRequestError(LLMRouterError):
    """请求错误 (HTTP 400)"""
    pass

class ProviderError(LLMRouterError):
    """提供商服务错误 (HTTP 5xx)"""
    pass

class TimeoutError(LLMRouterError):
    """请求超时错误"""
    pass

class NetworkError(LLMRouterError):
    """网络连接错误"""
    pass

class StreamError(LLMRouterError):
    """流式处理错误"""
    pass

class RetryExhaustedError(LLMRouterError):
    """重试次数耗尽错误"""
    pass