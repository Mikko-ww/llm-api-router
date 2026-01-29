"""LLM API Router - 统一的大语言模型API路由库"""

from .client import Client, AsyncClient
from .types import (
    ProviderConfig,
    RetryConfig,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedChunk,
    Message,
    Choice,
    ChunkChoice,
    Usage,
)
from .exceptions import (
    LLMRouterError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RateLimitError,
    BadRequestError,
    ProviderError,
    TimeoutError,
    NetworkError,
    StreamError,
    RetryExhaustedError,
)

__version__ = "0.1.2"

__all__ = [
    # Clients
    "Client",
    "AsyncClient",
    # Types
    "ProviderConfig",
    "RetryConfig",
    "UnifiedRequest",
    "UnifiedResponse",
    "UnifiedChunk",
    "Message",
    "Choice",
    "ChunkChoice",
    "Usage",
    # Exceptions
    "LLMRouterError",
    "AuthenticationError",
    "PermissionError",
    "NotFoundError",
    "RateLimitError",
    "BadRequestError",
    "ProviderError",
    "TimeoutError",
    "NetworkError",
    "StreamError",
    "RetryExhaustedError",
]
