"""LLM API Router - Unified API for multiple LLM providers."""

from .client import Client, AsyncClient
from .types import ProviderConfig, Message, UnifiedRequest, UnifiedResponse, UnifiedChunk
from .exceptions import (
    LLMRouterError,
    AuthenticationError,
    RateLimitError,
    ProviderError,
    StreamError
)

__version__ = "0.1.2"

__all__ = [
    "Client",
    "AsyncClient",
    "ProviderConfig",
    "Message",
    "UnifiedRequest",
    "UnifiedResponse",
    "UnifiedChunk",
    "LLMRouterError",
    "AuthenticationError",
    "RateLimitError",
    "ProviderError",
    "StreamError",
]
