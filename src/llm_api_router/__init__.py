"""LLM API Router - 统一的大语言模型API路由库"""

from .client import Client, AsyncClient
from .types import (
    ProviderConfig,
    RetryConfig,
    TimeoutConfig,
    ConnectionPoolConfig,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedChunk,
    Message,
    Choice,
    ChunkChoice,
    Usage,
    # Function calling types
    Tool,
    ToolCall,
    FunctionCall,
    FunctionDefinition,
    # Embeddings types
    EmbeddingRequest,
    EmbeddingResponse,
    Embedding,
    EmbeddingUsage,
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
from .logging_config import LogConfig, setup_logging, get_logger
from .metrics import (
    MetricsCollector,
    RequestMetrics,
    AggregatedMetrics,
    get_metrics_collector,
    set_metrics_collector,
)
from .cache import (
    CacheConfig,
    CacheManager,
    CacheBackend,
    MemoryCacheBackend,
    RedisCacheBackend,
    generate_cache_key,
)
from .rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    RateLimiterBackend,
    TokenBucketBackend,
    SlidingWindowBackend,
)
from .templates import (
    PromptTemplate,
    TemplateEngine,
    TemplateRegistry,
    BuiltinTemplates,
)
from .conversation import (
    ConversationManager,
    ConversationConfig,
    TokenCounter,
    TruncationStrategy,
    SlidingWindowStrategy,
    KeepRecentStrategy,
    ImportanceBasedStrategy,
    create_conversation,
)
from .load_balancer import (
    LoadBalancer,
    LoadBalancerConfig,
    Endpoint,
    EndpointStatus,
    EndpointStats,
    SelectionStrategy,
    RoundRobinStrategy,
    WeightedStrategy,
    LeastLatencyStrategy,
    RandomStrategy,
    FailoverStrategy,
    create_load_balancer,
)

__version__ = "0.1.2"

__all__ = [
    # Clients
    "Client",
    "AsyncClient",
    # Types
    "ProviderConfig",
    "RetryConfig",
    "TimeoutConfig",
    "ConnectionPoolConfig",
    "UnifiedRequest",
    "UnifiedResponse",
    "UnifiedChunk",
    "Message",
    "Choice",
    "ChunkChoice",
    "Usage",
    # Function calling types
    "Tool",
    "ToolCall",
    "FunctionCall",
    "FunctionDefinition",
    # Embeddings types
    "EmbeddingRequest",
    "EmbeddingResponse",
    "Embedding",
    "EmbeddingUsage",
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
    # Logging
    "LogConfig",
    "setup_logging",
    "get_logger",
    # Metrics
    "MetricsCollector",
    "RequestMetrics",
    "AggregatedMetrics",
    "get_metrics_collector",
    "set_metrics_collector",
    # Cache
    "CacheConfig",
    "CacheManager",
    "CacheBackend",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    "generate_cache_key",
    # Rate Limiter
    "RateLimiter",
    "RateLimiterConfig",
    "RateLimiterBackend",
    "TokenBucketBackend",
    "SlidingWindowBackend",
    # Prompt Templates
    "PromptTemplate",
    "TemplateEngine",
    "TemplateRegistry",
    "BuiltinTemplates",
    # Conversation Management
    "ConversationManager",
    "ConversationConfig",
    "TokenCounter",
    "TruncationStrategy",
    "SlidingWindowStrategy",
    "KeepRecentStrategy",
    "ImportanceBasedStrategy",
    "create_conversation",
    # Load Balancer
    "LoadBalancer",
    "LoadBalancerConfig",
    "Endpoint",
    "EndpointStatus",
    "EndpointStats",
    "SelectionStrategy",
    "RoundRobinStrategy",
    "WeightedStrategy",
    "LeastLatencyStrategy",
    "RandomStrategy",
    "FailoverStrategy",
    "create_load_balancer",
]
