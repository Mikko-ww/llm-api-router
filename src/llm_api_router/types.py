from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .logging_config import LogConfig
    from .metrics import MetricsCollector

@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3  # 最大重试次数
    initial_delay: float = 1.0  # 初始延迟（秒）
    max_delay: float = 60.0  # 最大延迟（秒）
    exponential_base: float = 2.0  # 指数退避基数
    retry_on_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)  # 需要重试的状态码


@dataclass
class TimeoutConfig:
    """超时配置 - 支持更细粒度的超时控制"""
    connect: float = 10.0  # 连接超时（秒）
    read: float = 60.0  # 读取超时（秒）
    write: float = 10.0  # 写入超时（秒）
    pool: float = 10.0  # 连接池获取连接的超时（秒）


@dataclass
class ConnectionPoolConfig:
    """HTTP连接池配置 - 优化连接复用和并发性能"""
    max_connections: int = 100  # 连接池最大连接数
    max_keepalive_connections: int = 20  # 最大保持活动的连接数
    keepalive_expiry: float = 300.0  # 连接保持活动时间（秒），默认5分钟
    stream_buffer_size: int = 65536  # 流式响应缓冲区大小（字节），默认64KB

@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_type: str
    api_key: str
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    api_version: Optional[str] = None  # 主要用于 Azure
    timeout: float = 60.0  # 简单超时配置（秒），用于向后兼容
    timeout_config: Optional[TimeoutConfig] = None  # 细粒度超时配置，优先于timeout使用
    connection_pool_config: Optional[ConnectionPoolConfig] = None  # 连接池配置，None表示使用默认配置
    retry_config: Optional[RetryConfig] = None  # 重试配置，None表示使用默认配置
    log_config: Optional['LogConfig'] = None  # 日志配置，None表示使用默认配置
    metrics_enabled: bool = True  # 是否启用性能指标收集
    metrics_collector: Optional['MetricsCollector'] = None  # 自定义metrics收集器，None表示使用全局收集器

@dataclass
class Message:
    """消息实体"""
    role: str
    content: Optional[str] = None  # Content can be None when there are tool_calls
    tool_calls: Optional[List['ToolCall']] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool messages (function results)


# --- Function/Tool Calling Types ---

@dataclass
class FunctionDefinition:
    """函数定义"""
    name: str  # Function name
    description: str  # Function description
    parameters: Dict[str, Any]  # JSON Schema for function parameters
    strict: Optional[bool] = None  # OpenAI: strict schema adherence


@dataclass
class Tool:
    """工具定义 (OpenAI style)"""
    type: str = "function"  # Always "function" for now
    function: Optional[FunctionDefinition] = None


@dataclass
class FunctionCall:
    """函数调用信息"""
    name: str  # Function name
    arguments: str  # JSON string of arguments


@dataclass
class ToolCall:
    """工具调用 (在响应中)"""
    id: str  # Unique identifier for this tool call
    type: str  # "function"
    function: FunctionCall


@dataclass
class UnifiedRequest:
    """统一请求对象"""
    messages: List[Dict[str, str]]  # 为了兼容性，保持为 Dict，但在内部可以使用 Message
    model: Optional[str] = None
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[Tool]] = None  # Function calling: tools available to the model
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # "none", "auto", "required", or specific tool
    request_id: Optional[str] = None  # Unique request ID for tracking

@dataclass
class Usage:
    """Token 使用情况"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class Choice:
    """补全选项"""
    index: int
    message: Message
    finish_reason: str

@dataclass
class UnifiedResponse:
    """统一响应对象 (非流式)"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

@dataclass
class ChunkChoice:
    """流式补全选项"""
    index: int
    delta: Message
    finish_reason: Optional[str] = None

@dataclass
class UnifiedChunk:
    """统一响应块 (流式)"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChunkChoice]


# --- Embeddings Types ---

@dataclass
class EmbeddingRequest:
    """嵌入请求对象"""
    input: List[str]  # 要嵌入的文本列表
    model: Optional[str] = None  # 模型名称，可选（使用默认模型）
    encoding_format: Optional[str] = None  # 编码格式: "float" 或 "base64"
    dimensions: Optional[int] = None  # 输出向量维度（仅部分模型支持）

@dataclass
class Embedding:
    """单个嵌入结果"""
    index: int  # 在输入列表中的索引
    embedding: List[float]  # 嵌入向量
    object: str = "embedding"

@dataclass
class EmbeddingUsage:
    """嵌入 API 的 token 使用情况"""
    prompt_tokens: int
    total_tokens: int

@dataclass
class EmbeddingResponse:
    """嵌入响应对象"""
    data: List[Embedding]  # 嵌入结果列表
    model: str  # 使用的模型
    usage: EmbeddingUsage  # token 使用情况
    object: str = "list"