from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_type: str
    api_key: str
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    api_version: Optional[str] = None  # 主要用于 Azure

@dataclass
class Message:
    """消息实体"""
    role: str
    content: str

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