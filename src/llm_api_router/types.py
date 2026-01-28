from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import os
import json
import yaml
import random
from pathlib import Path


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to delays
    retry_on_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff.
        
        Args:
            attempt: Current retry attempt (0-indexed)
        
        Returns:
            Delay in seconds
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter: delay * [0.5, 1.5]
            delay = delay * (0.5 + random.random())
        
        return delay


@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_type: str
    api_key: str
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    api_version: Optional[str] = None  # 主要用于 Azure
    
    @classmethod
    def from_env(
        cls,
        provider_type: str,
        api_key_env: Optional[str] = None,
        base_url_env: Optional[str] = None,
        default_model: Optional[str] = None,
        **kwargs
    ) -> "ProviderConfig":
        """从环境变量创建配置
        
        Args:
            provider_type: Provider type (e.g., 'openai', 'anthropic')
            api_key_env: Environment variable name for API key (defaults to <PROVIDER_TYPE>_API_KEY)
            base_url_env: Environment variable name for base URL (defaults to <PROVIDER_TYPE>_BASE_URL)
            default_model: Default model to use
            **kwargs: Additional configuration parameters
        
        Returns:
            ProviderConfig instance
        
        Raises:
            ValueError: If required environment variables are not set
        """
        # Default environment variable names
        if api_key_env is None:
            api_key_env = f"{provider_type.upper()}_API_KEY"
        
        # Get API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' not found or empty")
        
        # Get base URL from environment (optional)
        base_url = None
        if base_url_env:
            base_url = os.getenv(base_url_env)
        
        return cls(
            provider_type=provider_type,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            **kwargs
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """从字典创建配置
        
        Args:
            data: Configuration dictionary
        
        Returns:
            ProviderConfig instance
        
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ['provider_type', 'api_key']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, file_path: str) -> "ProviderConfig":
        """从文件创建配置（支持 JSON 和 YAML）
        
        Args:
            file_path: Path to configuration file (.json or .yaml/.yml)
        
        Returns:
            ProviderConfig instance
        
        Raises:
            ValueError: If file format is unsupported or file not found
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.json':
                data = json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Use .json, .yaml, or .yml")
        
        return cls.from_dict(data)
    
    def merge(self, other: Dict[str, Any]) -> "ProviderConfig":
        """合并配置，返回新的配置实例
        
        Args:
            other: Dictionary with configuration to merge (overrides current values)
        
        Returns:
            New ProviderConfig instance with merged configuration
        """
        # Convert current config to dict
        current = {
            'provider_type': self.provider_type,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'default_model': self.default_model,
            'extra_headers': self.extra_headers.copy(),
            'api_version': self.api_version,
        }
        
        # Merge other dict, overriding current values
        current.update(other)
        
        return ProviderConfig(**current)
    
    def validate(self) -> bool:
        """验证配置
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.provider_type:
            raise ValueError("provider_type cannot be empty")
        
        if not self.api_key:
            raise ValueError("api_key cannot be empty")
        
        # Additional validation for specific providers
        if self.provider_type == "azure" and not self.api_version:
            raise ValueError("api_version is required for Azure provider")
        
        return True

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