# API 参考

本节提供 LLM API Router 完整的 API 文档。

## 核心模块

### Client

::: llm_api_router.Client
    options:
      show_source: true
      heading_level: 3
      members:
        - chat
        - embeddings
        - close

### AsyncClient

::: llm_api_router.AsyncClient
    options:
      show_source: true
      heading_level: 3

### ProviderConfig

::: llm_api_router.ProviderConfig
    options:
      show_source: true
      heading_level: 3

## 快速参考

### Client 类

```python
class Client:
    """同步 LLM 客户端"""
    
    def __init__(
        self,
        config: ProviderConfig,
        cache: Optional[Cache] = None,
    ) -> None:
        """
        初始化客户端。
        
        Args:
            config: 提供商配置
            cache: 可选的缓存实例
        """
        
    def chat(self) -> ChatCompletions:
        """返回聊天完成接口"""
        
    def embeddings(self) -> Embeddings:
        """返回嵌入接口"""
        
    def close(self) -> None:
        """关闭客户端连接"""
        
    def __enter__(self) -> "Client":
        """上下文管理器入口"""
        
    def __exit__(self, *args) -> None:
        """上下文管理器退出"""
```

### ChatCompletions 接口

```python
class ChatCompletions:
    """聊天完成接口"""
    
    def create(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        创建聊天完成。
        
        Args:
            messages: 消息列表
            model: 模型名称（覆盖默认）
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 token 数
            stream: 是否流式输出
            tools: 工具定义列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数
            
        Returns:
            ChatCompletion 或流式迭代器
        """
```

### 响应类型

```python
@dataclass
class ChatCompletion:
    """聊天完成响应"""
    id: str
    choices: List[Choice]
    created: int
    model: str
    usage: Usage
    
@dataclass
class Choice:
    """选择项"""
    index: int
    message: Message
    finish_reason: str
    
@dataclass
class Message:
    """消息"""
    role: str
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    
@dataclass
class Usage:
    """使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### 流式响应

```python
@dataclass
class ChatCompletionChunk:
    """流式响应块"""
    id: str
    choices: List[ChunkChoice]
    created: int
    model: str
    
@dataclass
class ChunkChoice:
    """流式选择项"""
    index: int
    delta: Delta
    finish_reason: Optional[str]
    
@dataclass
class Delta:
    """增量内容"""
    role: Optional[str]
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
```

## 高级功能

### RateLimiter

```python
class RateLimiter:
    """速率限制器"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
        max_concurrent: int = 10,
    ) -> None:
        """初始化速率限制器"""
        
    async def acquire(self) -> AsyncContextManager:
        """获取请求许可"""
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
```

### LoadBalancer

```python
class LoadBalancer:
    """负载均衡器"""
    
    def __init__(
        self,
        configs: List[ProviderConfig],
        strategy: str = "round_robin",
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        初始化负载均衡器。
        
        Args:
            configs: 提供商配置列表
            strategy: 负载均衡策略
            weights: 权重列表（仅 weighted 策略）
        """
        
    def chat(self) -> LoadBalancedChat:
        """返回负载均衡聊天接口"""
```

### PromptTemplate

```python
class PromptTemplate:
    """提示模板"""
    
    def __init__(
        self,
        name: str,
        template: str,
        variables: List[str],
        default_values: Optional[Dict[str, str]] = None,
    ) -> None:
        """初始化模板"""
        
    def render(self, **kwargs) -> str:
        """渲染模板"""
        
    def validate(self, **kwargs) -> bool:
        """验证变量"""
```

### ConversationManager

```python
class ConversationManager:
    """会话管理器"""
    
    def __init__(
        self,
        max_history: int = 50,
        max_tokens: int = 4000,
        system_message: Optional[str] = None,
    ) -> None:
        """初始化会话管理器"""
        
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        
    def get_messages(self) -> List[Dict[str, str]]:
        """获取消息列表"""
        
    def clear(self) -> None:
        """清空历史"""
```

## 缓存模块

### MemoryCache

```python
class MemoryCache:
    """内存缓存"""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
    ) -> None:
        """初始化内存缓存"""
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        
    def set(self, key: str, value: Any) -> None:
        """设置缓存"""
        
    def clear(self) -> None:
        """清空缓存"""
```

### DiskCache

```python
class DiskCache:
    """磁盘缓存"""
    
    def __init__(
        self,
        cache_dir: str = ".llm_cache",
        max_size_mb: int = 500,
        ttl: int = 86400,
    ) -> None:
        """初始化磁盘缓存"""
```

## 异常类

```python
class LLMRouterError(Exception):
    """基础异常类"""
    
class AuthenticationError(LLMRouterError):
    """认证错误"""
    
class RateLimitError(LLMRouterError):
    """速率限制错误"""
    
class InvalidRequestError(LLMRouterError):
    """无效请求错误"""
    
class APIError(LLMRouterError):
    """API 错误"""
    
class TimeoutError(LLMRouterError):
    """超时错误"""
    
class ProviderError(LLMRouterError):
    """提供商错误"""
```

## 工厂函数

```python
def create_client(
    config: ProviderConfig,
    cache: Optional[Cache] = None,
) -> Client:
    """
    创建客户端的工厂函数。
    
    Args:
        config: 提供商配置
        cache: 可选的缓存实例
        
    Returns:
        配置好的 Client 实例
    """
```

## 类型定义

完整类型定义参见 [types.py](types.md)：

```python
from llm_api_router.types import (
    Message,
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Delta,
    Usage,
    ToolCall,
    Tool,
    EmbeddingResponse,
)
```
