# 类型参考

LLM API Router 使用统一的类型定义，确保跨提供商的一致性。

## 消息类型

### Message

表示对话中的单条消息：

```python
@dataclass
class Message:
    """聊天消息"""
    role: str           # "system" | "user" | "assistant" | "tool"
    content: str        # 消息内容
    name: Optional[str] = None  # 消息名称（可选）
    tool_calls: Optional[List[ToolCall]] = None  # 工具调用
    tool_call_id: Optional[str] = None  # 工具调用 ID
```

### MessageRole

支持的消息角色：

- `system` - 系统指令
- `user` - 用户输入
- `assistant` - 助手响应
- `tool` - 工具响应

## 响应类型

### ChatCompletion

聊天完成响应：

```python
@dataclass
class ChatCompletion:
    """聊天完成响应"""
    id: str              # 响应 ID
    choices: List[Choice]  # 选择列表
    created: int         # 创建时间戳
    model: str           # 使用的模型
    usage: Usage         # Token 使用统计
    system_fingerprint: Optional[str] = None
```

### Choice

响应选择项：

```python
@dataclass
class Choice:
    """聊天完成选择"""
    index: int           # 选择索引
    message: Message     # 响应消息
    finish_reason: str   # 完成原因: "stop" | "length" | "tool_calls"
```

### Usage

Token 使用统计：

```python
@dataclass
class Usage:
    """Token 使用统计"""
    prompt_tokens: int       # 输入 token 数
    completion_tokens: int   # 输出 token 数
    total_tokens: int        # 总 token 数
```

## 流式响应类型

### ChatCompletionChunk

流式响应块：

```python
@dataclass
class ChatCompletionChunk:
    """流式响应块"""
    id: str              # 响应 ID
    choices: List[ChunkChoice]  # 选择列表
    created: int         # 创建时间戳
    model: str           # 使用的模型
```

### ChunkChoice

流式选择项：

```python
@dataclass
class ChunkChoice:
    """流式选择"""
    index: int                     # 选择索引
    delta: Delta                   # 增量内容
    finish_reason: Optional[str]   # 完成原因
```

### Delta

增量内容：

```python
@dataclass
class Delta:
    """增量内容"""
    role: Optional[str] = None     # 角色（通常仅首块包含）
    content: Optional[str] = None  # 内容片段
    tool_calls: Optional[List[ToolCall]] = None
```

## 工具类型

### Tool

工具定义：

```python
@dataclass
class Tool:
    """工具定义"""
    type: str = "function"  # 工具类型
    function: FunctionDef   # 函数定义
```

### FunctionDef

函数定义：

```python
@dataclass
class FunctionDef:
    """函数定义"""
    name: str            # 函数名称
    description: str     # 函数描述
    parameters: dict     # JSON Schema 格式的参数定义
```

### ToolCall

工具调用：

```python
@dataclass
class ToolCall:
    """工具调用"""
    id: str              # 调用 ID
    type: str            # 类型（通常为 "function"）
    function: FunctionCall  # 函数调用详情
```

### FunctionCall

函数调用详情：

```python
@dataclass
class FunctionCall:
    """函数调用"""
    name: str            # 函数名称
    arguments: str       # JSON 格式的参数字符串
```

## 嵌入类型

### EmbeddingResponse

嵌入响应：

```python
@dataclass
class EmbeddingResponse:
    """嵌入响应"""
    data: List[EmbeddingData]  # 嵌入数据列表
    model: str                 # 使用的模型
    usage: EmbeddingUsage      # 使用统计
```

### EmbeddingData

单个嵌入数据：

```python
@dataclass
class EmbeddingData:
    """嵌入数据"""
    index: int           # 索引
    embedding: List[float]  # 嵌入向量
    object: str = "embedding"
```

## 配置类型

### ProviderConfig

提供商配置：

```python
@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_type: str           # 提供商类型
    api_key: str                 # API 密钥
    default_model: Optional[str] = None  # 默认模型
    base_url: Optional[str] = None      # 自定义 API 地址
    timeout: float = 30.0        # 请求超时（秒）
    max_retries: int = 3         # 最大重试次数
```

## 类型提示

LLM API Router 完全支持类型提示，可以与 IDE 和类型检查器（如 mypy）配合使用：

```python
from llm_api_router import Client, ProviderConfig
from llm_api_router.types import ChatCompletion, Message

def chat_with_llm(client: Client, prompt: str) -> str:
    response: ChatCompletion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}]
    )
    message: Message = response.choices[0].message
    return message.content or ""
```
