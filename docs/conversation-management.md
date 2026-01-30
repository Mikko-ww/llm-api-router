# Conversation Management

LLM API Router 提供智能的对话历史管理功能，包括自动 token 计数和多种截断策略。

## 快速开始

```python
from llm_api_router.conversation import create_conversation

# 创建对话管理器
conversation = create_conversation(
    max_tokens=4096,
    system_prompt="你是一个有帮助的助手。"
)

# 添加消息
conversation.add_user_message("你好！")
conversation.add_assistant_message("你好！有什么可以帮助你的吗？")

# 获取 API 格式的消息
messages = conversation.get_messages_for_api()
```

## 配置

```python
from llm_api_router.conversation import ConversationConfig, ConversationManager

config = ConversationConfig(
    max_tokens=4096,          # 最大 token 数
    max_messages=100,         # 最大消息数
    preserve_system=True,     # 保留系统消息
    model="gpt-3.5-turbo",   # 用于 token 计数的模型
    strategy="sliding_window" # 截断策略
)

manager = ConversationManager(config)
```

## 消息管理

### 添加消息

```python
# 设置系统提示
conversation.set_system_prompt("你是一个编程助手。")

# 添加用户消息
conversation.add_user_message("帮我写一个函数")

# 添加助手消息
conversation.add_assistant_message("好的，请告诉我需要什么功能。")

# 添加任意角色的消息
conversation.add_message("tool", "函数执行结果")
```

### 标记重要消息

重要消息在截断时会被优先保留（使用 importance 策略时）：

```python
conversation.add_user_message("记住我叫 Alice", important=True)
conversation.add_assistant_message("好的，Alice！", important=True)
```

### 其他操作

```python
# 获取消息列表（副本）
messages = conversation.messages

# 获取 API 格式消息（不含内部元数据）
api_messages = conversation.get_messages_for_api()

# 弹出最后一条消息
last = conversation.pop_last()

# 清空对话（保留系统消息）
conversation.clear(preserve_system=True)

# 完全清空
conversation.clear(preserve_system=False)
```

## 截断策略

### Sliding Window（滑动窗口）

**默认策略**。移除最旧的消息，保留最新的。

```python
config = ConversationConfig(
    strategy="sliding_window",
    max_tokens=4096
)
```

**工作原理：**
1. 计算系统消息占用的 token
2. 从最新消息开始，逐个添加直到达到限制
3. 丢弃无法容纳的旧消息

### Keep Recent（保留最近N条）

保留固定数量的最近消息。

```python
from llm_api_router.conversation import KeepRecentStrategy

strategy = KeepRecentStrategy(keep_count=10)
manager = ConversationManager(config, strategy=strategy)
```

### Importance Based（基于重要性）

优先保留标记为重要的消息。

```python
from llm_api_router.conversation import ImportanceBasedStrategy

strategy = ImportanceBasedStrategy()
manager = ConversationManager(config, strategy=strategy)

# 标记重要消息
manager.add_user_message("关键信息", important=True)
```

## Token 计数

### 使用 TokenCounter

```python
from llm_api_router.conversation import TokenCounter

counter = TokenCounter(model="gpt-3.5-turbo")

# 计算文本 token
tokens = counter.count_tokens("Hello, world!")

# 计算消息 token（含开销）
msg_tokens = counter.count_message_tokens({
    "role": "user",
    "content": "Hello"
})

# 计算多条消息
messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
]
total = counter.count_messages_tokens(messages)
```

### Token 计数说明

- 如果安装了 `tiktoken`，使用精确计数
- 否则使用估算（英文约 4 字符/token，中文约 1.5 字符/token）

安装 tiktoken：
```bash
pip install tiktoken
```

## 对话分叉

创建对话的独立副本：

```python
# 原始对话
conversation.add_user_message("帮我写 Python")

# 创建分叉
fork1 = conversation.fork()
fork2 = conversation.fork()

# 独立发展
fork1.add_user_message("列表推导式")
fork2.add_user_message("装饰器")
```

## 统计信息

```python
stats = conversation.get_stats()
print(stats)
# {
#   "message_count": 10,
#   "token_count": 850,
#   "max_tokens": 4096,
#   "max_messages": 100,
#   "roles": {"system": 1, "user": 5, "assistant": 4},
#   "strategy": "sliding_window"
# }
```

## 与 Client 集成

```python
from llm_api_router import Client, ProviderConfig
from llm_api_router.conversation import create_conversation

# 创建客户端
config = ProviderConfig(provider_type="openai", api_key="...")
client = Client(config)

# 创建对话
conversation = create_conversation(
    max_tokens=4096,
    model="gpt-4",
    system_prompt="你是一个有帮助的助手。"
)

# 对话循环
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    # 添加用户消息
    conversation.add_user_message(user_input)
    
    # 获取 API 消息
    messages = conversation.get_messages_for_api()
    
    # 调用 API
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4"
    )
    
    # 获取并保存助手响应
    assistant_message = response.choices[0].message.content
    conversation.add_assistant_message(assistant_message)
    
    print(f"Assistant: {assistant_message}")
```

## 自定义策略

实现自己的截断策略：

```python
from llm_api_router.conversation import TruncationStrategy, TokenCounter

class MyStrategy(TruncationStrategy):
    def truncate(
        self,
        messages: list,
        max_tokens: int,
        token_counter: TokenCounter,
        preserve_system: bool = True
    ) -> list:
        # 自定义截断逻辑
        return messages[-5:]  # 简单示例：保留最后 5 条

# 使用自定义策略
manager = ConversationManager(config, strategy=MyStrategy())
```

## 线程安全

`ConversationManager` 是线程安全的，可以在多线程环境中使用。

## 最佳实践

1. **选择合适的 max_tokens**
   - 留出足够空间给模型响应
   - 一般设置为模型上下文的 60-80%

2. **使用重要性标记**
   - 标记关键信息（如用户偏好、重要上下文）
   - 配合 importance 策略使用

3. **定期检查统计**
   - 监控 token 使用情况
   - 调整截断策略

4. **利用分叉功能**
   - 实现多路径对话
   - 测试不同的对话分支
