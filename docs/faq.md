# 常见问题 (FAQ)

## 安装问题

### Q: 安装时提示依赖冲突怎么办？

使用 `uv` 或创建虚拟环境隔离依赖：

```bash
# 使用 uv
uv venv
source .venv/bin/activate
uv pip install llm-api-router

# 或使用 pip
python -m venv venv
source venv/bin/activate
pip install llm-api-router
```

### Q: 如何安装特定功能的依赖？

```bash
# 安装 CLI 工具
pip install llm-api-router[cli]

# 安装文档工具
pip install llm-api-router[docs]

# 安装所有功能
pip install llm-api-router[cli,docs]

# 开发者安装
pip install llm-api-router[dev]
```

## API Key 问题

### Q: 如何安全存储 API Key？

推荐使用环境变量：

```bash
# .env 文件
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

config = ProviderConfig(
    provider_type="openai",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

### Q: 收到 AuthenticationError 怎么办？

1. 检查 API Key 是否正确
2. 确认 API Key 是否有效（未过期、未被撤销）
3. 验证是否有足够的配额
4. 检查提供商账户状态

```python
from llm_api_router.exceptions import AuthenticationError

try:
    response = client.chat.completions.create(...)
except AuthenticationError as e:
    print(f"认证失败: {e}")
```

## 使用问题

### Q: 如何处理流式响应？

```python
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)

full_response = ""
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        full_response += content
        print(content, end="", flush=True)
```

### Q: 如何实现多轮对话？

使用 `ConversationManager`：

```python
from llm_api_router import ConversationManager

manager = ConversationManager(system_message="你是一个有用的助手")

# 第一轮
manager.add_user_message("你好")
response = client.chat.completions.create(messages=manager.get_messages())
manager.add_assistant_message(response.choices[0].message.content)

# 第二轮
manager.add_user_message("你叫什么名字？")
response = client.chat.completions.create(messages=manager.get_messages())
```

### Q: 如何使用 Function Calling？

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"调用函数: {tool_call.function.name}")
    print(f"参数: {tool_call.function.arguments}")
```

### Q: 如何切换不同的模型？

在请求时指定 model 参数：

```python
# 使用默认模型
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}]
)

# 使用特定模型
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o"  # 覆盖默认模型
)
```

## 性能问题

### Q: 如何减少延迟？

1. **使用缓存**：相同请求直接返回缓存结果
2. **使用流式**：尽早开始处理响应
3. **选择更快的模型**：如 gpt-3.5-turbo、claude-3-haiku
4. **减少 token 数**：精简 prompt

```python
# 启用缓存
from llm_api_router.cache import MemoryCache

cache = MemoryCache(max_size=1000, ttl=3600)
with Client(config, cache=cache) as client:
    # 相同请求会命中缓存
    response = client.chat.completions.create(...)
```

### Q: 如何处理速率限制？

使用 `RateLimiter`：

```python
from llm_api_router import RateLimiter

limiter = RateLimiter(
    requests_per_minute=60,
    tokens_per_minute=100000,
)

async with limiter.acquire():
    response = await client.chat.completions.create(...)
```

### Q: 如何提高并发性能？

使用异步客户端和连接池：

```python
import asyncio
from llm_api_router import AsyncClient

async def process_batch(prompts):
    async with AsyncClient(config) as client:
        tasks = [
            client.chat.completions.create(
                messages=[{"role": "user", "content": p}]
            )
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
```

## 错误处理

### Q: 收到 RateLimitError 怎么办？

```python
from llm_api_router.exceptions import RateLimitError
import time

try:
    response = client.chat.completions.create(...)
except RateLimitError as e:
    # 等待后重试
    wait_time = e.retry_after or 60
    time.sleep(wait_time)
    response = client.chat.completions.create(...)
```

### Q: 如何处理超时？

```python
from llm_api_router.exceptions import TimeoutError

config = ProviderConfig(
    provider_type="openai",
    api_key="sk-xxx",
    timeout=60.0,  # 增加超时时间
)

try:
    response = client.chat.completions.create(...)
except TimeoutError:
    print("请求超时，请重试")
```

### Q: 如何实现自动重试？

内置重试机制会自动处理临时错误：

```python
config = ProviderConfig(
    provider_type="openai",
    api_key="sk-xxx",
    max_retries=5,  # 最多重试5次
)
```

## Ollama 本地部署

### Q: 如何使用本地 Ollama？

```bash
# 1. 安装 Ollama
brew install ollama

# 2. 启动服务
ollama serve

# 3. 拉取模型
ollama pull llama3.2
```

```python
config = ProviderConfig(
    provider_type="ollama",
    api_key="",  # 不需要
    base_url="http://localhost:11434",
    default_model="llama3.2"
)
```

### Q: Ollama 连接失败怎么办？

1. 确认 Ollama 服务正在运行：`ollama serve`
2. 检查端口是否正确（默认 11434）
3. 确认模型已下载：`ollama list`

## CLI 工具

### Q: CLI 命令找不到？

确保安装了 CLI 依赖：

```bash
pip install llm-api-router[cli]
```

或者使用 Python 模块方式运行：

```bash
python -m llm_api_router.cli --help
```

### Q: 如何使用 CLI 测试连接？

```bash
# 测试 OpenAI
llm-router test openai --api-key sk-xxx

# 测试 Anthropic
llm-router test anthropic --api-key sk-ant-xxx

# 测试本地 Ollama
llm-router test ollama --base-url http://localhost:11434
```

## 其他问题

### Q: 支持哪些 Python 版本？

Python 3.10 及以上版本。

### Q: 如何获取 token 使用统计？

```python
response = client.chat.completions.create(...)
print(f"输入 tokens: {response.usage.prompt_tokens}")
print(f"输出 tokens: {response.usage.completion_tokens}")
print(f"总 tokens: {response.usage.total_tokens}")
```

### Q: 如何报告 bug 或提出建议？

请在 GitHub 仓库提交 Issue：

1. 描述问题或建议
2. 提供复现步骤
3. 附上相关代码和错误信息
4. 注明 Python 版本和依赖版本

### Q: 如何贡献代码？

1. Fork 仓库
2. 创建特性分支
3. 编写代码和测试
4. 提交 Pull Request

参见 [贡献指南](contributing.md)。
