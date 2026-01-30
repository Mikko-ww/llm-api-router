# LangChain 集成

LLM API Router 提供与 [LangChain](https://www.langchain.com/) 框架的无缝集成，允许您在 LangChain 应用中使用统一的 LLM 路由能力。

## 安装

首先安装带有 LangChain 支持的 LLM API Router：

```bash
pip install llm-api-router[langchain]
```

或者单独安装 langchain-core：

```bash
pip install llm-api-router langchain-core
```

## 可用组件

LangChain 集成提供三个核心组件：

| 组件 | 用途 | LangChain 基类 |
|------|------|---------------|
| `RouterLLM` | 基础语言模型 | `langchain_core.language_models.llms.LLM` |
| `RouterChatModel` | 聊天模型 | `langchain_core.language_models.chat_models.BaseChatModel` |
| `RouterEmbeddings` | 文本嵌入 | `langchain_core.embeddings.Embeddings` |

## 快速开始

### RouterLLM - 基础语言模型

```python
from llm_api_router import ProviderConfig
from llm_api_router.integrations.langchain import RouterLLM

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="gpt-4o-mini"
)

llm = RouterLLM(config=config, temperature=0.7)

# 直接调用
result = llm.invoke("什么是机器学习？")
print(result)
```

### RouterChatModel - 聊天模型

```python
from llm_api_router import ProviderConfig
from llm_api_router.integrations.langchain import RouterChatModel
from langchain_core.messages import HumanMessage, SystemMessage

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="gpt-4o-mini"
)

chat = RouterChatModel(config=config, temperature=0.7)

messages = [
    SystemMessage(content="你是一个有帮助的助手。"),
    HumanMessage(content="你好！"),
]

result = chat.invoke(messages)
print(result.content)
```

### RouterEmbeddings - 文本嵌入

```python
from llm_api_router import ProviderConfig
from llm_api_router.integrations.langchain import RouterEmbeddings

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="text-embedding-3-small"
)

embeddings = RouterEmbeddings(config=config)

# 嵌入单个查询
vector = embeddings.embed_query("什么是人工智能？")
print(f"向量维度: {len(vector)}")

# 批量嵌入文档
documents = ["文档1", "文档2", "文档3"]
vectors = embeddings.embed_documents(documents)
print(f"嵌入了 {len(vectors)} 个文档")
```

## 配置选项

### RouterLLM 和 RouterChatModel

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | `ProviderConfig` | 必需 | LLM API Router 配置 |
| `model` | `Optional[str]` | `None` | 覆盖默认模型 |
| `temperature` | `float` | `1.0` | 生成温度 |
| `max_tokens` | `Optional[int]` | `None` | 最大生成 token 数 |
| `top_p` | `Optional[float]` | `None` | Top-p 采样参数 |
| `stop` | `Optional[List[str]]` | `None` | 停止序列 |
| `streaming` | `bool` | `False` | 是否流式输出 |

### RouterEmbeddings

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | `ProviderConfig` | 必需 | LLM API Router 配置 |
| `model` | `Optional[str]` | `None` | 覆盖默认模型 |
| `encoding_format` | `Optional[str]` | `None` | 编码格式 |
| `dimensions` | `Optional[int]` | `None` | 输出向量维度 |

## 高级用法

### 使用 LCEL 链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}。"),
    ("human", "{question}"),
])

# 创建链
chain = prompt | chat | StrOutputParser()

# 调用链
result = chain.invoke({
    "role": "翻译",
    "question": "请将 'Hello World' 翻译成中文"
})
```

### 流式输出

```python
chat = RouterChatModel(config=config, streaming=True)

messages = [HumanMessage(content="讲一个故事")]

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```

### 异步调用

```python
import asyncio

async def main():
    chat = RouterChatModel(config=config)
    embeddings = RouterEmbeddings(config=embed_config)
    
    # 并行执行
    chat_task = chat.ainvoke([HumanMessage(content="你好")])
    embed_task = embeddings.aembed_query("Hello")
    
    chat_result, embed_result = await asyncio.gather(chat_task, embed_task)
    print(chat_result.content)
    print(len(embed_result))

asyncio.run(main())
```

### Tool Calling / Function Calling

```python
# 定义工具
tools = [{
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
}]

# 绑定工具
chat_with_tools = chat.bind(tools=tools)

# 调用
result = chat_with_tools.invoke([
    HumanMessage(content="北京今天天气怎么样？")
])

# 检查是否请求调用工具
if result.tool_calls:
    for tool_call in result.tool_calls:
        print(f"工具: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
```

### 处理工具调用结果

```python
from langchain_core.messages import ToolMessage

# 假设模型请求调用 get_weather 工具
result = chat_with_tools.invoke([
    HumanMessage(content="北京今天天气怎么样？")
])

# 模拟工具执行
def get_weather(city: str) -> str:
    return f"{city}今天晴朗，气温 25°C"

if result.tool_calls:
    # 执行工具
    tool_result = get_weather(result.tool_calls[0]["args"]["city"])
    
    # 继续对话，包含工具结果
    final_result = chat_with_tools.invoke([
        HumanMessage(content="北京今天天气怎么样？"),
        result,  # AI 的工具调用消息
        ToolMessage(
            content=tool_result,
            tool_call_id=result.tool_calls[0]["id"]
        )
    ])
    
    print(final_result.content)
```

### 与 VectorStore 配合使用

```python
from langchain_community.vectorstores import FAISS

# 创建嵌入模型
embeddings = RouterEmbeddings(config=embed_config)

# 准备文档
texts = [
    "LangChain 是一个用于构建 LLM 应用的框架。",
    "LLM API Router 提供统一的 LLM API 访问。",
    "向量数据库用于存储和检索嵌入向量。",
]

# 创建向量存储
vectorstore = FAISS.from_texts(texts, embeddings)

# 相似性搜索
results = vectorstore.similarity_search("什么是 LangChain？", k=2)
for doc in results:
    print(doc.page_content)
```

## 切换 Provider

LangChain 集成的主要优势是可以轻松切换不同的 LLM 提供商：

```python
# OpenAI
openai_chat = RouterChatModel(config=ProviderConfig(
    provider_type="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    default_model="gpt-4o-mini"
))

# Anthropic
anthropic_chat = RouterChatModel(config=ProviderConfig(
    provider_type="anthropic",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    default_model="claude-3-haiku-20240307"
))

# 智谱 AI
zhipu_chat = RouterChatModel(config=ProviderConfig(
    provider_type="zhipu",
    api_key=os.getenv("ZHIPU_API_KEY"),
    default_model="glm-4-flash"
))

# 同一个链可以使用不同的模型
chain = prompt | openai_chat | StrOutputParser()
# 或者
chain = prompt | anthropic_chat | StrOutputParser()
```

## 与其他 LangChain 功能集成

### 使用 Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

chat = RouterChatModel(config=config)

# 注意：LangChain 的 Memory 功能需要使用 langchain 包而非 langchain-core
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

response1 = conversation.predict(input="你好，我叫小明")
response2 = conversation.predict(input="你还记得我的名字吗？")
```

### 使用 Agents

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"

chat = RouterChatModel(config=config)

# 创建代理
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(chat, [search], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search])

result = agent_executor.invoke({"input": "搜索关于 AI 的信息"})
```

## 错误处理

```python
from llm_api_router.exceptions import (
    AuthenticationError,
    RateLimitError,
    ProviderError
)

try:
    result = chat.invoke(messages)
except AuthenticationError:
    print("API 密钥无效")
except RateLimitError:
    print("请求过于频繁")
except ProviderError as e:
    print(f"提供商错误: {e}")
```

## 最佳实践

1. **重用客户端实例**：创建 `RouterChatModel` 或 `RouterEmbeddings` 实例后尽量复用，避免频繁创建新实例。

2. **使用合适的模型**：对于嵌入任务，使用专门的嵌入模型（如 `text-embedding-3-small`），而非聊天模型。

3. **批量处理嵌入**：使用 `embed_documents` 批量处理文档，比多次调用 `embed_query` 更高效。

4. **异步操作**：在高并发场景下使用异步方法（`ainvoke`, `astream`, `aembed_documents`）。

5. **配置缓存**：为重复请求启用缓存可以显著降低成本和延迟。

```python
from llm_api_router import ProviderConfig, CacheConfig

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    cache_config=CacheConfig(
        enabled=True,
        ttl=3600,  # 1 小时
        max_size=1000
    )
)

chat = RouterChatModel(config=config)
```

## 完整示例

查看 [examples/langchain_integration_example.py](https://github.com/Mikko-ww/llm-api-router/blob/main/examples/langchain_integration_example.py) 获取更多完整示例。
