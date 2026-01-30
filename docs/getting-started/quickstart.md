# 快速开始

本指南帮助你在 5 分钟内开始使用 LLM API Router。

## 第一步：安装

```bash
pip install llm-api-router
```

## 第二步：获取 API Key

从你的 LLM 提供商获取 API Key：

- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com)
- **Google Gemini**: [aistudio.google.com](https://aistudio.google.com)

## 第三步：发送第一个请求

```python
from llm_api_router import Client, ProviderConfig

# 配置提供商
config = ProviderConfig(
    provider_type="openai",
    api_key="sk-your-api-key",  # 替换为你的 API Key
    default_model="gpt-3.5-turbo"
)

# 创建客户端并发送请求
with Client(config) as client:
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "用一句话介绍自己"}
        ]
    )
    print(response.choices[0].message.content)
```

## 使用环境变量

推荐将 API Key 存储在环境变量中：

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

```python
import os
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    default_model="gpt-3.5-turbo"
)
```

## 流式响应

获取实时流式输出：

```python
with Client(config) as client:
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "写一首关于春天的诗"}],
        stream=True
    )
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
```

## 异步调用

使用 `AsyncClient` 进行异步操作：

```python
import asyncio
from llm_api_router import AsyncClient, ProviderConfig

async def main():
    config = ProviderConfig(
        provider_type="openai",
        api_key="sk-xxx",
        default_model="gpt-3.5-turbo"
    )
    
    async with AsyncClient(config) as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

## 切换提供商

只需更改配置即可切换到不同的提供商：

=== "OpenAI"

    ```python
    config = ProviderConfig(
        provider_type="openai",
        api_key="sk-xxx",
        default_model="gpt-4o"
    )
    ```

=== "Anthropic"

    ```python
    config = ProviderConfig(
        provider_type="anthropic",
        api_key="sk-ant-xxx",
        default_model="claude-3-5-sonnet-20241022"
    )
    ```

=== "Google Gemini"

    ```python
    config = ProviderConfig(
        provider_type="gemini",
        api_key="AIza-xxx",
        default_model="gemini-1.5-flash"
    )
    ```

=== "本地 Ollama"

    ```python
    config = ProviderConfig(
        provider_type="ollama",
        api_key="not-required",
        base_url="http://localhost:11434",
        default_model="llama3.2"
    )
    ```

## 使用 CLI 快速测试

安装 CLI 工具后，可以快速测试连接：

```bash
# 安装 CLI
pip install llm-api-router[cli]

# 测试连接
llm-router test openai --api-key sk-xxx

# 交互式聊天
llm-router chat openai --api-key sk-xxx
```

## 下一步

- 查看 [提供商配置](../user-guide/providers.md) 了解各提供商详情
- 学习 [高级配置](../user-guide/configuration.md) 自定义客户端行为
- 探索 [API 参考](../api-reference/client.md) 了解完整功能
