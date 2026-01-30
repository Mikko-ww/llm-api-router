#!/usr/bin/env python3
"""
快速开始示例

最简单的 LLM API Router 使用方式，5 行代码即可完成。
"""

import os
from llm_api_router import Client, ProviderConfig

# 配置提供商（使用环境变量存储 API Key）
config = ProviderConfig(
    provider_type="openai",
    api_key=os.environ.get("OPENAI_API_KEY", "sk-your-api-key"),
    default_model="gpt-3.5-turbo"
)

# 发送请求并获取响应
with Client(config) as client:
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "用一句话介绍 Python 编程语言"}
        ]
    )
    print(response.choices[0].message.content)

# 输出示例：
# Python 是一种简洁、易读且功能强大的高级编程语言，广泛应用于Web开发、数据科学、人工智能等领域。
