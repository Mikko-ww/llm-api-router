"""LangChain 集成示例

本示例展示如何将 LLM API Router 与 LangChain 框架集成使用。

运行前请确保:
1. 安装依赖: pip install llm-api-router[langchain]
2. 设置环境变量: export OPENAI_API_KEY="your-api-key"
"""

import os
import asyncio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入 LLM API Router
from llm_api_router import ProviderConfig
from llm_api_router.integrations.langchain import RouterLLM, RouterChatModel, RouterEmbeddings

# 导入 LangChain
try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    print("请先安装 langchain-core: pip install langchain-core")
    exit(1)


def get_config() -> ProviderConfig:
    """创建 provider 配置"""
    return ProviderConfig(
        provider_type="openai",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model="gpt-4o-mini",
    )


# ========== 示例 1: 基础 LLM 使用 ==========

def example_basic_llm():
    """使用 RouterLLM 作为基础语言模型"""
    print("\n" + "=" * 50)
    print("示例 1: 基础 LLM 使用")
    print("=" * 50)
    
    config = get_config()
    llm = RouterLLM(config=config, temperature=0.7)
    
    # 直接调用
    result = llm.invoke("请用一句话解释什么是机器学习？")
    print(f"\n问: 请用一句话解释什么是机器学习？")
    print(f"答: {result}")


# ========== 示例 2: ChatModel 多轮对话 ==========

def example_chat_model():
    """使用 RouterChatModel 进行多轮对话"""
    print("\n" + "=" * 50)
    print("示例 2: ChatModel 多轮对话")
    print("=" * 50)
    
    config = get_config()
    chat = RouterChatModel(config=config, temperature=0.7)
    
    # 多轮对话
    messages = [
        SystemMessage(content="你是一个有趣的助手，喜欢用幽默的方式回答问题。"),
        HumanMessage(content="你好，请用有趣的方式介绍一下自己。"),
    ]
    
    result = chat.invoke(messages)
    print(f"\n系统: 你是一个有趣的助手，喜欢用幽默的方式回答问题。")
    print(f"用户: 你好，请用有趣的方式介绍一下自己。")
    print(f"助手: {result.content}")
    
    # 继续对话
    messages.append(result)
    messages.append(HumanMessage(content="你能讲个关于程序员的笑话吗？"))
    
    result2 = chat.invoke(messages)
    print(f"\n用户: 你能讲个关于程序员的笑话吗？")
    print(f"助手: {result2.content}")


# ========== 示例 3: 使用 LCEL 链 ==========

def example_lcel_chain():
    """使用 LangChain Expression Language (LCEL) 构建链"""
    print("\n" + "=" * 50)
    print("示例 3: 使用 LCEL 构建链")
    print("=" * 50)
    
    config = get_config()
    chat = RouterChatModel(config=config, temperature=0.7)
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的翻译，擅长将{source_lang}翻译成{target_lang}。"),
        ("human", "请翻译: {text}"),
    ])
    
    # 创建链
    chain = prompt | chat | StrOutputParser()
    
    # 调用链
    result = chain.invoke({
        "source_lang": "中文",
        "target_lang": "英文",
        "text": "人工智能正在改变我们的生活方式。"
    })
    
    print(f"\n原文: 人工智能正在改变我们的生活方式。")
    print(f"翻译: {result}")


# ========== 示例 4: Embeddings 使用 ==========

def example_embeddings():
    """使用 RouterEmbeddings 创建文本嵌入"""
    print("\n" + "=" * 50)
    print("示例 4: Embeddings 使用")
    print("=" * 50)
    
    config = ProviderConfig(
        provider_type="openai",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        default_model="text-embedding-3-small",  # 使用嵌入模型
    )
    
    embeddings = RouterEmbeddings(config=config)
    
    # 嵌入单个查询
    query = "什么是人工智能？"
    query_vector = embeddings.embed_query(query)
    print(f"\n查询文本: {query}")
    print(f"向量维度: {len(query_vector)}")
    print(f"向量前5维: {query_vector[:5]}")
    
    # 批量嵌入文档
    documents = [
        "机器学习是人工智能的一个子领域。",
        "深度学习使用神经网络处理复杂问题。",
        "自然语言处理让计算机理解人类语言。",
    ]
    
    doc_vectors = embeddings.embed_documents(documents)
    print(f"\n文档数量: {len(documents)}")
    print(f"嵌入向量数量: {len(doc_vectors)}")


# ========== 示例 5: 流式输出 ==========

def example_streaming():
    """使用流式输出"""
    print("\n" + "=" * 50)
    print("示例 5: 流式输出")
    print("=" * 50)
    
    config = get_config()
    chat = RouterChatModel(config=config, streaming=True)
    
    messages = [
        SystemMessage(content="你是一个讲故事的高手。"),
        HumanMessage(content="讲一个简短的科幻故事，不超过100字。"),
    ]
    
    print(f"\n用户: 讲一个简短的科幻故事，不超过100字。")
    print("助手: ", end="", flush=True)
    
    # 流式输出
    for chunk in chat.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    
    print()  # 换行


# ========== 示例 6: 异步调用 ==========

async def example_async():
    """异步调用示例"""
    print("\n" + "=" * 50)
    print("示例 6: 异步调用")
    print("=" * 50)
    
    config = get_config()
    chat = RouterChatModel(config=config, temperature=0.7)
    embeddings = RouterEmbeddings(
        config=ProviderConfig(
            provider_type="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            default_model="text-embedding-3-small",
        )
    )
    
    # 并行执行多个任务
    chat_task = chat.ainvoke([HumanMessage(content="用一个词形容AI的发展速度")])
    embed_task = embeddings.aembed_query("人工智能")
    
    # 等待所有任务完成
    chat_result, embed_result = await asyncio.gather(chat_task, embed_task)
    
    print(f"\n聊天结果: {chat_result.content}")
    print(f"嵌入维度: {len(embed_result)}")


# ========== 示例 7: 切换不同 Provider ==========

def example_multi_provider():
    """演示如何在不同 provider 之间切换"""
    print("\n" + "=" * 50)
    print("示例 7: 多 Provider 切换")
    print("=" * 50)
    
    # 定义多个 provider 配置
    providers = {
        "openai": ProviderConfig(
            provider_type="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            default_model="gpt-4o-mini",
        ),
        # 如果有其他 provider 的 API key，可以取消注释
        # "anthropic": ProviderConfig(
        #     provider_type="anthropic",
        #     api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        #     default_model="claude-3-haiku-20240307",
        # ),
    }
    
    question = "什么是量子计算？用一句话回答。"
    
    for name, config in providers.items():
        if not config.api_key:
            print(f"\n跳过 {name}（未配置 API key）")
            continue
            
        chat = RouterChatModel(config=config, temperature=0.7)
        result = chat.invoke([HumanMessage(content=question)])
        
        print(f"\n{name.upper()} 的回答:")
        print(f"  {result.content}")


# ========== 示例 8: 与 LangChain 工具集成 ==========

def example_with_tools():
    """与 LangChain 工具集成的示例"""
    print("\n" + "=" * 50)
    print("示例 8: Tool Calling (Function Calling)")
    print("=" * 50)
    
    config = get_config()
    chat = RouterChatModel(config=config, temperature=0)
    
    # 定义工具
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'"
                    }
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
    
    print(f"\n用户: 北京今天天气怎么样？")
    
    if result.tool_calls:
        print(f"模型请求调用工具: {result.tool_calls[0]['name']}")
        print(f"工具参数: {result.tool_calls[0]['args']}")
    else:
        print(f"助手: {result.content}")


def main():
    """运行所有示例"""
    print("=" * 50)
    print("LLM API Router - LangChain 集成示例")
    print("=" * 50)
    
    # 检查 API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n警告: 未设置 OPENAI_API_KEY 环境变量")
        print("请设置环境变量后再运行示例")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # 运行同步示例
        example_basic_llm()
        example_chat_model()
        example_lcel_chain()
        example_embeddings()
        example_streaming()
        example_multi_provider()
        example_with_tools()
        
        # 运行异步示例
        asyncio.run(example_async())
        
        print("\n" + "=" * 50)
        print("所有示例运行完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
