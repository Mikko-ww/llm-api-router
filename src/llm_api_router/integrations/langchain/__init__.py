"""LangChain 集成模块

提供与 LangChain 框架的无缝集成，支持:
- LLM: 基础语言模型接口
- ChatModel: 聊天模型接口
- Embeddings: 文本嵌入接口

使用示例:
    ```python
    from llm_api_router.integrations.langchain import RouterLLM, RouterChatModel, RouterEmbeddings
    from llm_api_router import ProviderConfig
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key",
        default_model="gpt-4o-mini"
    )
    
    # 作为 LangChain LLM 使用
    llm = RouterLLM(config=config)
    result = llm.invoke("Hello, world!")
    
    # 作为 LangChain ChatModel 使用
    chat = RouterChatModel(config=config)
    result = chat.invoke([HumanMessage(content="Hello!")])
    
    # 作为 LangChain Embeddings 使用
    embeddings = RouterEmbeddings(config=config)
    vectors = embeddings.embed_documents(["text1", "text2"])
    ```
"""

from .llm import RouterLLM
from .chat_model import RouterChatModel
from .embeddings import RouterEmbeddings

__all__ = [
    "RouterLLM",
    "RouterChatModel",
    "RouterEmbeddings",
]
