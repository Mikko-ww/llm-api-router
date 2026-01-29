"""
Embeddings API 使用示例

本示例演示如何使用 LLM API Router 的 Embeddings API 创建文本嵌入向量。
支持 OpenAI、Gemini、Zhipu、Aliyun 等多个提供商。
"""
import asyncio
from llm_api_router import Client, AsyncClient, ProviderConfig


def basic_embedding_example():
    """基础嵌入示例：创建单个文本的嵌入向量"""
    print("=" * 50)
    print("基础嵌入示例")
    print("=" * 50)
    
    # 配置 OpenAI 提供商
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-openai-api-key",  # 替换为您的 API key
        default_model="text-embedding-3-small"
    )
    
    with Client(config) as client:
        # 创建单个文本的嵌入
        response = client.embeddings.create(
            input="Hello, world!"
        )
        
        print(f"模型: {response.model}")
        print(f"Token 使用量: {response.usage.total_tokens}")
        print(f"嵌入向量维度: {len(response.data[0].embedding)}")
        print(f"前5个值: {response.data[0].embedding[:5]}")


def batch_embedding_example():
    """批量嵌入示例：同时处理多个文本"""
    print("\n" + "=" * 50)
    print("批量嵌入示例")
    print("=" * 50)
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-openai-api-key",
        default_model="text-embedding-3-small"
    )
    
    texts = [
        "机器学习是人工智能的一个子领域",
        "深度学习使用神经网络来处理数据",
        "自然语言处理让计算机理解人类语言",
        "计算机视觉让机器能够理解图像"
    ]
    
    with Client(config) as client:
        response = client.embeddings.create(input=texts)
        
        print(f"处理了 {len(response.data)} 个文本")
        print(f"总 Token 使用量: {response.usage.total_tokens}")
        
        for item in response.data:
            print(f"  文本 {item.index}: 向量维度 {len(item.embedding)}")


def custom_dimensions_example():
    """自定义维度示例：使用较小的嵌入维度"""
    print("\n" + "=" * 50)
    print("自定义维度示例")
    print("=" * 50)
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-openai-api-key",
        default_model="text-embedding-3-small"
    )
    
    with Client(config) as client:
        # OpenAI 的 text-embedding-3-* 模型支持自定义维度
        response = client.embeddings.create(
            input="测试自定义维度",
            dimensions=256  # 降低维度以节省存储空间
        )
        
        print(f"嵌入向量维度: {len(response.data[0].embedding)}")


def multi_provider_example():
    """多提供商示例：使用不同的提供商创建嵌入"""
    print("\n" + "=" * 50)
    print("多提供商示例")
    print("=" * 50)
    
    text = "这是一段测试文本"
    
    # OpenAI
    openai_config = ProviderConfig(
        provider_type="openai",
        api_key="your-openai-api-key",
        default_model="text-embedding-3-small"
    )
    
    # Zhipu (智谱)
    zhipu_config = ProviderConfig(
        provider_type="zhipu",
        api_key="your-zhipu-api-key",
        default_model="embedding-3"
    )
    
    # Gemini
    gemini_config = ProviderConfig(
        provider_type="gemini",
        api_key="your-gemini-api-key",
        default_model="embedding-001"
    )
    
    providers = [
        ("OpenAI", openai_config),
        ("Zhipu", zhipu_config),
        ("Gemini", gemini_config),
    ]
    
    for name, config in providers:
        try:
            with Client(config) as client:
                response = client.embeddings.create(input=text)
                print(f"{name}: 维度={len(response.data[0].embedding)}")
        except Exception as e:
            print(f"{name}: 错误 - {e}")


async def async_embedding_example():
    """异步嵌入示例：使用异步客户端"""
    print("\n" + "=" * 50)
    print("异步嵌入示例")
    print("=" * 50)
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-openai-api-key",
        default_model="text-embedding-3-small"
    )
    
    async with AsyncClient(config) as client:
        # 异步创建嵌入
        response = await client.embeddings.create(
            input=["异步嵌入测试1", "异步嵌入测试2"]
        )
        
        print(f"异步处理了 {len(response.data)} 个文本")
        print(f"Token 使用量: {response.usage.total_tokens}")


def similarity_search_example():
    """相似度搜索示例：使用嵌入进行文本相似度计算"""
    print("\n" + "=" * 50)
    print("相似度搜索示例")
    print("=" * 50)
    
    import math
    
    def cosine_similarity(vec1, vec2):
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (norm1 * norm2)
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-openai-api-key",
        default_model="text-embedding-3-small"
    )
    
    # 查询文本
    query = "如何学习编程"
    
    # 文档库
    documents = [
        "Python 是一门简单易学的编程语言",
        "今天天气很好，适合出去散步",
        "编程入门推荐从基础语法开始学习",
        "机器学习需要数学和编程基础",
        "周末我去了公园看花"
    ]
    
    with Client(config) as client:
        # 获取查询的嵌入
        query_response = client.embeddings.create(input=query)
        query_embedding = query_response.data[0].embedding
        
        # 获取所有文档的嵌入
        docs_response = client.embeddings.create(input=documents)
        
        # 计算相似度并排序
        similarities = []
        for i, doc in enumerate(documents):
            doc_embedding = docs_response.data[i].embedding
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, doc))
        
        # 按相似度排序
        similarities.sort(reverse=True)
        
        print(f"查询: \"{query}\"")
        print("\n最相似的文档:")
        for sim, doc in similarities[:3]:
            print(f"  [{sim:.4f}] {doc}")


if __name__ == "__main__":
    # 注意：运行示例前请设置有效的 API key
    
    print("LLM API Router - Embeddings API 示例")
    print("=" * 50)
    print("注意：请替换示例中的 API key 后再运行")
    print()
    
    # 取消注释以运行各个示例
    # basic_embedding_example()
    # batch_embedding_example()
    # custom_dimensions_example()
    # multi_provider_example()
    # asyncio.run(async_embedding_example())
    # similarity_search_example()
    
    print("\n所有示例代码已准备就绪。")
    print("请取消注释相应函数调用并设置有效的 API key 来运行示例。")
