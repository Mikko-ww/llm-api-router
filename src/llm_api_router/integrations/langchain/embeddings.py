"""LangChain Embeddings 接口实现

实现 LangChain 的 Embeddings 接口，允许 LLM API Router 作为 LangChain 的嵌入模型使用。
"""

from typing import List, Optional

try:
    from langchain_core.embeddings import Embeddings
except ImportError as e:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install it with: pip install langchain-core"
    ) from e

from ...types import ProviderConfig
from ...client import Client, AsyncClient


class RouterEmbeddings(Embeddings):
    """LangChain Embeddings 接口实现
    
    将 LLM API Router 作为 LangChain 的嵌入模型使用。
    
    Example:
        ```python
        from llm_api_router.integrations.langchain import RouterEmbeddings
        from llm_api_router import ProviderConfig
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="your-api-key",
            default_model="text-embedding-3-small"
        )
        
        embeddings = RouterEmbeddings(config=config)
        
        # 嵌入单个文档
        vector = embeddings.embed_query("Hello, world!")
        
        # 嵌入多个文档
        vectors = embeddings.embed_documents(["Hello", "World"])
        ```
    
    与 LangChain VectorStore 配合使用:
        ```python
        from langchain_community.vectorstores import FAISS
        
        # 创建向量存储
        texts = ["Document 1", "Document 2", "Document 3"]
        vectorstore = FAISS.from_texts(texts, embeddings)
        
        # 相似性搜索
        results = vectorstore.similarity_search("query text")
        ```
    """
    
    config: ProviderConfig
    """LLM API Router 配置"""
    
    model: Optional[str] = None
    """模型名称，覆盖 config 中的默认模型"""
    
    encoding_format: Optional[str] = None
    """编码格式: "float" 或 "base64"，默认为 "float" """
    
    dimensions: Optional[int] = None
    """输出向量维度（仅部分模型支持）"""
    
    _client: Optional[Client] = None
    """同步客户端实例"""
    
    def __init__(
        self,
        config: ProviderConfig,
        model: Optional[str] = None,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """初始化 RouterEmbeddings
        
        Args:
            config: LLM API Router 配置
            model: 模型名称，覆盖 config 中的默认模型
            encoding_format: 编码格式
            dimensions: 输出向量维度
        """
        self.config = config
        self.model = model
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self._client = None
    
    def _get_client(self) -> Client:
        """获取或创建客户端实例"""
        if self._client is None:
            self._client = Client(self.config)
        return self._client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表
        
        Args:
            texts: 要嵌入的文档列表
            
        Returns:
            嵌入向量列表，每个向量是一个浮点数列表
        """
        if not texts:
            return []
        
        client = self._get_client()
        
        response = client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
        )
        
        # 按索引排序，确保顺序正确
        sorted_data = sorted(response.data, key=lambda x: x.index)
        
        return [item.embedding for item in sorted_data]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本
        
        Args:
            text: 要嵌入的查询文本
            
        Returns:
            嵌入向量
        """
        embeddings = self.embed_documents([text])
        return embeddings[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档列表
        
        Args:
            texts: 要嵌入的文档列表
            
        Returns:
            嵌入向量列表，每个向量是一个浮点数列表
        """
        if not texts:
            return []
        
        async with AsyncClient(self.config) as client:
            response = await client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format=self.encoding_format,
                dimensions=self.dimensions,
            )
            
            # 按索引排序，确保顺序正确
            sorted_data = sorted(response.data, key=lambda x: x.index)
            
            return [item.embedding for item in sorted_data]
    
    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入单个查询文本
        
        Args:
            text: 要嵌入的查询文本
            
        Returns:
            嵌入向量
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
    
    def __del__(self):
        """清理客户端资源"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
