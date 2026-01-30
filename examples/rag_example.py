#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) 应用示例

演示如何结合检索和生成，构建一个简单的知识问答系统。
"""

import os
from dataclasses import dataclass
from typing import Optional
from llm_api_router import Client, ProviderConfig, PromptTemplate


@dataclass
class Document:
    """文档数据类"""
    id: str
    title: str
    content: str
    metadata: dict = None


class SimpleVectorStore:
    """
    简单的向量存储（演示用）
    
    生产环境建议使用专业向量数据库：
    - Pinecone
    - Weaviate
    - Milvus
    - Chroma
    - Qdrant
    """
    
    def __init__(self, client: Client):
        self.client = client
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []
    
    def add_documents(self, documents: list[Document]):
        """添加文档到存储"""
        for doc in documents:
            self.documents.append(doc)
            # 获取文档嵌入
            embedding = self._get_embedding(doc.content)
            self.embeddings.append(embedding)
        print(f"✓ 已添加 {len(documents)} 个文档")
    
    def _get_embedding(self, text: str) -> list[float]:
        """获取文本嵌入向量"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    
    def search(self, query: str, top_k: int = 3) -> list[tuple[Document, float]]:
        """搜索最相关的文档"""
        if not self.documents:
            return []
        
        # 获取查询嵌入
        query_embedding = self._get_embedding(query)
        
        # 计算相似度并排序
        similarities = []
        for i, (doc, emb) in enumerate(zip(self.documents, self.embeddings)):
            score = self._cosine_similarity(query_embedding, emb)
            similarities.append((doc, score))
        
        # 返回 top-k 结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class RAGApplication:
    """RAG 应用"""
    
    def __init__(
        self,
        provider_type: str = "openai",
        model: Optional[str] = None,
    ):
        self.config = self._create_config(provider_type, model)
        self.client = Client(self.config)
        self.vector_store = SimpleVectorStore(self.client)
        
        # RAG 提示模板
        self.qa_template = PromptTemplate(
            name="rag_qa",
            template="""根据以下上下文信息回答用户问题。如果上下文中没有相关信息，请诚实地说不知道。

上下文信息：
{context}

用户问题：{question}

请用中文回答，确保回答准确、简洁且有帮助。""",
            variables=["context", "question"],
        )
    
    def _create_config(self, provider_type: str, model: Optional[str]) -> ProviderConfig:
        """创建提供商配置"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        return ProviderConfig(
            provider_type=provider_type,
            api_key=api_key,
            default_model=model or "gpt-4o",
        )
    
    def ingest(self, documents: list[Document]):
        """导入文档"""
        self.vector_store.add_documents(documents)
    
    def query(self, question: str, top_k: int = 3) -> str:
        """
        查询问答
        
        Args:
            question: 用户问题
            top_k: 检索的文档数量
        
        Returns:
            生成的回答
        """
        # 1. 检索相关文档
        results = self.vector_store.search(question, top_k=top_k)
        
        if not results:
            return "抱歉，我没有找到相关信息。"
        
        # 2. 构建上下文
        context_parts = []
        print("\n📚 检索到的相关文档:")
        for doc, score in results:
            context_parts.append(f"[{doc.title}]\n{doc.content}")
            print(f"  - {doc.title} (相似度: {score:.3f})")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 3. 生成回答
        prompt = self.qa_template.render(
            context=context,
            question=question,
        )
        
        print("\n🤔 正在生成回答...")
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content
    
    def close(self):
        """关闭客户端"""
        self.client.close()


# 示例知识库
SAMPLE_DOCUMENTS = [
    Document(
        id="1",
        title="Python 简介",
        content="""Python 是一种高级、解释型、通用的编程语言。由 Guido van Rossum 于 1991 年首次发布。
Python 的设计哲学强调代码的可读性，使用显著的缩进来定义代码块。
Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python 拥有庞大的标准库和活跃的社区，被广泛应用于 Web 开发、数据科学、人工智能、自动化等领域。"""
    ),
    Document(
        id="2",
        title="机器学习基础",
        content="""机器学习是人工智能的一个分支，通过数据和算法让计算机自动学习和改进。
主要类型包括：
1. 监督学习：使用标记数据训练模型，如分类和回归
2. 无监督学习：发现未标记数据中的模式，如聚类和降维
3. 强化学习：智能体通过与环境交互学习最优策略
常用算法包括线性回归、决策树、随机森林、支持向量机、神经网络等。"""
    ),
    Document(
        id="3",
        title="深度学习概述",
        content="""深度学习是机器学习的子领域，基于人工神经网络，特别是深层神经网络。
关键技术包括：
1. 卷积神经网络 (CNN)：主要用于图像处理
2. 循环神经网络 (RNN)：处理序列数据
3. Transformer：自注意力机制，革新了 NLP 领域
4. 生成对抗网络 (GAN)：生成逼真的数据
深度学习推动了计算机视觉、自然语言处理、语音识别等领域的重大突破。"""
    ),
    Document(
        id="4",
        title="大语言模型 (LLM)",
        content="""大语言模型是一种基于深度学习的自然语言处理模型，通过海量文本数据进行训练。
代表性模型包括：
1. GPT 系列 (OpenAI)：GPT-3, GPT-4
2. Claude (Anthropic)：注重安全性和有用性
3. LLaMA (Meta)：开源模型
4. Gemini (Google)：多模态能力
LLM 能够理解和生成人类语言，应用于聊天机器人、代码生成、文本摘要、翻译等任务。"""
    ),
    Document(
        id="5",
        title="RAG 技术",
        content="""RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术。
工作流程：
1. 检索：根据用户查询从知识库中检索相关文档
2. 增强：将检索到的文档作为上下文
3. 生成：LLM 基于上下文生成回答
优势：
- 减少幻觉：基于真实数据生成
- 知识更新：无需重新训练模型
- 可解释性：可以追溯信息来源
RAG 广泛应用于企业知识问答、客服系统等场景。"""
    ),
]


def main():
    """演示 RAG 应用"""
    print("=" * 50)
    print("🔍 RAG 知识问答系统示例")
    print("=" * 50)
    
    try:
        # 初始化应用
        print("\n正在初始化...")
        app = RAGApplication()
        
        # 导入示例文档
        print("\n📥 导入知识库...")
        app.ingest(SAMPLE_DOCUMENTS)
        
        # 示例查询
        questions = [
            "Python 是什么？有什么特点？",
            "什么是深度学习？有哪些关键技术？",
            "RAG 技术是如何工作的？",
        ]
        
        for q in questions:
            print("\n" + "=" * 50)
            print(f"❓ 问题: {q}")
            print("=" * 50)
            
            answer = app.query(q)
            print(f"\n💡 回答:\n{answer}")
        
        # 交互式问答
        print("\n" + "=" * 50)
        print("📝 交互式问答（输入 'quit' 退出）")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n你的问题: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    break
                if not question:
                    continue
                
                answer = app.query(question)
                print(f"\n💡 回答:\n{answer}")
                
            except KeyboardInterrupt:
                break
        
        print("\n再见！👋")
        
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'app' in locals():
            app.close()


if __name__ == "__main__":
    main()
