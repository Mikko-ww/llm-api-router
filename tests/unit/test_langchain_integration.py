"""Unit tests for LangChain integration.

这些测试使用 mock 来避免实际的 API 调用，专注于测试接口的正确实现。
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List

# 检查 langchain-core 是否可用
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from llm_api_router.types import (
    ProviderConfig, UnifiedResponse, UnifiedChunk, Message, Choice, ChunkChoice, Usage,
    EmbeddingResponse, Embedding, EmbeddingUsage
)


# 跳过测试如果 langchain 不可用
pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE,
    reason="langchain-core is not installed"
)


@pytest.fixture
def langchain_provider_config() -> ProviderConfig:
    """LangChain 集成测试用的 provider 配置"""
    return ProviderConfig(
        provider_type="openai",
        api_key="test-api-key",
        default_model="gpt-4o-mini"
    )


@pytest.fixture
def sample_chat_response() -> UnifiedResponse:
    """Sample chat response for testing."""
    return UnifiedResponse(
        id="chatcmpl-test123",
        object="chat.completion",
        created=1234567890,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Hello! I'm doing well, thank you!"),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18
        )
    )


@pytest.fixture
def sample_embedding_response() -> EmbeddingResponse:
    """Sample embedding response for testing."""
    return EmbeddingResponse(
        data=[
            Embedding(index=0, embedding=[0.1, 0.2, 0.3, 0.4, 0.5]),
            Embedding(index=1, embedding=[0.5, 0.4, 0.3, 0.2, 0.1]),
        ],
        model="text-embedding-3-small",
        usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
    )


class TestRouterLLM:
    """Test RouterLLM class."""

    def test_llm_initialization(self, langchain_provider_config):
        """Test RouterLLM initialization."""
        from llm_api_router.integrations.langchain import RouterLLM
        
        llm = RouterLLM(config=langchain_provider_config)
        
        assert llm.router_config == langchain_provider_config
        assert llm._llm_type == "llm-api-router-openai"
        assert llm.temperature == 1.0

    def test_llm_identifying_params(self, langchain_provider_config):
        """Test RouterLLM identifying parameters."""
        from llm_api_router.integrations.langchain import RouterLLM
        
        llm = RouterLLM(
            config=langchain_provider_config,
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=100,
        )
        
        params = llm._identifying_params
        assert params["provider_type"] == "openai"
        assert params["model_name"] == "gpt-4"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100

    def test_llm_call(self, langchain_provider_config, sample_chat_response):
        """Test RouterLLM _call method."""
        from llm_api_router.integrations.langchain import RouterLLM
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request = Mock(return_value=sample_chat_response)
            mock_factory.return_value = mock_provider
            
            llm = RouterLLM(config=langchain_provider_config)
            result = llm._call("Hello, how are you?")
            
            assert result == "Hello! I'm doing well, thank you!"
            mock_provider.send_request.assert_called_once()

    def test_llm_call_with_stop(self, langchain_provider_config, sample_chat_response):
        """Test RouterLLM _call method with stop sequences."""
        from llm_api_router.integrations.langchain import RouterLLM
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request = Mock(return_value=sample_chat_response)
            mock_factory.return_value = mock_provider
            
            llm = RouterLLM(config=langchain_provider_config)
            result = llm._call("Hello", stop=["stop1", "stop2"])
            
            # 验证 stop 序列被传递
            call_args = mock_provider.send_request.call_args
            request = call_args[0][1]
            assert request.stop == ["stop1", "stop2"]

    @pytest.mark.asyncio
    async def test_llm_acall(self, langchain_provider_config, sample_chat_response):
        """Test RouterLLM _acall method."""
        from llm_api_router.integrations.langchain import RouterLLM
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request_async = AsyncMock(return_value=sample_chat_response)
            mock_factory.return_value = mock_provider
            
            llm = RouterLLM(config=langchain_provider_config)
            result = await llm._acall("Hello, how are you?")
            
            assert result == "Hello! I'm doing well, thank you!"


class TestRouterChatModel:
    """Test RouterChatModel class."""

    def test_chat_model_initialization(self, langchain_provider_config):
        """Test RouterChatModel initialization."""
        from llm_api_router.integrations.langchain import RouterChatModel
        
        chat = RouterChatModel(config=langchain_provider_config)
        
        assert chat.router_config == langchain_provider_config
        assert chat._llm_type == "llm-api-router-chat-openai"
        assert chat.temperature == 1.0

    def test_chat_model_identifying_params(self, langchain_provider_config):
        """Test RouterChatModel identifying parameters."""
        from llm_api_router.integrations.langchain import RouterChatModel
        
        chat = RouterChatModel(
            config=langchain_provider_config,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=200,
        )
        
        params = chat._identifying_params
        assert params["provider_type"] == "openai"
        assert params["model_name"] == "gpt-4"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 200

    def test_chat_model_generate(self, langchain_provider_config, sample_chat_response):
        """Test RouterChatModel _generate method."""
        from llm_api_router.integrations.langchain import RouterChatModel
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request = Mock(return_value=sample_chat_response)
            mock_factory.return_value = mock_provider
            
            chat = RouterChatModel(config=langchain_provider_config)
            
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Hello, how are you?"),
            ]
            
            result = chat._generate(messages)
            
            assert isinstance(result, ChatResult)
            assert len(result.generations) == 1
            
            generation = result.generations[0]
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.message, AIMessage)
            assert generation.message.content == "Hello! I'm doing well, thank you!"
            
            # 验证 token 用量
            assert result.llm_output["token_usage"]["total_tokens"] == 18

    def test_chat_model_message_conversion(self, langchain_provider_config, sample_chat_response):
        """Test message type conversion."""
        from llm_api_router.integrations.langchain import RouterChatModel
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request = Mock(return_value=sample_chat_response)
            mock_factory.return_value = mock_provider
            
            chat = RouterChatModel(config=langchain_provider_config)
            
            messages = [
                SystemMessage(content="System prompt"),
                HumanMessage(content="User message"),
            ]
            
            chat._generate(messages)
            
            # 验证消息转换正确
            call_args = mock_provider.send_request.call_args
            request = call_args[0][1]
            
            assert len(request.messages) == 2
            assert request.messages[0]["role"] == "system"
            assert request.messages[0]["content"] == "System prompt"
            assert request.messages[1]["role"] == "user"
            assert request.messages[1]["content"] == "User message"

    @pytest.mark.asyncio
    async def test_chat_model_agenerate(self, langchain_provider_config, sample_chat_response):
        """Test RouterChatModel _agenerate method."""
        from llm_api_router.integrations.langchain import RouterChatModel
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request_async = AsyncMock(return_value=sample_chat_response)
            mock_factory.return_value = mock_provider
            
            chat = RouterChatModel(config=langchain_provider_config)
            
            messages = [HumanMessage(content="Hello")]
            result = await chat._agenerate(messages)
            
            assert isinstance(result, ChatResult)
            assert len(result.generations) == 1

    def test_chat_model_with_tool_calls_response(self, langchain_provider_config):
        """Test ChatModel handling tool calls in response."""
        from llm_api_router.integrations.langchain import RouterChatModel
        from llm_api_router.types import ToolCall, FunctionCall
        
        # 创建带有 tool_calls 的响应
        response_with_tools = UnifiedResponse(
            id="chatcmpl-test123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_123",
                                type="function",
                                function=FunctionCall(
                                    name="get_weather",
                                    arguments='{"city": "Beijing"}'
                                )
                            )
                        ]
                    ),
                    finish_reason="tool_calls"
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18
            )
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.send_request = Mock(return_value=response_with_tools)
            mock_factory.return_value = mock_provider
            
            chat = RouterChatModel(config=langchain_provider_config)
            
            messages = [HumanMessage(content="What's the weather in Beijing?")]
            result = chat._generate(messages)
            
            assert isinstance(result, ChatResult)
            ai_message = result.generations[0].message
            
            # 验证 tool calls 被正确解析
            assert len(ai_message.tool_calls) == 1
            assert ai_message.tool_calls[0]["name"] == "get_weather"
            assert ai_message.tool_calls[0]["args"]["city"] == "Beijing"


class TestRouterEmbeddings:
    """Test RouterEmbeddings class."""

    def test_embeddings_initialization(self, langchain_provider_config):
        """Test RouterEmbeddings initialization."""
        from llm_api_router.integrations.langchain import RouterEmbeddings
        
        embeddings = RouterEmbeddings(config=langchain_provider_config)
        
        assert embeddings.config == langchain_provider_config
        assert embeddings.model is None

    def test_embeddings_with_model(self, langchain_provider_config):
        """Test RouterEmbeddings with custom model."""
        from llm_api_router.integrations.langchain import RouterEmbeddings
        
        embeddings = RouterEmbeddings(
            config=langchain_provider_config,
            model="text-embedding-3-large",
            dimensions=1024,
        )
        
        assert embeddings.model == "text-embedding-3-large"
        assert embeddings.dimensions == 1024

    def test_embed_documents(self, langchain_provider_config, sample_embedding_response):
        """Test RouterEmbeddings embed_documents method."""
        from llm_api_router.integrations.langchain import RouterEmbeddings
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings = Mock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            embeddings = RouterEmbeddings(config=langchain_provider_config)
            
            texts = ["Hello", "World"]
            vectors = embeddings.embed_documents(texts)
            
            assert len(vectors) == 2
            assert vectors[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert vectors[1] == [0.5, 0.4, 0.3, 0.2, 0.1]

    def test_embed_query(self, langchain_provider_config, sample_embedding_response):
        """Test RouterEmbeddings embed_query method."""
        from llm_api_router.integrations.langchain import RouterEmbeddings
        
        # 单个嵌入的响应
        single_embedding_response = EmbeddingResponse(
            data=[Embedding(index=0, embedding=[0.1, 0.2, 0.3, 0.4, 0.5])],
            model="text-embedding-3-small",
            usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings = Mock(return_value=single_embedding_response)
            mock_factory.return_value = mock_provider
            
            embeddings = RouterEmbeddings(config=langchain_provider_config)
            
            vector = embeddings.embed_query("Hello")
            
            assert vector == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_embed_empty_documents(self, langchain_provider_config):
        """Test RouterEmbeddings embed_documents with empty list."""
        from llm_api_router.integrations.langchain import RouterEmbeddings
        
        embeddings = RouterEmbeddings(config=langchain_provider_config)
        
        vectors = embeddings.embed_documents([])
        
        assert vectors == []

    @pytest.mark.asyncio
    async def test_aembed_documents(self, langchain_provider_config, sample_embedding_response):
        """Test RouterEmbeddings aembed_documents method."""
        from llm_api_router.integrations.langchain import RouterEmbeddings
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings_async = AsyncMock(return_value=sample_embedding_response)
            mock_factory.return_value = mock_provider
            
            embeddings = RouterEmbeddings(config=langchain_provider_config)
            
            texts = ["Hello", "World"]
            vectors = await embeddings.aembed_documents(texts)
            
            assert len(vectors) == 2
            assert vectors[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_aembed_query(self, langchain_provider_config):
        """Test RouterEmbeddings aembed_query method."""
        from llm_api_router.integrations.langchain import RouterEmbeddings
        
        single_embedding_response = EmbeddingResponse(
            data=[Embedding(index=0, embedding=[0.1, 0.2, 0.3])],
            model="text-embedding-3-small",
            usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
        )
        
        with patch('llm_api_router.client.ProviderFactory.get_provider') as mock_factory:
            mock_provider = Mock()
            mock_provider.create_embeddings_async = AsyncMock(return_value=single_embedding_response)
            mock_factory.return_value = mock_provider
            
            embeddings = RouterEmbeddings(config=langchain_provider_config)
            
            vector = await embeddings.aembed_query("Hello")
            
            assert vector == [0.1, 0.2, 0.3]


class TestMessageConversion:
    """Test message conversion functions."""

    def test_convert_human_message(self):
        """Test converting HumanMessage to dict."""
        from llm_api_router.integrations.langchain.chat_model import _convert_message_to_dict
        
        msg = HumanMessage(content="Hello")
        result = _convert_message_to_dict(msg)
        
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_convert_ai_message(self):
        """Test converting AIMessage to dict."""
        from llm_api_router.integrations.langchain.chat_model import _convert_message_to_dict
        
        msg = AIMessage(content="Hi there!")
        result = _convert_message_to_dict(msg)
        
        assert result["role"] == "assistant"
        assert result["content"] == "Hi there!"

    def test_convert_system_message(self):
        """Test converting SystemMessage to dict."""
        from llm_api_router.integrations.langchain.chat_model import _convert_message_to_dict
        
        msg = SystemMessage(content="You are a helpful assistant.")
        result = _convert_message_to_dict(msg)
        
        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_convert_tool_message(self):
        """Test converting ToolMessage to dict."""
        from llm_api_router.integrations.langchain.chat_model import _convert_message_to_dict
        
        msg = ToolMessage(content='{"result": "sunny"}', tool_call_id="call_123")
        result = _convert_message_to_dict(msg)
        
        assert result["role"] == "tool"
        assert result["content"] == '{"result": "sunny"}'
        assert result["tool_call_id"] == "call_123"

    def test_convert_ai_message_with_tool_calls(self):
        """Test converting AIMessage with tool_calls to dict."""
        from llm_api_router.integrations.langchain.chat_model import _convert_message_to_dict
        
        msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_123",
                "name": "get_weather",
                "args": {"city": "Beijing"}
            }]
        )
        result = _convert_message_to_dict(msg)
        
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_convert_dict_to_ai_message(self):
        """Test converting dict to AIMessage."""
        from llm_api_router.integrations.langchain.chat_model import _convert_dict_to_message
        
        msg_dict = {"role": "assistant", "content": "Hello!"}
        result = _convert_dict_to_message(msg_dict)
        
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

    def test_convert_dict_to_ai_message_with_tool_calls(self):
        """Test converting dict with tool_calls to AIMessage."""
        from llm_api_router.integrations.langchain.chat_model import _convert_dict_to_message
        
        msg_dict = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Beijing"}'
                }
            }]
        }
        result = _convert_dict_to_message(msg_dict)
        
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"]["city"] == "Beijing"


class TestToolConversion:
    """Test tool conversion functions."""

    def test_convert_dict_tools(self):
        """Test converting dict tools to router format."""
        from llm_api_router.integrations.langchain.chat_model import _convert_tools_to_router_format
        
        tools = [{
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }
        }]
        
        result = _convert_tools_to_router_format(tools)
        
        assert result is not None
        assert len(result) == 1
        assert result[0].type == "function"
        assert result[0].function.name == "get_weather"

    def test_convert_none_tools(self):
        """Test converting None tools."""
        from llm_api_router.integrations.langchain.chat_model import _convert_tools_to_router_format
        
        result = _convert_tools_to_router_format(None)
        
        assert result is None

    def test_convert_empty_tools(self):
        """Test converting empty tools list."""
        from llm_api_router.integrations.langchain.chat_model import _convert_tools_to_router_format
        
        result = _convert_tools_to_router_format([])
        
        assert result is None
