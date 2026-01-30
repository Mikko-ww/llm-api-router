"""LangChain Chat Model 接口实现

实现 LangChain 的 BaseChatModel 接口，允许 LLM API Router 作为 LangChain 的聊天模型使用。
支持:
- 标准聊天功能
- 流式输出
- 异步调用
- Function Calling / Tool Use
"""

from typing import Any, Dict, List, Optional, Iterator, AsyncIterator, Union

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
        ToolMessage,
        AIMessageChunk,
    )
    from langchain_core.callbacks.manager import (
        CallbackManagerForLLMRun,
        AsyncCallbackManagerForLLMRun,
    )
    from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
    from langchain_core.tools import BaseTool
    from pydantic import PrivateAttr, model_validator
except ImportError as e:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install it with: pip install langchain-core"
    ) from e

from ...types import ProviderConfig, Tool, FunctionDefinition, ToolCall, FunctionCall
from ...client import Client, AsyncClient


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """将 LangChain 消息转换为 LLM API Router 消息格式
    
    Args:
        message: LangChain 消息对象
        
    Returns:
        消息字典
    """
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": message.content}
        # 处理 tool_calls
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                import json
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {})) if isinstance(tc.get("args"), dict) else tc.get("args", "")
                    }
                })
            msg_dict["tool_calls"] = tool_calls
        return msg_dict
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id
        }
    else:
        # 对于其他消息类型，尝试作为用户消息处理
        return {"role": "user", "content": str(message.content)}


def _convert_dict_to_message(msg_dict: Dict[str, Any]) -> BaseMessage:
    """将响应消息字典转换为 LangChain 消息
    
    Args:
        msg_dict: 消息字典
        
    Returns:
        LangChain 消息对象
    """
    role = msg_dict.get("role", "assistant")
    content = msg_dict.get("content", "")
    
    if role == "assistant":
        # 处理 tool_calls
        tool_calls = msg_dict.get("tool_calls")
        if tool_calls:
            import json
            lc_tool_calls = []
            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                lc_tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "args": args
                })
            return AIMessage(content=content or "", tool_calls=lc_tool_calls)
        return AIMessage(content=content or "")
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return AIMessage(content=content or "")


def _convert_tools_to_router_format(tools: Optional[List[Any]]) -> Optional[List[Tool]]:
    """将 LangChain tools 转换为 LLM API Router 格式
    
    Args:
        tools: LangChain tools 列表
        
    Returns:
        LLM API Router Tool 列表
    """
    if not tools:
        return None
    
    converted_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            # 已经是字典格式
            if "function" in tool:
                func = tool["function"]
                converted_tools.append(Tool(
                    type="function",
                    function=FunctionDefinition(
                        name=func.get("name", ""),
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    )
                ))
            else:
                converted_tools.append(Tool(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.get("name", ""),
                        description=tool.get("description", ""),
                        parameters=tool.get("parameters", {}),
                    )
                ))
        elif hasattr(tool, "name") and hasattr(tool, "description"):
            # LangChain BaseTool 实例
            schema = {}
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.schema()
            elif hasattr(tool, "args"):
                schema = tool.args
            
            converted_tools.append(Tool(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=schema,
                )
            ))
    
    return converted_tools if converted_tools else None


class RouterChatModel(BaseChatModel):
    """LangChain Chat Model 接口实现
    
    将 LLM API Router 作为 LangChain 的聊天模型使用，支持:
    - 多轮对话
    - 流式输出
    - 异步调用
    - Function Calling / Tool Use
    
    Example:
        ```python
        from llm_api_router.integrations.langchain import RouterChatModel
        from llm_api_router import ProviderConfig
        from langchain_core.messages import HumanMessage, SystemMessage
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="your-api-key",
            default_model="gpt-4o-mini"
        )
        
        chat = RouterChatModel(config=config)
        
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?"),
        ]
        
        result = chat.invoke(messages)
        print(result.content)
        ```
    
    Tool Use Example:
        ```python
        from langchain_core.tools import tool
        
        @tool
        def get_weather(city: str) -> str:
            '''获取城市天气'''
            return f"{city}的天气是晴朗的"
        
        chat_with_tools = chat.bind_tools([get_weather])
        result = chat_with_tools.invoke([HumanMessage(content="北京天气如何?")])
        ```
    """
    
    # 使用 Any 类型存储 config，避免 Pydantic 类型解析问题
    router_config: Any = None
    """LLM API Router 配置（ProviderConfig 实例）"""
    
    model_name: Optional[str] = None
    """模型名称，覆盖 config 中的默认模型"""
    
    temperature: float = 1.0
    """生成温度"""
    
    max_tokens: Optional[int] = None
    """最大生成 token 数"""
    
    top_p: Optional[float] = None
    """Top-p 采样参数"""
    
    stop_sequences: Optional[List[str]] = None
    """停止序列"""
    
    streaming: bool = False
    """是否使用流式输出"""
    
    # 使用 PrivateAttr 存储客户端实例
    _client: Optional[Client] = PrivateAttr(default=None)
    
    class Config:
        """Pydantic 配置"""
        arbitrary_types_allowed = True
    
    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """验证并处理 config 参数"""
        # 支持使用 'config' 作为参数名（兼容性）
        if 'config' in values and 'router_config' not in values:
            values['router_config'] = values.pop('config')
        
        # 验证 router_config 是 ProviderConfig 实例
        config = values.get('router_config')
        if config is not None and not isinstance(config, ProviderConfig):
            raise ValueError("router_config must be a ProviderConfig instance")
        
        return values
    
    def __init__(self, **kwargs):
        # 处理 'config' 参数
        if 'config' in kwargs and 'router_config' not in kwargs:
            kwargs['router_config'] = kwargs.pop('config')
        super().__init__(**kwargs)
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        if self.router_config:
            return f"llm-api-router-chat-{self.router_config.provider_type}"
        return "llm-api-router-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别此模型配置的参数"""
        params = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
        }
        if self.router_config:
            params["provider_type"] = self.router_config.provider_type
            params["model_name"] = self.model_name or self.router_config.default_model
        return params
    
    def _get_client(self) -> Client:
        """获取或创建客户端实例"""
        if self._client is None:
            if self.router_config is None:
                raise ValueError("router_config is required")
            self._client = Client(self.router_config)
        return self._client
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """执行聊天生成
        
        Args:
            messages: LangChain 消息列表
            stop: 停止序列
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult 对象
        """
        client = self._get_client()
        
        # 转换消息格式
        converted_messages = [_convert_message_to_dict(m) for m in messages]
        
        # 合并参数
        effective_stop = stop or self.stop_sequences
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        model = kwargs.get("model", self.model_name)
        
        # 处理 tools
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        
        # 构建请求参数
        request_kwargs: Dict[str, Any] = {
            "messages": converted_messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": effective_stop,
            "stream": False,
        }
        
        # 如果有 tools，需要直接调用 provider
        if tools:
            from ...types import UnifiedRequest
            from ...logging_config import generate_request_id
            
            converted_tools = _convert_tools_to_router_format(tools)
            request = UnifiedRequest(
                messages=converted_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=effective_stop,
                stream=False,
                tools=converted_tools,
                tool_choice=tool_choice,
                request_id=generate_request_id()
            )
            response = client._provider.send_request(client._http_client, request)
        else:
            response = client.chat.completions.create(**request_kwargs)
        
        # 转换响应
        msg = response.choices[0].message
        msg_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        
        # 添加 tool_calls 信息
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        
        ai_message = _convert_dict_to_message(msg_dict)
        
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                "finish_reason": response.choices[0].finish_reason,
                "model": response.model,
            },
        )
        
        return ChatResult(
            generations=[generation],
            llm_output={
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model_name": response.model,
            },
        )
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式执行聊天生成
        
        Args:
            messages: LangChain 消息列表
            stop: 停止序列
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Yields:
            ChatGenerationChunk 对象
        """
        client = self._get_client()
        
        # 转换消息格式
        converted_messages = [_convert_message_to_dict(m) for m in messages]
        
        # 合并参数
        effective_stop = stop or self.stop_sequences
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        model = kwargs.get("model", self.model_name)
        
        stream = client.chat.completions.create(
            messages=converted_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=effective_stop,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                
                if content:
                    if run_manager:
                        run_manager.on_llm_new_token(content)
                    
                    chunk_msg = AIMessageChunk(content=content)
                    yield ChatGenerationChunk(
                        message=chunk_msg,
                        generation_info={
                            "finish_reason": chunk.choices[0].finish_reason,
                        } if chunk.choices[0].finish_reason else None,
                    )
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步执行聊天生成
        
        Args:
            messages: LangChain 消息列表
            stop: 停止序列
            run_manager: 异步回调管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult 对象
        """
        if self.router_config is None:
            raise ValueError("router_config is required")
            
        async with AsyncClient(self.router_config) as client:
            # 转换消息格式
            converted_messages = [_convert_message_to_dict(m) for m in messages]
            
            # 合并参数
            effective_stop = stop or self.stop_sequences
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            top_p = kwargs.get("top_p", self.top_p)
            model = kwargs.get("model", self.model_name)
            
            # 处理 tools
            tools = kwargs.get("tools")
            tool_choice = kwargs.get("tool_choice")
            
            # 构建请求参数
            request_kwargs: Dict[str, Any] = {
                "messages": converted_messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stop": effective_stop,
                "stream": False,
            }
            
            # 如果有 tools，需要直接调用 provider
            if tools:
                from ...types import UnifiedRequest
                from ...logging_config import generate_request_id
                
                converted_tools = _convert_tools_to_router_format(tools)
                request = UnifiedRequest(
                    messages=converted_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=effective_stop,
                    stream=False,
                    tools=converted_tools,
                    tool_choice=tool_choice,
                    request_id=generate_request_id()
                )
                response = await client._provider.send_request_async(client._http_client, request)
            else:
                response = await client.chat.completions.create(**request_kwargs)
            
            # 转换响应
            msg = response.choices[0].message
            msg_dict: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content,
            }
            
            # 添加 tool_calls 信息
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            ai_message = _convert_dict_to_message(msg_dict)
            
            generation = ChatGeneration(
                message=ai_message,
                generation_info={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                },
            )
            
            return ChatResult(
                generations=[generation],
                llm_output={
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "model_name": response.model,
                },
            )
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """异步流式执行聊天生成
        
        Args:
            messages: LangChain 消息列表
            stop: 停止序列
            run_manager: 异步回调管理器
            **kwargs: 其他参数
            
        Yields:
            ChatGenerationChunk 对象
        """
        if self.router_config is None:
            raise ValueError("router_config is required")
            
        async with AsyncClient(self.router_config) as client:
            # 转换消息格式
            converted_messages = [_convert_message_to_dict(m) for m in messages]
            
            # 合并参数
            effective_stop = stop or self.stop_sequences
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            top_p = kwargs.get("top_p", self.top_p)
            model = kwargs.get("model", self.model_name)
            
            stream = await client.chat.completions.create(
                messages=converted_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=effective_stop,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    
                    if content:
                        if run_manager:
                            await run_manager.on_llm_new_token(content)
                        
                        chunk_msg = AIMessageChunk(content=content)
                        yield ChatGenerationChunk(
                            message=chunk_msg,
                            generation_info={
                                "finish_reason": chunk.choices[0].finish_reason,
                            } if chunk.choices[0].finish_reason else None,
                        )
    
    def bind_tools(
        self,
        tools: List[Any],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "RouterChatModel":
        """绑定工具到聊天模型
        
        Args:
            tools: 工具列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数
            
        Returns:
            绑定了工具的新 ChatModel 实例
        """
        # 创建新实例，继承当前配置
        new_kwargs = {
            "config": self.router_config,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences,
            "streaming": self.streaming,
        }
        new_kwargs.update(kwargs)
        
        new_model = RouterChatModel(**new_kwargs)
        
        # 绑定工具参数，使用 LangChain 的 bind 方法
        return new_model.bind(tools=tools, tool_choice=tool_choice)
