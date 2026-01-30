"""LangChain LLM 接口实现

实现 LangChain 的 BaseLLM 接口，允许 LLM API Router 作为 LangChain 的语言模型使用。
"""

from typing import Any, Dict, List, Optional, Iterator, AsyncIterator

try:
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
    from langchain_core.outputs import GenerationChunk
    from pydantic import PrivateAttr, model_validator
except ImportError as e:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install it with: pip install langchain-core"
    ) from e

from ...types import ProviderConfig
from ...client import Client, AsyncClient


class RouterLLM(LLM):
    """LangChain LLM 接口实现
    
    将 LLM API Router 作为 LangChain 的语言模型使用。
    
    Example:
        ```python
        from llm_api_router.integrations.langchain import RouterLLM
        from llm_api_router import ProviderConfig
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="your-api-key",
            default_model="gpt-4o-mini"
        )
        
        llm = RouterLLM(config=config)
        result = llm.invoke("What is the capital of France?")
        print(result)
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
            return f"llm-api-router-{self.router_config.provider_type}"
        return "llm-api-router"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别此 LLM 配置的参数"""
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
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行 LLM 调用
        
        Args:
            prompt: 输入提示
            stop: 停止序列
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        client = self._get_client()
        
        # 合并停止序列
        effective_stop = stop or self.stop_sequences
        
        # 构建消息（LLM 接口使用单个用户消息）
        messages = [{"role": "user", "content": prompt}]
        
        # 合并参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        model = kwargs.get("model", self.model_name)
        
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=effective_stop,
            stream=False,
        )
        
        return response.choices[0].message.content or ""
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """流式执行 LLM 调用
        
        Args:
            prompt: 输入提示
            stop: 停止序列
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Yields:
            生成的文本块
        """
        client = self._get_client()
        
        # 合并停止序列
        effective_stop = stop or self.stop_sequences
        
        # 构建消息
        messages = [{"role": "user", "content": prompt}]
        
        # 合并参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        model = kwargs.get("model", self.model_name)
        
        stream = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=effective_stop,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if run_manager:
                    run_manager.on_llm_new_token(content)
                yield GenerationChunk(text=content)
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """异步执行 LLM 调用
        
        Args:
            prompt: 输入提示
            stop: 停止序列
            run_manager: 异步回调管理器
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        if self.router_config is None:
            raise ValueError("router_config is required")
            
        async with AsyncClient(self.router_config) as client:
            # 合并停止序列
            effective_stop = stop or self.stop_sequences
            
            # 构建消息
            messages = [{"role": "user", "content": prompt}]
            
            # 合并参数
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            top_p = kwargs.get("top_p", self.top_p)
            model = kwargs.get("model", self.model_name)
            
            response = await client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=effective_stop,
                stream=False,
            )
            
            return response.choices[0].message.content or ""
    
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """异步流式执行 LLM 调用
        
        Args:
            prompt: 输入提示
            stop: 停止序列
            run_manager: 异步回调管理器
            **kwargs: 其他参数
            
        Yields:
            生成的文本块
        """
        if self.router_config is None:
            raise ValueError("router_config is required")
            
        async with AsyncClient(self.router_config) as client:
            # 合并停止序列
            effective_stop = stop or self.stop_sequences
            
            # 构建消息
            messages = [{"role": "user", "content": prompt}]
            
            # 合并参数
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            top_p = kwargs.get("top_p", self.top_p)
            model = kwargs.get("model", self.model_name)
            
            stream = await client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=effective_stop,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if run_manager:
                        await run_manager.on_llm_new_token(content)
                    yield GenerationChunk(text=content)
