import importlib
from typing import Dict, Tuple, Type
from .types import ProviderConfig
from .providers.base import BaseProvider

class ProviderFactory:
    # Mapping of provider_type -> (module_path, class_name)
    _PROVIDER_MAPPING: Dict[str, Tuple[str, str]] = {
        "openai": ("llm_api_router.providers.openai", "OpenAIProvider"),
        "openrouter": ("llm_api_router.providers.openrouter", "OpenRouterProvider"),
        "deepseek": ("llm_api_router.providers.deepseek", "DeepSeekProvider"),
        "anthropic": ("llm_api_router.providers.anthropic", "AnthropicProvider"),
        "gemini": ("llm_api_router.providers.gemini", "GeminiProvider"),
        "zhipu": ("llm_api_router.providers.zhipu", "ZhipuProvider"),
        "aliyun": ("llm_api_router.providers.aliyun", "AliyunProvider"),
        "xai": ("llm_api_router.providers.xai", "XAIProvider"),
    }

    @classmethod
    def register(cls, provider_type: str, module_path: str, class_name: str):
        """Register a new provider dynamically"""
        cls._PROVIDER_MAPPING[provider_type] = (module_path, class_name)

    @classmethod
    def get_provider(cls, config: ProviderConfig) -> BaseProvider:
        provider_type = config.provider_type
        
        if provider_type not in cls._PROVIDER_MAPPING:
            raise ValueError(f"不支持的提供商类型: {provider_type}")

        module_path, class_name = cls._PROVIDER_MAPPING[provider_type]
        
        try:
            module = importlib.import_module(module_path)
            provider_cls = getattr(module, class_name)
            return provider_cls(config)
        except ImportError as e:
            raise ValueError(f"无法加载提供商模块 '{provider_type}': {e}")
        except AttributeError as e:
            raise ValueError(f"提供商类 '{class_name}' 未在模块 '{module_path}' 中找到: {e}")
