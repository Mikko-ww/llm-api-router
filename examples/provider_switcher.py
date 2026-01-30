#!/usr/bin/env python3
"""
动态提供商切换示例

展示如何在运行时切换不同的 LLM 提供商，实现故障转移或成本优化。
"""

import os
from typing import Optional
from llm_api_router import Client, ProviderConfig
from llm_api_router.exceptions import LLMRouterError


class ProviderSwitcher:
    """提供商切换器，支持动态切换和故障转移"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider: Optional[str] = None
        self._setup_providers()
    
    def _setup_providers(self):
        """设置可用的提供商配置"""
        # OpenAI - 主要提供商
        if os.environ.get("OPENAI_API_KEY"):
            self.providers["openai"] = ProviderConfig(
                provider_type="openai",
                api_key=os.environ["OPENAI_API_KEY"],
                default_model="gpt-4o",
                timeout=30.0,
            )
        
        # Anthropic - 备用提供商
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.providers["anthropic"] = ProviderConfig(
                provider_type="anthropic",
                api_key=os.environ["ANTHROPIC_API_KEY"],
                default_model="claude-3-5-sonnet-20241022",
                timeout=30.0,
            )
        
        # DeepSeek - 经济实惠选项
        if os.environ.get("DEEPSEEK_API_KEY"):
            self.providers["deepseek"] = ProviderConfig(
                provider_type="deepseek",
                api_key=os.environ["DEEPSEEK_API_KEY"],
                default_model="deepseek-chat",
                timeout=30.0,
            )
        
        # Ollama - 本地免费选项
        self.providers["ollama"] = ProviderConfig(
            provider_type="ollama",
            api_key="not-required",
            base_url="http://localhost:11434",
            default_model="llama3.2",
            timeout=60.0,
        )
        
        # 设置默认提供商
        if self.providers:
            self.current_provider = list(self.providers.keys())[0]
    
    def list_providers(self) -> list[str]:
        """列出所有可用的提供商"""
        return list(self.providers.keys())
    
    def switch_to(self, provider_name: str) -> bool:
        """切换到指定提供商"""
        if provider_name in self.providers:
            self.current_provider = provider_name
            print(f"✓ 已切换到提供商: {provider_name}")
            return True
        print(f"✗ 未知提供商: {provider_name}")
        return False
    
    def get_current_config(self) -> Optional[ProviderConfig]:
        """获取当前提供商配置"""
        if self.current_provider:
            return self.providers.get(self.current_provider)
        return None
    
    def chat(self, message: str, fallback: bool = True) -> str:
        """
        发送聊天消息，支持自动故障转移
        
        Args:
            message: 用户消息
            fallback: 是否在失败时自动切换到其他提供商
        
        Returns:
            助手响应内容
        """
        providers_to_try = [self.current_provider] if not fallback else list(self.providers.keys())
        last_error = None
        
        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue
                
            config = self.providers[provider_name]
            try:
                with Client(config) as client:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": message}]
                    )
                    
                    # 更新当前提供商（故障转移后）
                    if provider_name != self.current_provider:
                        print(f"⚠ 故障转移: {self.current_provider} -> {provider_name}")
                        self.current_provider = provider_name
                    
                    return response.choices[0].message.content
                    
            except LLMRouterError as e:
                last_error = e
                print(f"✗ {provider_name} 失败: {e}")
                continue
        
        raise last_error or RuntimeError("所有提供商都不可用")


def main():
    """演示提供商切换功能"""
    switcher = ProviderSwitcher()
    
    print("=" * 50)
    print("LLM 提供商切换器示例")
    print("=" * 50)
    
    # 显示可用提供商
    print(f"\n可用提供商: {', '.join(switcher.list_providers())}")
    print(f"当前提供商: {switcher.current_provider}")
    
    # 示例 1: 使用当前提供商
    print("\n--- 示例 1: 基本使用 ---")
    try:
        response = switcher.chat("用一句话介绍你自己")
        print(f"响应: {response}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 示例 2: 手动切换提供商
    print("\n--- 示例 2: 手动切换 ---")
    if "anthropic" in switcher.list_providers():
        switcher.switch_to("anthropic")
        try:
            response = switcher.chat("用一句话介绍你自己")
            print(f"响应: {response}")
        except Exception as e:
            print(f"错误: {e}")
    
    # 示例 3: 自动故障转移
    print("\n--- 示例 3: 自动故障转移 ---")
    # 切换到一个可能失败的提供商
    switcher.switch_to("ollama")
    try:
        response = switcher.chat("用一句话介绍你自己", fallback=True)
        print(f"响应: {response}")
    except Exception as e:
        print(f"所有提供商都失败: {e}")
    
    # 示例 4: 按成本选择提供商
    print("\n--- 示例 4: 成本优化 ---")
    cost_priority = ["ollama", "deepseek", "openai", "anthropic"]
    for provider in cost_priority:
        if provider in switcher.list_providers():
            switcher.switch_to(provider)
            break
    
    try:
        response = switcher.chat("什么是人工智能？简短回答。")
        print(f"使用 {switcher.current_provider}: {response}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
