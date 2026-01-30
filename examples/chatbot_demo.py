#!/usr/bin/env python3
"""
聊天机器人示例

完整的命令行聊天机器人，支持多轮对话、流式输出和会话管理。
"""

import os
import sys
from typing import Optional
from llm_api_router import Client, ProviderConfig, ConversationManager
from llm_api_router.exceptions import LLMRouterError


class ChatBot:
    """交互式聊天机器人"""
    
    def __init__(
        self,
        provider_type: str = "openai",
        model: Optional[str] = None,
        system_message: Optional[str] = None,
    ):
        self.config = self._create_config(provider_type, model)
        self.conversation = ConversationManager(
            max_history=50,
            system_message=system_message or "你是一个友好、有帮助的AI助手。请用中文回答问题。",
        )
        self.client: Optional[Client] = None
    
    def _create_config(self, provider_type: str, model: Optional[str]) -> ProviderConfig:
        """根据提供商类型创建配置"""
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
        }
        
        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-1.5-flash",
            "deepseek": "deepseek-chat",
            "zhipu": "glm-4",
            "ollama": "llama3.2",
        }
        
        if provider_type == "ollama":
            return ProviderConfig(
                provider_type="ollama",
                api_key="",
                base_url="http://localhost:11434",
                default_model=model or default_models.get(provider_type),
            )
        
        api_key_env = api_key_map.get(provider_type)
        if not api_key_env:
            raise ValueError(f"不支持的提供商: {provider_type}")
        
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"请设置环境变量 {api_key_env}")
        
        return ProviderConfig(
            provider_type=provider_type,
            api_key=api_key,
            default_model=model or default_models.get(provider_type),
        )
    
    def start(self):
        """启动客户端"""
        self.client = Client(self.config)
    
    def stop(self):
        """停止客户端"""
        if self.client:
            self.client.close()
            self.client = None
    
    def chat(self, user_input: str, stream: bool = True) -> str:
        """
        发送消息并获取响应
        
        Args:
            user_input: 用户输入
            stream: 是否使用流式输出
        
        Returns:
            助手响应
        """
        if not self.client:
            raise RuntimeError("请先调用 start() 启动客户端")
        
        # 添加用户消息
        self.conversation.add_user_message(user_input)
        
        if stream:
            return self._chat_stream()
        else:
            return self._chat_normal()
    
    def _chat_normal(self) -> str:
        """非流式聊天"""
        response = self.client.chat.completions.create(
            messages=self.conversation.get_messages(),
        )
        
        content = response.choices[0].message.content
        self.conversation.add_assistant_message(content)
        return content
    
    def _chat_stream(self) -> str:
        """流式聊天"""
        stream = self.client.chat.completions.create(
            messages=self.conversation.get_messages(),
            stream=True,
        )
        
        full_content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content
        
        print()  # 换行
        self.conversation.add_assistant_message(full_content)
        return full_content
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation.clear()
        print("✓ 对话历史已清空")
    
    def show_history(self):
        """显示对话历史"""
        messages = self.conversation.get_messages()
        print("\n--- 对话历史 ---")
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"[{role}] {content}")
        print("--- 历史结束 ---\n")


def print_help():
    """打印帮助信息"""
    print("""
命令:
  /help     显示帮助
  /clear    清空对话历史
  /history  显示对话历史
  /exit     退出程序
  /quit     退出程序
  
直接输入文字开始对话。
""")


def main():
    """主函数"""
    # 从命令行参数获取提供商
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    model = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 50)
    print("🤖 LLM API Router 聊天机器人")
    print("=" * 50)
    print(f"提供商: {provider}")
    print(f"模型: {model or '默认'}")
    print("输入 /help 查看帮助，/exit 退出")
    print("=" * 50)
    
    try:
        bot = ChatBot(provider_type=provider, model=model)
        bot.start()
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)
    
    try:
        while True:
            try:
                user_input = input("\n你: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd in ("/exit", "/quit"):
                    print("再见！👋")
                    break
                elif cmd == "/help":
                    print_help()
                elif cmd == "/clear":
                    bot.clear_history()
                elif cmd == "/history":
                    bot.show_history()
                else:
                    print(f"未知命令: {user_input}")
                continue
            
            # 发送消息
            print("\n助手: ", end="")
            try:
                bot.chat(user_input, stream=True)
            except LLMRouterError as e:
                print(f"\n错误: {e}")
            except KeyboardInterrupt:
                print("\n(已中断)")
                
    except KeyboardInterrupt:
        print("\n\n再见！👋")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
