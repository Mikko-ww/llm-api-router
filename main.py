from dotenv import load_dotenv
from llm_api_router.types import ProviderConfig
from llm_api_router.client import Client, AsyncClient
import os
import asyncio  
import sys

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")

PROVIDER_CONFIG = ProviderConfig(
    provider_type="xai",
    api_key=XAI_API_KEY,
)

#创建一个同步聊天补全对话
def sync_chat_completion():
    try:
        with Client(PROVIDER_CONFIG) as client:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Say 'Hello' only."}],
                model="grok-4-1-fast-reasoning",
                max_tokens=30
            )
            print(f"Response from xai: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")



#创建一个异步聊天补全对话
async def async_chat_completion():
    try:
        async with AsyncClient(PROVIDER_CONFIG) as client:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Say 'Hello' only."}],
                model="grok-4-1-fast-reasoning",
                max_tokens=30
            )
            print(f"Response from xai: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


#创建一个异步补全对话，实现多轮对话，需要用户进行输入，如果用户输入exit 则退出
async def async_multi_turn_chat():
    try:
        async with AsyncClient(PROVIDER_CONFIG) as client:
            # 初始化对话历史以维持上下文记忆
            messages = []
            # 添加系统消息以设定助手行为
            messages.append({"role": "system", "content": "你是一个人工智能助手，名字叫小智。"})

            while True:
                user_input = input("User: ")
                if user_input.lower() == "exit":
                    break
                
                # 将用户输入添加到对话历史
                messages.append({"role": "user", "content": user_input})
                
                response = await client.chat.completions.create(
                    messages=messages,
                    model="grok-4-1-fast-reasoning",
                    max_tokens=150
                )
                
                assistant_content = response.choices[0].message.content
                print(f"Response from xai: {assistant_content}")

                # 打印一下使用情况
                print(f"Usage: {response.usage}")
                
                # 将助手回复添加到对话历史，实现多轮对话记忆
                messages.append({"role": "assistant", "content": assistant_content})
    except Exception as e:
        print(f"Error: {e}")



#创建一个同步流式聊天补全对话
def sync_stream_chat_completion():
    try:
        with Client(PROVIDER_CONFIG) as client:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "给我讲一个100字以内的故事"}],
                model="grok-4-1-fast-reasoning",
                stream=True
            )
            
            # chunks = []
            sys.stdout.write("Response from xai: \n")
            sys.stdout.flush()
            for chunk in response:
                # print(f"Chunk: {chunk}")
                if chunk.choices and chunk.choices[0].delta.content:
                    # chunks.append(chunk.choices[0].delta.content)
                    sys.stdout.write(chunk.choices[0].delta.content)
                    sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
            
            # full_text = "".join(chunks)
            # print(f"Full stream text: {full_text}")
            # assert len(full_text) > 0
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    # sync_chat_completion()
    # asyncio.run(async_chat_completion())
    sync_stream_chat_completion()
    # asyncio.run(async_multi_turn_chat())