#!/usr/bin/env python3
"""
Ollama Provider Example Script

This script demonstrates how to use the llm-api-router library with Ollama.

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Pull a model: ollama pull llama3.2
3. Ensure Ollama service is running: ollama serve

Usage:
    python examples/ollama_example.py
"""

import sys
import asyncio
from llm_api_router import Client, AsyncClient, ProviderConfig

# Configure Ollama provider
OLLAMA_CONFIG = ProviderConfig(
    provider_type="ollama",
    api_key="not-required",  # Ollama doesn't require authentication
    base_url="http://localhost:11434",  # Default Ollama server URL
    default_model="llama3.2"  # Change this to any model you have pulled
)


def sync_chat_completion():
    """Basic synchronous chat completion example"""
    print("=== Synchronous Chat Completion ===")
    try:
        with Client(OLLAMA_CONFIG) as client:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Say 'Hello' and introduce yourself in one sentence."}],
                max_tokens=50
            )
            print(f"Response: {response.choices[0].message.content}")
            print(f"Model: {response.model}")
            print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running and you have pulled the model:")
        print("  ollama serve")
        print("  ollama pull llama3.2")


def sync_stream_chat_completion():
    """Synchronous streaming chat completion example"""
    print("\n=== Synchronous Streaming ===")
    try:
        with Client(OLLAMA_CONFIG) as client:
            stream = client.chat.completions.create(
                messages=[{"role": "user", "content": "Write a very short poem about AI (max 4 lines)"}],
                stream=True
            )
            
            print("Response: ", end="", flush=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # New line after stream completes
    except Exception as e:
        print(f"Error: {e}")


async def async_chat_completion():
    """Asynchronous chat completion example"""
    print("\n=== Asynchronous Chat Completion ===")
    try:
        async with AsyncClient(OLLAMA_CONFIG) as client:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
                max_tokens=30
            )
            print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


async def async_stream_chat_completion():
    """Asynchronous streaming chat completion example"""
    print("\n=== Asynchronous Streaming ===")
    try:
        async with AsyncClient(OLLAMA_CONFIG) as client:
            stream = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Count from 1 to 5"}],
                stream=True
            )
            
            print("Response: ", end="", flush=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # New line after stream completes
    except Exception as e:
        print(f"Error: {e}")


def multi_turn_conversation():
    """Example of maintaining conversation context"""
    print("\n=== Multi-turn Conversation ===")
    try:
        with Client(OLLAMA_CONFIG) as client:
            # Conversation history
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
                {"role": "user", "content": "My name is Alice."},
            ]
            
            # First turn
            response = client.chat.completions.create(messages=messages, max_tokens=50)
            assistant_msg = response.choices[0].message.content
            print(f"User: My name is Alice.")
            print(f"Assistant: {assistant_msg}")
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": assistant_msg})
            
            # Second turn
            messages.append({"role": "user", "content": "What's my name?"})
            response = client.chat.completions.create(messages=messages, max_tokens=30)
            print(f"User: What's my name?")
            print(f"Assistant: {response.choices[0].message.content}")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples"""
    print("Ollama Provider Examples")
    print("========================\n")
    
    # Synchronous examples
    sync_chat_completion()
    sync_stream_chat_completion()
    
    # Multi-turn conversation
    multi_turn_conversation()
    
    # Asynchronous examples
    asyncio.run(async_chat_completion())
    asyncio.run(async_stream_chat_completion())
    
    print("\n=== All Examples Completed ===")


if __name__ == "__main__":
    main()
