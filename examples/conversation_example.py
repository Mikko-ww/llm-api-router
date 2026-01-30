"""
Conversation Management Example

This example demonstrates various ways to use the conversation manager
in LLM API Router.
"""

from llm_api_router.conversation import (
    ConversationConfig,
    ConversationManager,
    TokenCounter,
    SlidingWindowStrategy,
    KeepRecentStrategy,
    ImportanceBasedStrategy,
    create_conversation,
)


def basic_conversation():
    """Basic conversation management"""
    print("=== Basic Conversation ===\n")
    
    # Create a conversation manager
    manager = create_conversation(
        max_tokens=4096,
        system_prompt="You are a helpful assistant."
    )
    
    # Add messages
    manager.add_user_message("Hello!")
    manager.add_assistant_message("Hi there! How can I help you today?")
    manager.add_user_message("What's the weather like?")
    manager.add_assistant_message("I don't have access to real-time weather data.")
    
    # Get messages for API call
    messages = manager.get_messages_for_api()
    print(f"Messages for API ({len(messages)} total):")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content'][:50]}...")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\nStats:")
    print(f"  Token count: {stats['token_count']}")
    print(f"  Message count: {stats['message_count']}")
    print()


def token_counting():
    """Token counting demonstration"""
    print("=== Token Counting ===\n")
    
    counter = TokenCounter(model="gpt-3.5-turbo")
    
    # Count tokens in text
    text1 = "Hello, how are you today?"
    text2 = "你好，今天天气怎么样？"  # Chinese text
    
    print(f"English: '{text1}'")
    print(f"  Tokens: {counter.count_tokens(text1)}")
    
    print(f"Chinese: '{text2}'")
    print(f"  Tokens: {counter.count_tokens(text2)}")
    
    # Count message tokens
    message = {"role": "user", "content": "Can you help me write some code?"}
    print(f"\nMessage: {message}")
    print(f"  Tokens (including overhead): {counter.count_message_tokens(message)}")
    print()


def truncation_strategies():
    """Different truncation strategies"""
    print("=== Truncation Strategies ===\n")
    
    # Create sample messages
    def create_messages(count):
        messages = [{"role": "system", "content": "System prompt"}]
        for i in range(count):
            messages.append({"role": "user", "content": f"User message {i+1}"})
            messages.append({"role": "assistant", "content": f"Assistant response {i+1}"})
        return messages
    
    messages = create_messages(5)
    counter = TokenCounter()
    max_tokens = 150
    
    # Sliding Window Strategy
    sliding = SlidingWindowStrategy()
    result = sliding.truncate(messages, max_tokens, counter)
    print(f"Sliding Window (max {max_tokens} tokens):")
    print(f"  Original: {len(messages)} messages")
    print(f"  After truncation: {len(result)} messages")
    if result:
        print(f"  Last message: {result[-1]['content']}")
    
    # Keep Recent Strategy
    keep_recent = KeepRecentStrategy(keep_count=4)
    result = keep_recent.truncate(messages, max_tokens, counter)
    print(f"\nKeep Recent (keep 4):")
    print(f"  After truncation: {len(result)} messages")
    
    print()


def important_messages():
    """Working with important messages"""
    print("=== Important Messages ===\n")
    
    config = ConversationConfig(
        max_tokens=200,
        strategy="importance"
    )
    manager = ConversationManager(config, strategy=ImportanceBasedStrategy())
    
    manager.set_system_prompt("System prompt")
    
    # Add regular and important messages
    manager.add_user_message("Regular question 1")
    manager.add_assistant_message("Regular answer 1")
    manager.add_user_message("IMPORTANT: Remember my name is Alice", important=True)
    manager.add_assistant_message("I'll remember that, Alice!", important=True)
    manager.add_user_message("Regular question 2")
    manager.add_assistant_message("Regular answer 2")
    manager.add_user_message("What's my name?")
    
    messages = manager.get_messages_for_api()
    print(f"Messages after truncation ({len(messages)} remaining):")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content'][:40]}...")
    
    print()


def conversation_forking():
    """Forking conversations"""
    print("=== Conversation Forking ===\n")
    
    # Create initial conversation
    manager = create_conversation(system_prompt="You are a coding assistant.")
    manager.add_user_message("Help me with Python")
    manager.add_assistant_message("Sure! What would you like to know about Python?")
    
    # Fork for different paths
    fork1 = manager.fork()
    fork2 = manager.fork()
    
    # Continue different conversations
    fork1.add_user_message("Tell me about list comprehensions")
    fork2.add_user_message("Tell me about decorators")
    
    print(f"Original: {manager.message_count} messages")
    print(f"Fork 1: {fork1.message_count} messages")
    print(f"  Last: {fork1.messages[-1]['content']}")
    print(f"Fork 2: {fork2.message_count} messages")
    print(f"  Last: {fork2.messages[-1]['content']}")
    print()


def auto_truncation():
    """Automatic truncation demonstration"""
    print("=== Auto Truncation ===\n")
    
    config = ConversationConfig(
        max_tokens=300,
        max_messages=10
    )
    manager = ConversationManager(config)
    
    manager.set_system_prompt("You are a helpful assistant.")
    
    # Add many messages
    for i in range(20):
        manager.add_user_message(f"Question number {i+1}?")
        manager.add_assistant_message(f"This is answer {i+1}.")
    
    stats = manager.get_stats()
    print(f"After adding 40 messages:")
    print(f"  Current message count: {stats['message_count']}")
    print(f"  Current token count: {stats['token_count']}")
    print(f"  Max tokens: {stats['max_tokens']}")
    print(f"  Max messages: {stats['max_messages']}")
    
    # Check what's remaining
    messages = manager.messages
    non_system = [m for m in messages if m['role'] != 'system']
    if non_system:
        print(f"  Oldest non-system message: {non_system[0]['content']}")
        print(f"  Newest message: {non_system[-1]['content']}")
    print()


def with_client_integration():
    """Example of integrating with LLM client (pseudo-code)"""
    print("=== Client Integration Example ===\n")
    
    # Create conversation
    conversation = create_conversation(
        max_tokens=4096,
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant."
    )
    
    # Simulate a conversation loop
    user_inputs = [
        "Hello!",
        "What's 2+2?",
        "Thanks!"
    ]
    
    print("Simulated conversation:")
    for user_input in user_inputs:
        # Add user message
        conversation.add_user_message(user_input)
        print(f"User: {user_input}")
        
        # In real code, you would call the LLM API here:
        # messages = conversation.get_messages_for_api()
        # response = client.chat.completions.create(messages=messages)
        # assistant_response = response.choices[0].message.content
        
        # Simulate response
        assistant_response = f"[Simulated response to: {user_input}]"
        
        # Add assistant response
        conversation.add_assistant_message(assistant_response)
        print(f"Assistant: {assistant_response}")
    
    print(f"\nFinal stats: {conversation.get_stats()}")
    print()


def main():
    """Run all examples"""
    basic_conversation()
    token_counting()
    truncation_strategies()
    important_messages()
    conversation_forking()
    auto_truncation()
    with_client_integration()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()
