"""
Unit tests for conversation module.
"""

import pytest
from llm_api_router.conversation import (
    ConversationConfig,
    TokenCounter,
    TruncationStrategy,
    SlidingWindowStrategy,
    KeepRecentStrategy,
    ImportanceBasedStrategy,
    ConversationManager,
    create_conversation,
)


class TestConversationConfig:
    """Tests for ConversationConfig"""
    
    def test_default_values(self):
        """Test default configuration"""
        config = ConversationConfig()
        
        assert config.max_tokens == 4096
        assert config.max_messages == 100
        assert config.preserve_system is True
        assert config.strategy == "sliding_window"
    
    def test_custom_values(self):
        """Test custom configuration"""
        config = ConversationConfig(
            max_tokens=8192,
            max_messages=50,
            preserve_system=False,
            strategy="keep_recent"
        )
        
        assert config.max_tokens == 8192
        assert config.max_messages == 50
        assert config.preserve_system is False
        assert config.strategy == "keep_recent"


class TestTokenCounter:
    """Tests for TokenCounter"""
    
    def test_initialization(self):
        """Test counter initialization"""
        counter = TokenCounter(model="gpt-3.5-turbo")
        assert counter.model == "gpt-3.5-turbo"
    
    def test_count_tokens_basic(self):
        """Test basic token counting"""
        counter = TokenCounter()
        
        # Empty string
        assert counter.count_tokens("") == 0
        
        # Simple text
        count = counter.count_tokens("Hello, world!")
        assert count > 0
    
    def test_count_message_tokens(self):
        """Test message token counting"""
        counter = TokenCounter()
        
        message = {"role": "user", "content": "Hello"}
        count = counter.count_message_tokens(message)
        
        # Should include overhead
        content_count = counter.count_tokens("Hello")
        assert count > content_count
    
    def test_count_messages_tokens(self):
        """Test multiple messages token counting"""
        counter = TokenCounter()
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        total = counter.count_messages_tokens(messages)
        
        # Should be sum of individual messages plus base overhead
        individual = sum(counter.count_message_tokens(m) for m in messages)
        assert total >= individual
    
    def test_empty_message_content(self):
        """Test handling of empty content"""
        counter = TokenCounter()
        
        message = {"role": "user", "content": ""}
        count = counter.count_message_tokens(message)
        
        # Should just be overhead
        assert count > 0


class TestSlidingWindowStrategy:
    """Tests for SlidingWindowStrategy"""
    
    def test_no_truncation_needed(self):
        """Test when truncation is not needed"""
        strategy = SlidingWindowStrategy()
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        result = strategy.truncate(messages, 1000, counter)
        
        assert len(result) == 1
    
    def test_truncate_removes_oldest(self):
        """Test that oldest messages are removed"""
        strategy = SlidingWindowStrategy()
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "Message 1 with some longer content here"},
            {"role": "assistant", "content": "Response 1 with more text to increase tokens"},
            {"role": "user", "content": "Message 2 adding even more content"},
            {"role": "assistant", "content": "Response 2 with additional words"},
            {"role": "user", "content": "Message 3 the final message"},
        ]
        
        # Calculate actual tokens needed
        total_tokens = counter.count_messages_tokens(messages)
        # Use a limit that will force truncation (about half)
        result = strategy.truncate(messages, total_tokens // 2, counter)
        
        # Should have fewer messages
        assert len(result) < len(messages)
        # Most recent message should remain
        if len(result) > 0:
            assert result[-1]["content"] == "Message 3 the final message"
    
    def test_preserve_system_message(self):
        """Test system message preservation"""
        strategy = SlidingWindowStrategy()
        counter = TokenCounter()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        
        result = strategy.truncate(messages, 100, counter, preserve_system=True)
        
        # System message should be preserved
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1
    
    def test_empty_messages(self):
        """Test with empty message list"""
        strategy = SlidingWindowStrategy()
        counter = TokenCounter()
        
        result = strategy.truncate([], 1000, counter)
        
        assert result == []


class TestKeepRecentStrategy:
    """Tests for KeepRecentStrategy"""
    
    def test_keep_count(self):
        """Test keeping specified number of messages"""
        strategy = KeepRecentStrategy(keep_count=3)
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "5"},
        ]
        
        result = strategy.truncate(messages, 10000, counter)
        
        assert len(result) == 3
        assert result[0]["content"] == "3"
        assert result[2]["content"] == "5"
    
    def test_preserve_system(self):
        """Test system message preservation"""
        strategy = KeepRecentStrategy(keep_count=2)
        counter = TokenCounter()
        
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]
        
        result = strategy.truncate(messages, 10000, counter, preserve_system=True)
        
        assert len(result) == 3  # 1 system + 2 recent
        assert result[0]["role"] == "system"


class TestImportanceBasedStrategy:
    """Tests for ImportanceBasedStrategy"""
    
    def test_preserve_important(self):
        """Test important messages are preserved"""
        strategy = ImportanceBasedStrategy()
        counter = TokenCounter()
        
        messages = [
            {"role": "user", "content": "Normal 1"},
            {"role": "user", "content": "Important!", "_important": True},
            {"role": "user", "content": "Normal 2"},
            {"role": "user", "content": "Normal 3"},
        ]
        
        result = strategy.truncate(messages, 80, counter)
        
        # Important message should be in result
        important_in_result = any(
            m.get("content") == "Important!" for m in result
        )
        assert important_in_result


class TestConversationManager:
    """Tests for ConversationManager"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = ConversationManager()
        
        assert manager.message_count == 0
        assert manager.token_count > 0  # Base overhead
    
    def test_add_user_message(self):
        """Test adding user message"""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        
        messages = manager.messages
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
    
    def test_add_assistant_message(self):
        """Test adding assistant message"""
        manager = ConversationManager()
        manager.add_assistant_message("Hi there!")
        
        messages = manager.messages
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
    
    def test_set_system_prompt(self):
        """Test setting system prompt"""
        manager = ConversationManager()
        manager.set_system_prompt("You are helpful.")
        
        messages = manager.messages
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
    
    def test_system_prompt_at_beginning(self):
        """Test system prompt is always first"""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        manager.set_system_prompt("System prompt")
        
        messages = manager.messages
        assert messages[0]["role"] == "system"
    
    def test_clear_messages(self):
        """Test clearing messages"""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        manager.clear()
        
        assert manager.message_count == 0
    
    def test_clear_preserve_system(self):
        """Test clearing with system preservation"""
        manager = ConversationManager()
        manager.set_system_prompt("System")
        manager.add_user_message("Hello")
        
        manager.clear(preserve_system=True)
        
        assert manager.message_count == 1
        assert manager.messages[0]["role"] == "system"
    
    def test_pop_last(self):
        """Test popping last message"""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        popped = manager.pop_last()
        
        assert popped["role"] == "assistant"
        assert manager.message_count == 1
    
    def test_pop_last_empty(self):
        """Test popping from empty conversation"""
        manager = ConversationManager()
        
        result = manager.pop_last()
        
        assert result is None
    
    def test_get_messages_for_api(self):
        """Test getting API-ready messages"""
        manager = ConversationManager()
        manager.add_user_message("Hello", important=True)
        
        messages = manager.get_messages_for_api()
        
        # Should not include internal metadata
        assert "_important" not in messages[0]
        assert "role" in messages[0]
        assert "content" in messages[0]
    
    def test_get_stats(self):
        """Test getting conversation stats"""
        manager = ConversationManager()
        manager.set_system_prompt("System")
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        stats = manager.get_stats()
        
        assert stats["message_count"] == 3
        assert stats["token_count"] > 0
        assert stats["roles"]["system"] == 1
        assert stats["roles"]["user"] == 1
        assert stats["roles"]["assistant"] == 1
    
    def test_fork(self):
        """Test forking conversation"""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        
        forked = manager.fork()
        
        # Should be independent
        forked.add_user_message("Goodbye")
        
        assert manager.message_count == 1
        assert forked.message_count == 2
    
    def test_auto_truncation(self):
        """Test automatic truncation on max tokens"""
        config = ConversationConfig(
            max_messages=1000,  # High message limit
            max_tokens=100  # Low token limit to force truncation
        )
        manager = ConversationManager(config)
        
        # Add many messages
        for i in range(20):
            manager.add_user_message(f"Question number {i+1} with some content?")
        
        # Should have been truncated
        assert manager.message_count < 20
        # Token count should be within limit
        assert manager.token_count <= config.max_tokens + 50  # Small buffer for overhead
    
    def test_important_message(self):
        """Test adding important messages"""
        manager = ConversationManager()
        manager.add_user_message("Important!", important=True)
        
        messages = manager.messages
        assert messages[0].get("_important") is True
    
    def test_custom_strategy(self):
        """Test using custom strategy"""
        config = ConversationConfig(
            max_tokens=50,  # Very low token limit
            max_messages=1000
        )
        strategy = KeepRecentStrategy(keep_count=2)
        manager = ConversationManager(config, strategy=strategy)
        
        # Use longer messages to trigger truncation
        for i in range(5):
            manager.add_user_message(f"This is a longer message number {i} to increase token count")
        
        # Should have been truncated
        messages = manager.messages
        assert len(messages) <= 3  # keep_count=2 plus maybe 1 more due to token budget


class TestCreateConversation:
    """Tests for create_conversation factory"""
    
    def test_basic_creation(self):
        """Test basic conversation creation"""
        manager = create_conversation()
        
        assert isinstance(manager, ConversationManager)
    
    def test_with_system_prompt(self):
        """Test creation with system prompt"""
        manager = create_conversation(
            system_prompt="You are a helpful assistant."
        )
        
        messages = manager.messages
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
    
    def test_with_custom_settings(self):
        """Test creation with custom settings"""
        manager = create_conversation(
            max_tokens=8192,
            model="gpt-4",
            strategy="keep_recent"
        )
        
        assert manager.config.max_tokens == 8192
        assert manager.config.model == "gpt-4"
        assert manager.config.strategy == "keep_recent"


class TestThreadSafety:
    """Tests for thread safety"""
    
    def test_concurrent_add(self):
        """Test concurrent message adding"""
        import threading
        
        manager = ConversationManager()
        threads = []
        
        def add_messages():
            for i in range(10):
                manager.add_user_message(f"Thread message {i}")
        
        # Create and start threads
        for _ in range(5):
            t = threading.Thread(target=add_messages)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should not crash and have reasonable count
        assert manager.message_count > 0
