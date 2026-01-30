"""
Conversation management module for LLM API Router.

This module provides intelligent conversation history management,
including token counting and various truncation strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from threading import Lock
import copy


@dataclass
class ConversationConfig:
    """Configuration for conversation management"""
    max_tokens: int = 4096  # Maximum total tokens for conversation
    max_messages: int = 100  # Maximum number of messages to keep
    preserve_system: bool = True  # Always preserve system messages
    model: str = "gpt-3.5-turbo"  # Model for token counting (affects tokenizer)
    strategy: str = "sliding_window"  # Truncation strategy


class TokenCounter:
    """
    Token counter for estimating message token counts.
    
    Supports tiktoken for OpenAI models, falls back to character-based
    estimation for other models.
    """
    
    _tokenizers: Dict[str, Any] = {}
    _lock = Lock()
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for tokenizer selection
        """
        self.model = model
        self._tokenizer = self._get_tokenizer(model)
    
    @classmethod
    def _get_tokenizer(cls, model: str):
        """Get or create tokenizer for model"""
        with cls._lock:
            if model in cls._tokenizers:
                return cls._tokenizers[model]
            
            try:
                import tiktoken
                # Try to get encoding for the specific model
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fall back to cl100k_base for GPT-4 and GPT-3.5-turbo models
                    encoding = tiktoken.get_encoding("cl100k_base")
                cls._tokenizers[model] = encoding
                return encoding
            except ImportError:
                # tiktoken not installed, will use estimation
                return None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count (estimated if tiktoken not available)
        """
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        else:
            # Estimate: ~4 characters per token for English
            # ~2 characters per token for Chinese
            return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens without tiktoken"""
        # Count Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        # Chinese: ~1.5 chars per token, Other: ~4 chars per token
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    def count_message_tokens(self, message: Dict[str, str]) -> int:
        """
        Count tokens in a message.
        
        Args:
            message: Message dictionary with 'role' and 'content'
            
        Returns:
            Token count including message overhead
        """
        # Token overhead per message (role, formatting)
        overhead = 4  # <|start|>role<|sep|>content<|end|>
        
        content = message.get("content", "")
        role = message.get("role", "")
        
        return self.count_tokens(content) + self.count_tokens(role) + overhead
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count total tokens in a list of messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total token count
        """
        total = 3  # Base overhead for messages array
        for message in messages:
            total += self.count_message_tokens(message)
        return total


class TruncationStrategy(ABC):
    """Abstract base class for conversation truncation strategies"""
    
    @abstractmethod
    def truncate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        token_counter: TokenCounter,
        preserve_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Truncate messages to fit within token limit.
        
        Args:
            messages: List of messages to truncate
            max_tokens: Maximum allowed tokens
            token_counter: Token counter instance
            preserve_system: Whether to preserve system messages
            
        Returns:
            Truncated list of messages
        """
        pass


class SlidingWindowStrategy(TruncationStrategy):
    """
    Sliding window truncation strategy.
    
    Removes oldest messages (except system messages if preserved)
    until the conversation fits within the token limit.
    """
    
    def truncate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        token_counter: TokenCounter,
        preserve_system: bool = True
    ) -> List[Dict[str, str]]:
        """Truncate using sliding window"""
        if not messages:
            return messages
        
        # Separate system messages and other messages
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if msg.get("role") == "system" and preserve_system:
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        # Calculate system message tokens
        system_tokens = sum(
            token_counter.count_message_tokens(msg)
            for msg in system_messages
        )
        
        # Available tokens for other messages
        available_tokens = max_tokens - system_tokens - 3  # 3 for base overhead
        
        if available_tokens <= 0:
            # Not enough space, return just system messages
            return system_messages
        
        # Add messages from the end (most recent) until we exceed limit
        result_messages = []
        current_tokens = 0
        
        for msg in reversed(other_messages):
            msg_tokens = token_counter.count_message_tokens(msg)
            if current_tokens + msg_tokens <= available_tokens:
                result_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        # Combine system messages (at the beginning) with other messages
        return system_messages + result_messages


class KeepRecentStrategy(TruncationStrategy):
    """
    Keep N most recent messages strategy.
    
    Keeps a fixed number of most recent messages, always preserving
    system messages if configured.
    """
    
    def __init__(self, keep_count: int = 10):
        """
        Initialize strategy.
        
        Args:
            keep_count: Number of recent messages to keep
        """
        self.keep_count = keep_count
    
    def truncate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        token_counter: TokenCounter,
        preserve_system: bool = True
    ) -> List[Dict[str, str]]:
        """Truncate by keeping N recent messages"""
        if not messages:
            return messages
        
        # Separate system messages
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if msg.get("role") == "system" and preserve_system:
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        # Keep only the most recent messages
        recent_messages = other_messages[-self.keep_count:]
        
        # Now apply token limit as well
        result = system_messages + recent_messages
        total_tokens = token_counter.count_messages_tokens(result)
        
        # If still over limit, remove oldest non-system messages
        while total_tokens > max_tokens and recent_messages:
            recent_messages.pop(0)
            result = system_messages + recent_messages
            total_tokens = token_counter.count_messages_tokens(result)
        
        return result


class ImportanceBasedStrategy(TruncationStrategy):
    """
    Importance-based truncation strategy.
    
    Allows marking messages as important and preserves them during truncation.
    """
    
    def __init__(self, importance_key: str = "_important"):
        """
        Initialize strategy.
        
        Args:
            importance_key: Key in message dict to check for importance flag
        """
        self.importance_key = importance_key
    
    def truncate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        token_counter: TokenCounter,
        preserve_system: bool = True
    ) -> List[Dict[str, str]]:
        """Truncate while preserving important messages"""
        if not messages:
            return messages
        
        # Categorize messages
        system_messages = []
        important_messages = []
        normal_messages = []
        
        for msg in messages:
            if msg.get("role") == "system" and preserve_system:
                system_messages.append(msg)
            elif msg.get(self.importance_key):
                important_messages.append(msg)
            else:
                normal_messages.append(msg)
        
        # Calculate tokens for preserved messages
        preserved_tokens = sum(
            token_counter.count_message_tokens(msg)
            for msg in (system_messages + important_messages)
        ) + 3
        
        available_tokens = max_tokens - preserved_tokens
        
        if available_tokens <= 0:
            # Return just preserved messages
            return system_messages + important_messages
        
        # Fill remaining space with normal messages (most recent first)
        result_normal = []
        current_tokens = 0
        
        for msg in reversed(normal_messages):
            msg_tokens = token_counter.count_message_tokens(msg)
            if current_tokens + msg_tokens <= available_tokens:
                result_normal.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        # Reconstruct in correct order
        return system_messages + result_normal + important_messages


class ConversationManager:
    """
    Manages conversation history with automatic token management.
    
    Features:
    - Automatic token counting
    - Multiple truncation strategies
    - System message preservation
    - Message importance marking
    """
    
    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        strategy: Optional[TruncationStrategy] = None
    ):
        """
        Initialize conversation manager.
        
        Args:
            config: Conversation configuration
            strategy: Custom truncation strategy (overrides config.strategy)
        """
        self.config = config or ConversationConfig()
        self._messages: List[Dict[str, str]] = []
        self._token_counter = TokenCounter(self.config.model)
        self._lock = Lock()
        
        # Set up truncation strategy
        if strategy:
            self._strategy = strategy
        elif self.config.strategy == "sliding_window":
            self._strategy = SlidingWindowStrategy()
        elif self.config.strategy == "keep_recent":
            self._strategy = KeepRecentStrategy()
        elif self.config.strategy == "importance":
            self._strategy = ImportanceBasedStrategy()
        else:
            self._strategy = SlidingWindowStrategy()
    
    @property
    def messages(self) -> List[Dict[str, str]]:
        """Get current messages (copy)"""
        with self._lock:
            return copy.deepcopy(self._messages)
    
    @property
    def token_count(self) -> int:
        """Get current total token count"""
        with self._lock:
            return self._token_counter.count_messages_tokens(self._messages)
    
    @property
    def message_count(self) -> int:
        """Get current message count"""
        with self._lock:
            return len(self._messages)
    
    def set_system_prompt(self, content: str) -> None:
        """
        Set or update system prompt.
        
        Args:
            content: System prompt content
        """
        with self._lock:
            # Remove existing system messages
            self._messages = [
                msg for msg in self._messages
                if msg.get("role") != "system"
            ]
            # Add new system message at the beginning
            self._messages.insert(0, {"role": "system", "content": content})
    
    def add_user_message(self, content: str, important: bool = False) -> None:
        """
        Add a user message.
        
        Args:
            content: Message content
            important: Mark message as important
        """
        msg = {"role": "user", "content": content}
        if important:
            msg["_important"] = True
        self._add_message(msg)
    
    def add_assistant_message(self, content: str, important: bool = False) -> None:
        """
        Add an assistant message.
        
        Args:
            content: Message content
            important: Mark message as important
        """
        msg = {"role": "assistant", "content": content}
        if important:
            msg["_important"] = True
        self._add_message(msg)
    
    def add_message(self, role: str, content: str, important: bool = False) -> None:
        """
        Add a message with specified role.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            important: Mark message as important
        """
        msg = {"role": role, "content": content}
        if important:
            msg["_important"] = True
        
        if role == "system":
            self.set_system_prompt(content)
        else:
            self._add_message(msg)
    
    def _add_message(self, message: Dict[str, str]) -> None:
        """Internal method to add a message"""
        with self._lock:
            self._messages.append(message)
            
            # Check if truncation is needed
            if (len(self._messages) > self.config.max_messages or
                self._token_counter.count_messages_tokens(self._messages) > self.config.max_tokens):
                self._truncate()
    
    def _truncate(self) -> None:
        """Apply truncation strategy"""
        self._messages = self._strategy.truncate(
            self._messages,
            self.config.max_tokens,
            self._token_counter,
            self.config.preserve_system
        )
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for API call.
        
        Removes internal metadata (like _important flag).
        
        Returns:
            List of messages for API
        """
        with self._lock:
            result = []
            for msg in self._messages:
                clean_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                # Copy other standard fields if present
                for key in ["name", "tool_call_id", "tool_calls"]:
                    if key in msg:
                        clean_msg[key] = msg[key]
                result.append(clean_msg)
            return result
    
    def clear(self, preserve_system: bool = True) -> None:
        """
        Clear conversation history.
        
        Args:
            preserve_system: Whether to keep system messages
        """
        with self._lock:
            if preserve_system:
                self._messages = [
                    msg for msg in self._messages
                    if msg.get("role") == "system"
                ]
            else:
                self._messages = []
    
    def pop_last(self) -> Optional[Dict[str, str]]:
        """
        Remove and return the last message.
        
        Returns:
            Last message or None if empty
        """
        with self._lock:
            if self._messages:
                return self._messages.pop()
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary with stats
        """
        with self._lock:
            roles = {}
            for msg in self._messages:
                role = msg.get("role", "unknown")
                roles[role] = roles.get(role, 0) + 1
            
            return {
                "message_count": len(self._messages),
                "token_count": self._token_counter.count_messages_tokens(self._messages),
                "max_tokens": self.config.max_tokens,
                "max_messages": self.config.max_messages,
                "roles": roles,
                "strategy": self.config.strategy
            }
    
    def fork(self) -> 'ConversationManager':
        """
        Create a copy of this conversation manager.
        
        Returns:
            New ConversationManager with same config and messages
        """
        new_manager = ConversationManager(
            config=self.config,
            strategy=self._strategy
        )
        with self._lock:
            new_manager._messages = copy.deepcopy(self._messages)
        return new_manager


# --- Convenience factory functions ---

def create_conversation(
    max_tokens: int = 4096,
    model: str = "gpt-3.5-turbo",
    strategy: str = "sliding_window",
    system_prompt: Optional[str] = None
) -> ConversationManager:
    """
    Create a conversation manager with common settings.
    
    Args:
        max_tokens: Maximum tokens for conversation
        model: Model for token counting
        strategy: Truncation strategy name
        system_prompt: Optional system prompt to set
        
    Returns:
        Configured ConversationManager
    """
    config = ConversationConfig(
        max_tokens=max_tokens,
        model=model,
        strategy=strategy
    )
    manager = ConversationManager(config)
    
    if system_prompt:
        manager.set_system_prompt(system_prompt)
    
    return manager
