"""Mock data and fixtures for provider testing."""
from typing import Dict, Any


def get_openai_chat_response() -> Dict[str, Any]:
    """Mock OpenAI chat completion response."""
    return {
        "id": "chatcmpl-9hPLb2QkApJzETC5FNq4XqtXgZGz1",
        "object": "chat.completion",
        "created": 1720513699,
        "model": "gpt-3.5-turbo-0125",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm an AI assistant. How can I help you today?"
                },
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 15,
            "total_tokens": 27
        },
        "system_fingerprint": None
    }


def get_anthropic_chat_response() -> Dict[str, Any]:
    """Mock Anthropic chat completion response."""
    return {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello! I'm Claude, an AI assistant. How can I help you today?"
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 15
        }
    }


def get_gemini_chat_response() -> Dict[str, Any]:
    """Mock Gemini chat completion response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello! I'm Gemini, a large language model. How can I assist you?"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": []
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 15,
            "totalTokenCount": 27
        }
    }


def get_deepseek_chat_response() -> Dict[str, Any]:
    """Mock DeepSeek chat completion response."""
    return {
        "id": "chatcmpl-deepseek-123",
        "object": "chat.completion",
        "created": 1720513699,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm DeepSeek. How can I help you?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 12,
            "total_tokens": 22
        }
    }


def get_openai_stream_chunks() -> list[str]:
    """Mock OpenAI streaming response chunks."""
    return [
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        'data: [DONE]\n\n'
    ]


def get_anthropic_stream_chunks() -> list[str]:
    """Mock Anthropic streaming response chunks."""
    return [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":12,"output_tokens":0}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    ]


# Sample request payloads
SAMPLE_MESSAGES = [
    {"role": "user", "content": "Hello, how are you?"}
]

SAMPLE_MESSAGES_WITH_SYSTEM = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

SAMPLE_MESSAGES_CONVERSATION = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"}
]


def get_openai_function_call_response() -> Dict[str, Any]:
    """Mock OpenAI function calling response."""
    return {
        "id": "chatcmpl-function-test",
        "object": "chat.completion",
        "created": 1720513699,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 82,
            "completion_tokens": 17,
            "total_tokens": 99
        }
    }


def get_anthropic_function_call_response() -> Dict[str, Any]:
    """Mock Anthropic tool use response."""
    return {
        "id": "msg_tool_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01A09q90qw90lq917835lq9",
                "name": "get_weather",
                "input": {
                    "location": "San Francisco, CA",
                    "unit": "fahrenheit"
                }
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 82,
            "output_tokens": 17
        }
    }


def get_openai_function_call_stream_chunks() -> list[str]:
    """Mock OpenAI streaming function call response chunks."""
    return [
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"location"}}]},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":  \\"San"}}]},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Francisco, CA\\"}"}}]},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        'data: [DONE]\n\n'
    ]
