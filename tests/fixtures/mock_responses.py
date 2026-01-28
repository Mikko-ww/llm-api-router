"""Test fixtures for mock API responses."""

# OpenAI mock response
OPENAI_MOCK_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking."
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

# OpenAI streaming mock chunks
OPENAI_MOCK_STREAM_CHUNKS = [
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    'data: [DONE]\n\n'
]

# Anthropic mock response
ANTHROPIC_MOCK_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! I'm doing well, thank you for asking."
        }
    ],
    "model": "claude-3-5-sonnet-20240620",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 12
    }
}

# Gemini mock response
GEMINI_MOCK_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "Hello! I'm doing well, thank you for asking."}
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 12,
        "totalTokenCount": 22
    }
}

# Error responses
ERROR_RESPONSES = {
    "401": {
        "error": {
            "message": "Invalid authentication credentials",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }
    },
    "429": {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    },
    "500": {
        "error": {
            "message": "Internal server error",
            "type": "server_error",
            "code": "internal_error"
        }
    }
}
