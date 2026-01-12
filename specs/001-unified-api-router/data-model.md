# Data Model: Unified LLM API Router

## Core Configuration

### ProviderConfig
*Configuration for a specific LLM provider.*

| Field | Type | Description | Required | Default |
|-------|------|-------------|----------|---------|
| `provider_type` | `str` | Type of provider (e.g., "openai", "azure") | Yes | - |
| `api_key` | `str` | API Key for authentication | Yes | - |
| `base_url` | `str` | Base URL for API requests | No | Provider default |
| `api_version` | `str` | API Version (mostly for Azure) | No | None |
| `default_model` | `str` | Default model ID to use | No | None |
| `extra_headers` | `Dict[str, str]` | Additional HTTP headers | No | {} |

## Unified Data Structures

### UnifiedRequest
*Standardized request object, mirroring OpenAI's ChatCompletion parameters.*

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Dict[str, str]]` | List of message objects (role, content) |
| `model` | `str` | Model ID to target |
| `temperature` | `float` | Sampling temperature (0.0 - 2.0) |
| `max_tokens` | `int` | Maximum tokens to generate |
| `stream` | `bool` | Whether to stream responses |
| `top_p` | `float` | Nucleus sampling parameter |
| `stop` | `List[str]` | Stop sequences |

### UnifiedResponse
*Standardized response object for non-streaming requests.*

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique request ID |
| `object` | `str` | Object type (always "chat.completion") |
| `created` | `int` | Unix timestamp of creation |
| `model` | `str` | Model used for generation |
| `choices` | `List[Choice]` | List of completion choices |
| `usage` | `Usage` | Token usage statistics |

### UnifiedChunk
*Standardized chunk object for streaming requests.*

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique request ID |
| `object` | `str` | Object type (always "chat.completion.chunk") |
| `created` | `int` | Unix timestamp |
| `model` | `str` | Model used |
| `choices` | `List[ChunkChoice]` | List of chunk choices |

## Sub-Entities

### Choice
| Field | Type | Description |
|-------|------|-------------|
| `index` | `int` | Index of the choice |
| `message` | `Message` | The generated message (role, content) |
| `finish_reason` | `str` | Why generation stopped (stop, length, etc.) |

### ChunkChoice
| Field | Type | Description |
|-------|------|-------------|
| `index` | `int` | Index of the choice |
| `delta` | `Message` | The generated content delta (role, content) |
| `finish_reason` | `str` | Why generation stopped (nullable) |

### Usage
| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Tokens in prompt |
| `completion_tokens` | `int` | Tokens generated |
| `total_tokens` | `int` | Total tokens used |
