# Multi-Provider Integration Spec

## 1. Goal
Support 5 additional LLM providers:
- DeepSeek
- Anthropic
- Google Gemini
- Zhipu (ChatGLM)
- Alibaba (Tongyi/DashScope)

## 2. Architecture
All providers inherit from `BaseProvider`.
The `Client` class factory `_get_provider` method routes based on `provider_type`.

## 3. Provider Details

### 3.1 DeepSeek
- **Type**: `deepseek`
- **Base URL**: `https://api.deepseek.com`
- **Auth**: Header `Authorization: Bearer <key>`
- **Compat**: OpenAI Compatible.
- **Notes**: Simplest integration.

### 3.2 Anthropic
- **Type**: `anthropic`
- **Base URL**: `https://api.anthropic.com/v1`
- **Auth**: Header `x-api-key`.
- **Headers**: `anthropic-version: 2023-06-01`
- **Req Body**:
  - `messages`: List of `{role, content}`. No specific system message in `messages` typically (moved to top level parameter in new API), or handled via conversion.
  - `system`: Top level parameter string.
  - `max_tokens`: `max_tokens`
- **Response**: Distinct JSON structure.

### 3.3 Google Gemini
- **Type**: `gemini`
- **Base URL**: `https://generativelanguage.googleapis.com/v1beta`
- **Auth**: Query param `?key=KEY` or Header `x-goog-api-key`.
- **Req Body**:
  - `contents`: `[{role: "user", parts: [{text: "..."}]}]`
  - Roles are `user` and `model`.
- **Response**: `candidates[0].content.parts[0].text`

### 3.4 Zhipu (BigModel)
- **Type**: `zhipu`
- **Base URL**: `https://open.bigmodel.cn/api/paas/v4`
- **Auth**: Header `Authorization: Bearer <token>`.
- **Complexity**:
  - Token is generated. If V4 SDK standard API key is used, it might need local JWT generation (Signing `id`, `secret` with TTL).
  - *Decision*: We will implement a lightweight JWT generator using `time` + `base64` + `hmac` (standard lib) to avoid heavy dependencies if possible, or verify if they allow direct API Key usage now.
  - Docs say: `Authorization: Bearer <generated_jwt>`. We need a helper `generate_zhipu_token(apikey, ttl)`.

### 3.5 Alibaba (DashScope)
- **Type**: `aliyun`
- **Base URL**: `https://dashscope.aliyuncs.com/api/v1`
- **Auth**: Header `Authorization: Bearer <key>`
- **Req Body**:
  - `input`: `{messages: [...]}` wrapper.
  - `parameters`: `{result_format: "message"}`
- **Response**: `output.choices[0].message.content`
