# Implementation Tasks: Unified LLM API Router

**Branch**: `001-unified-api-router` | **Spec**: [spec.md](../spec.md) | **Plan**: [plan.md](../plan.md)

## Phase 1: Setup
*Goal: Initialize the project structure and dependencies.*

- [x] T001 初始化项目目录结构 (`src/llm_api_router`, `tests`)
- [x] T002 配置 `pyproject.toml`，添加 `httpx` 依赖和构建配置

## Phase 2: Foundational (Blocking)
*Goal: Implement core data structures, exceptions, and base abstractions required by all user stories.*

- [x] T003 [P] 在 `src/llm_api_router/types.py` 中实现核心数据类 (`ProviderConfig`, `UnifiedRequest`, `UnifiedResponse`, `Usage`, `Choice`, `Message`)
- [x] T004 [P] 在 `src/llm_api_router/exceptions.py` 中实现统一异常体系 (`LLMRouterError`, `AuthenticationError`, `RateLimitError` 等)
- [x] T005 在 `src/llm_api_router/providers/base.py` 中定义 `BaseProvider` 抽象基类 (ABC)，包含所有必须实现的抽象方法签名

## Phase 3: User Story 1 - Basic Provider Integration (OpenAI)
*Goal: Enable basic synchronous requests to OpenAI using the unified Client.*
*Priority: P1*

- [x] T006 [US1] 在 `src/llm_api_router/providers/openai.py` 中实现 `OpenAIProvider` 类的同步基础方法 (`convert_request`, `convert_response`, `send_request`)
- [x] T007 [US1] 在 `src/llm_api_router/client.py` 中实现 `Client` 类，包含初始化逻辑和 `chat.completions.create` 方法的基本代理逻辑
- [x] T008 [US1] 创建 `tests/unit/test_providers/test_openai.py`，使用 Mock 测试基本的 OpenAI 请求和响应映射

## Phase 4: User Story 2 - Zero-Code Model Switching
*Goal: Ensure clients can switch providers purely via configuration.*
*Priority: P1*

- [x] T009 [US2] 在 `src/llm_api_router/client.py` 中完善构造函数，实现基于 `ProviderConfig.provider_type` 动态加载对应 `ProviderAdapter` 的工厂逻辑
- [x] T010 [US2] 创建 `tests/unit/test_client.py`，验证修改配置对象即可切换底层 Provider 实例，且无需修改调用代码

## Phase 5: User Story 3 - Unified Streaming Support
*Goal: Support streaming responses with unified chunk objects.*
*Priority: P2*

- [x] T011 [US3] 在 `src/llm_api_router/types.py` 中补充 `UnifiedChunk` 和 `ChunkChoice` 数据类定义
- [x] T012 [US3] 在 `src/llm_api_router/providers/openai.py` 中实现 `stream_request` 方法，处理 SSE 流式响应并转换为 `UnifiedChunk`
- [x] T013 [US3] 在 `src/llm_api_router/client.py` 中更新 `create` 方法，当 `stream=True` 时返回迭代器
- [x] T014 [US3] 创建 `tests/unit/test_providers/test_openai_stream.py`，模拟 SSE 流数据验证流式解析逻辑

## Phase 6: User Story 4 - Asynchronous Support
*Goal: Provide async/await support for high concurrency.*
*Priority: P2*

- [x] T015 [US4] 在 `src/llm_api_router/providers/openai.py` 中实现异步方法 `send_request_async` 和 `stream_request_async`
- [x] T016 [P] [US4] 在 `src/llm_api_router/client.py` 中实现 `AsyncClient` 类，镜像 `Client` 的接口但使用异步调用
- [x] T017 [US4] 创建 `tests/unit/test_client_async.py`，测试 `AsyncClient` 的并发请求处理能力

## Phase 7: Polish & Cross-Cutting
*Goal: Ensure code quality, type safety, and documentation standards.*

- [x] T018 在所有 Python 文件中添加/检查简体中文 Docstrings 和注释
- [x] T019 运行静态类型检查 (如 mypy) 并修复所有类型警告，确保 `types.py` 和接口定义的类型安全
- [x] T020 验证所有公共 API 是否符合 OpenAI 风格接口规范 (Interface First check)

## Dependencies

1.  **Setup & Foundation**: T001-T005 must be completed first.
2.  **US1**: Depends on Foundation. T006 and T007 can be parallelized if interfaces are strict.
3.  **US2**: Depends on US1 (Client existence).
4.  **US3 & US4**: Can be developed in parallel after US1/US2 are stable.

## Parallel Execution Opportunities

-   **Foundation**: T003 (Types) and T004 (Exceptions) are independent.
-   **US4**: T016 (AsyncClient shell) can be written while T015 (Async Provider impl) is being developed.

## Implementation Strategy

We will start with the **Foundation** to establish the core interfaces. Then we build the **Sync OpenAI** path (US1) as the MVP. Once stable, we verify the **Switching** capability (US2). Finally, we extend the system with **Streaming** (US3) and **Async** (US4) capabilities as advanced features.
