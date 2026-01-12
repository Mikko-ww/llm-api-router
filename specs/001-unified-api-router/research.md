# Research: Unified LLM API Router

**Date**: 2026-01-12
**Feature**: Unified LLM API Router

## Decision 1: Architecture & Interface Design

**Decision**: 
Adopt a "Bridge Pattern" where the public `Client` acts as an abstraction and `ProviderAdapter` implementations act as the implementors.
The public interface will strictly mimic the OpenAI Python SDK v1.x structure (`client.chat.completions.create`).

**Rationale**:
- **Interface First**: OpenAI's SDK is the industry standard. Mimicking it reduces the learning curve for developers.
- **Decoupling**: The `Client` class handles user interaction and validation, while `ProviderAdapter` subclasses handle the nitty-gritty of specific provider APIs (OpenAI, Anthropic, etc.).
- **Extensibility**: Adding a new provider only requires implementing a new `ProviderAdapter` subclass without changing the core `Client` logic.

**Alternatives Considered**:
- *Generic "Router" Class*: A single class with conditional logic (if provider == 'openai': ...). **Rejected**: Violates Open/Closed principle; hard to maintain.
- *Plugin System with Entry Points*: Using `pkg_resources` to load providers dynamically. **Rejected**: Overkill for the initial version. Standard subclassing is sufficient and easier to debug.

## Decision 2: Configuration Management

**Decision**: 
Use standard Python `dataclasses` for configuration entities (`ProviderConfig`, `RouterConfig`).
If validation complexity grows, consider `pydantic` in future, but start with standard library to minimize dependencies (unless `pydantic` is already in the project, but `uv` suggests a modern stack where `pydantic` is common. *Correction*: Spec allows standard dependencies. Given "Python >= 3.10", dataclasses are robust). 
**Update**: To ensure robust validation (e.g. URL formats, required keys), we will use `pydantic` if available, or strictly typed `dataclasses`. Given the "Unified" nature, strict validation is key. We will stick to `dataclasses` with manual validation in `__post_init__` to keep the core lightweight, unless `pydantic` is explicitly requested.

**Rationale**:
- **Simplicity**: Dataclasses are built-in and sufficient for holding config.
- **Explicit**: Autocomplete works out of the box.

## Decision 3: HTTP Client Strategy

**Decision**: 
Use `httpx` as the sole HTTP engine.
The `ProviderAdapter` abstract base class will enforce both Sync and Async methods:
- `send_request(request: UnifiedRequest, client: httpx.Client) -> UnifiedResponse`
- `send_request_async(request: UnifiedRequest, client: httpx.AsyncClient) -> UnifiedResponse`
- (And corresponding streaming methods)

**Rationale**:
- **Constraint Compliance**: Spec requires `httpx`.
- **Async/Sync Parity**: `httpx` offers a consistent API for both sync and async, simplifying the adapter logic.

## Decision 4: Streaming Implementation

**Decision**: 
Use `httpx`'s `stream()` context manager.
The Adapter must yield `UnifiedChunk` objects.
For SSE (Server-Sent Events), we will implement a lightweight parser or use `httpx-sse` if acceptable, but a custom lightweight parser is often safer for minimal dependencies. We will implement a basic SSE line parser as it's just text processing (`data: ...`).

**Rationale**:
- **Control**: Manual SSE parsing allows handling provider quirks (e.g. Anthropic's different event types vs OpenAI's).

## Open Questions (Resolved)

- **Q**: How to handle different model parameters (e.g. `max_tokens` vs `max_output_tokens`)?
- **A**: The `UnifiedRequest` will use standard OpenAI naming (`max_tokens`). The `ProviderAdapter` is responsible for mapping this to the specific provider's expected field name.

- **Q**: Project Structure?
- **A**: Will use a standard `src/` layout.
  `src/llm_api_router/`
    `__init__.py`
    `client.py`
    `types.py` (UnifiedRequest, UnifiedResponse, etc.)
    `exceptions.py`
    `providers/`
        `base.py`
        `openai.py`
        `anthropic.py` (future)
