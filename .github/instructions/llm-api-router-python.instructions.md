---
description: "用于 llm-api-router 项目的 Python 开发、Provider 适配、测试与文档更新。适用于新增/修改 src、tests、examples、README 与 docs 时，确保统一接口、类型安全、异常处理和可测试性。"
name: "llm-api-router 开发规范（中文）"
applyTo: "src/**/*.py, tests/**/*.py, examples/**/*.py, main.py, README.md, README_zh.md, docs/**/*.md, pyproject.toml"
---
# llm-api-router 项目开发 Instructions（简体中文）

## 项目目标与架构

- 目标：提供统一、稳定、类型安全的多厂商 LLM 调用接口。
- 架构：遵循 Bridge + Factory 思路。
  - 统一入口：`Client` / `AsyncClient`
  - Provider 装配：`ProviderFactory`
  - Provider 抽象基类：`BaseProvider`
- 新功能或修复必须优先复用现有抽象，不要绕过统一层直接写“厂商特例分叉逻辑”。

## 代码风格与可维护性

- Python 版本基线：3.10+
- 命名约定：
  - 类名：`PascalCase`
  - 函数/变量：`snake_case`
  - 常量：`UPPER_SNAKE_CASE`
  - 私有成员：前缀 `_`
- 公开函数、方法和关键内部接口必须带完整类型注解。
- 新增/修改逻辑时，优先小步改动，避免与需求无关的重构。
- 代码注释聚焦“为什么”，避免重复“代码正在做什么”。

## 同步与异步一致性

- 若修改 `Client` 侧能力，检查是否需要同步更新 `AsyncClient` 对应能力。
- 同步上下文使用 `with Client(...)`，异步上下文使用 `async with AsyncClient(...)`。
- 避免只修同步或只修异步导致行为不一致。

## Provider 开发与变更规则

当新增 Provider 或修改 `src/llm_api_router/providers/` 下实现时：

1. 继承 `BaseProvider` 并实现必要接口：
   - `convert_request`
   - `convert_response`
   - `send_request` / `send_request_async`
   - `stream_request` / `stream_request_async`
2. 错误处理必须走统一异常语义（认证、限流、服务端错误等）。
3. 重试能力复用基类提供的重试装饰器，不重复发明重试机制。
4. 如果增加新 provider_type，必须同步更新工厂映射与相应测试。

## 类型安全与异常处理

- 保持 `mypy` 可通过，禁止引入明显破坏类型检查的写法。
- 统一抛出项目异常体系中的异常，错误信息应包含定位上下文（provider、状态码、关键细节）。
- 不要吞掉异常；若转换异常，保留关键信息并保证调用方可判定错误类型。

## 缓存、流式与行为一致性

- 流式响应与非流式响应逻辑必须明确区分。
- 缓存策略仅用于可缓存路径，避免污染流式调用语义。
- 修改缓存键、序列化或反序列化逻辑时，需补充对应测试覆盖。

## 测试要求

- 每次功能变更至少包含：
  - 1 个正向测试（正常路径）
  - 1 个异常/边界测试（失败路径）
- 优先使用 `tests/conftest.py` 中共享 fixtures，减少重复样板。
- 测试目录约定：
  - 单元测试：`tests/unit/`
  - 集成测试：`tests/integration/`
- 涉及异步逻辑必须覆盖异步测试路径。

## 文档与示例联动

- 影响用户使用方式的变更，需要同步更新以下至少一处：
  - `README.md` / `README_zh.md`
  - `docs/` 对应专题文档
  - `examples/` 对应示例
- 文档和示例必须与当前 API 行为一致，避免“代码已变、示例失效”。

## 提交前自检（代理执行时默认遵循）

- 运行测试并确认通过（至少与改动直接相关的测试）。
- 运行类型检查并确认无新增类型错误。
- 检查是否误改无关文件。
- 说明变更影响范围：接口、行为、兼容性、文档。

## 修改建议优先级

1. 正确性（接口语义、异常语义、同步异步一致）
2. 可测试性（可重复验证）
3. 类型安全（mypy 友好）
4. 可读性与最小改动
5. 性能优化（在不破坏语义前提下）
