# 安装指南

## 系统要求

- Python 3.10 或更高版本
- pip 或 uv 包管理器

## 基础安装

使用 pip 安装：

```bash
pip install llm-api-router
```

使用 uv 安装：

```bash
uv pip install llm-api-router
```

## 可选依赖

### CLI 工具

安装命令行工具依赖：

```bash
pip install llm-api-router[cli]
```

安装后可使用 `llm-router` 命令：

```bash
llm-router --help
llm-router test openai --api-key sk-xxx
llm-router chat anthropic
```

### 文档构建

如果需要本地构建文档：

```bash
pip install llm-api-router[docs]
```

### 开发环境

安装所有开发依赖：

```bash
pip install llm-api-router[dev]
```

### 完整安装

安装所有可选依赖：

```bash
pip install llm-api-router[all]
```

## 从源码安装

```bash
git clone https://github.com/Mikko-ww/llm-api-router.git
cd llm-api-router
pip install -e .
```

## 验证安装

```python
>>> import llm_api_router
>>> print(llm_api_router.__version__)
0.1.2
```

或使用 CLI：

```bash
llm-router --version
```

## 依赖说明

核心依赖（自动安装）：

| 包 | 用途 |
|---|------|
| `httpx` | HTTP 客户端 |
| `python-dotenv` | 环境变量管理 |

可选依赖：

| 包 | 用途 | 安装方式 |
|---|------|---------|
| `typer` | CLI 框架 | `[cli]` |
| `rich` | 终端美化 | `[cli]` |
| `mkdocs-material` | 文档主题 | `[docs]` |
| `redis` | Redis 缓存后端 | 手动安装 |
| `tiktoken` | Token 计数 | 手动安装 |

## 常见问题

### 安装失败：没有找到匹配的版本

确保 Python 版本 >= 3.10：

```bash
python --version
```

### ImportError: No module named 'llm_api_router'

确保在正确的 Python 环境中安装，或检查虚拟环境是否激活。

### CLI 命令未找到

确保安装了 CLI 依赖，并且 Python bin 目录在 PATH 中：

```bash
pip install llm-api-router[cli]
```
