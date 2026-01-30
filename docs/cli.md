# CLI 命令参考

LLM API Router 提供命令行工具，方便测试、调试和日常使用。

## 安装

```bash
pip install llm-api-router[cli]
```

## 命令概览

```bash
llm-router --help
```

| 命令 | 说明 |
|------|------|
| `test` | 测试提供商连接 |
| `validate` | 验证配置文件 |
| `benchmark` | 运行性能基准测试 |
| `models` | 列出可用模型 |
| `chat` | 交互式聊天 |

## test - 测试连接

测试与 LLM 提供商的连接：

```bash
# 基本使用
llm-router test openai --api-key sk-xxx

# 指定模型
llm-router test openai --api-key sk-xxx --model gpt-4o

# 测试 Ollama（本地）
llm-router test ollama --base-url http://localhost:11434

# JSON 输出
llm-router test openai --api-key sk-xxx --output json
```

### 参数

| 参数 | 说明 |
|------|------|
| `PROVIDER` | 提供商名称（必需） |
| `--api-key, -k` | API 密钥 |
| `--base-url, -u` | 自定义 API 地址 |
| `--model, -m` | 测试模型 |
| `--output, -o` | 输出格式：table/json |

## validate - 验证配置

验证配置文件格式：

```bash
# 验证 JSON 配置
llm-router validate config.json

# 验证 YAML 配置
llm-router validate config.yaml

# 详细输出
llm-router validate config.json --verbose
```

### 参数

| 参数 | 说明 |
|------|------|
| `CONFIG_PATH` | 配置文件路径（必需） |
| `--verbose, -v` | 显示详细信息 |

### 配置文件格式

```json
{
  "provider_type": "openai",
  "api_key": "sk-xxx",
  "default_model": "gpt-3.5-turbo",
  "timeout": 30,
  "max_retries": 3
}
```

## benchmark - 性能测试

运行性能基准测试：

```bash
# 基本基准测试
llm-router benchmark openai --api-key sk-xxx

# 指定请求数量
llm-router benchmark openai --api-key sk-xxx --requests 20

# 并发测试
llm-router benchmark openai --api-key sk-xxx --concurrency 5

# 指定提示词
llm-router benchmark openai --api-key sk-xxx --prompt "Hello, how are you?"
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `PROVIDER` | 提供商名称（必需） | - |
| `--api-key, -k` | API 密钥 | - |
| `--model, -m` | 测试模型 | 提供商默认 |
| `--requests, -n` | 请求数量 | 10 |
| `--concurrency, -c` | 并发数 | 1 |
| `--prompt, -p` | 测试提示词 | "Say hello" |
| `--output, -o` | 输出格式 | table |

### 输出示例

```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ 指标            ┃ 值         ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 总请求数        │ 10         │
│ 成功请求        │ 10         │
│ 失败请求        │ 0          │
│ 平均延迟        │ 523.4ms    │
│ P50 延迟        │ 498.2ms    │
│ P95 延迟        │ 712.8ms    │
│ P99 延迟        │ 782.1ms    │
│ 吞吐量          │ 1.9 req/s  │
└─────────────────┴────────────┘
```

## models - 列出模型

列出提供商支持的模型：

```bash
# 列出 OpenAI 模型
llm-router models openai

# 列出 Anthropic 模型
llm-router models anthropic

# JSON 格式输出
llm-router models openai --output json
```

### 参数

| 参数 | 说明 |
|------|------|
| `PROVIDER` | 提供商名称（必需） |
| `--output, -o` | 输出格式：table/json |

## chat - 交互式聊天

启动交互式聊天会话：

```bash
# 基本聊天
llm-router chat openai --api-key sk-xxx

# 指定模型
llm-router chat openai --api-key sk-xxx --model gpt-4o

# 设置系统提示
llm-router chat openai --api-key sk-xxx --system "你是一个专业的程序员"

# 与本地 Ollama 聊天
llm-router chat ollama --base-url http://localhost:11434 --model llama3.2
```

### 参数

| 参数 | 说明 |
|------|------|
| `PROVIDER` | 提供商名称（必需） |
| `--api-key, -k` | API 密钥 |
| `--base-url, -u` | 自定义 API 地址 |
| `--model, -m` | 使用的模型 |
| `--system, -s` | 系统提示词 |
| `--no-stream` | 禁用流式输出 |

### 聊天命令

在聊天中可使用的命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/clear` | 清空对话历史 |
| `/history` | 显示对话历史 |
| `/exit` | 退出聊天 |

## 全局选项

```bash
# 显示版本
llm-router --version
llm-router -v

# 显示帮助
llm-router --help
```

## 环境变量

CLI 支持通过环境变量配置 API 密钥：

```bash
export OPENAI_API_KEY="sk-xxx"
export ANTHROPIC_API_KEY="sk-ant-xxx"

# 然后可以省略 --api-key 参数
llm-router test openai
llm-router chat anthropic
```

## 使用示例

### 快速测试所有提供商

```bash
#!/bin/bash
providers=("openai" "anthropic" "gemini")

for provider in "${providers[@]}"; do
    echo "Testing $provider..."
    llm-router test "$provider" --output json
done
```

### 批量基准测试

```bash
#!/bin/bash
# 测试不同模型的性能
for model in "gpt-3.5-turbo" "gpt-4o" "gpt-4o-mini"; do
    echo "Benchmarking $model..."
    llm-router benchmark openai --model "$model" --requests 5
done
```
