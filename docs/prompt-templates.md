# Prompt Templates

LLM API Router 提供了灵活的 Prompt 模板系统，帮助你创建结构化、可复用的提示词。

## 快速开始

```python
from llm_api_router.templates import render_template, render_messages

# 使用内置模板
result = render_template("summarize", text="要总结的长文本...")

# 生成消息列表（用于 LLM API）
messages = render_messages("translate", 
    source_language="中文",
    target_language="英文",
    text="你好，世界！"
)
```

## 模板语法

### 变量替换

使用 `{{variable}}` 语法插入变量：

```python
from llm_api_router.templates import TemplateEngine

engine = TemplateEngine()
result = engine.render("Hello, {{name}}!", {"name": "World"})
# 输出: Hello, World!
```

### 默认值

使用 `|` 指定默认值：

```python
result = engine.render("Language: {{lang|Python}}", {})
# 输出: Language: Python

result = engine.render("Language: {{lang|Python}}", {"lang": "Go"})
# 输出: Language: Go
```

### 条件块

使用 `{% if %}...{% endif %}` 进行条件渲染：

```python
template = """
{% if show_details %}
详细信息: {{details}}
{% endif %}
"""

result = engine.render(template, {"show_details": True, "details": "..."})
```

支持 else 分支：

```python
template = "{% if is_admin %}管理员{% else %}用户{% endif %}"
```

### 循环

使用 `{% for %}...{% endfor %}` 进行循环：

```python
template = """
项目列表:
{% for item in items %}- {{item}}
{% endfor %}
"""

result = engine.render(template, {"items": ["A", "B", "C"]})
```

### 过滤器

使用 `|filter:` 语法应用过滤器：

```python
# 内置过滤器
engine.render("{{name|upper:}}", {"name": "hello"})   # HELLO
engine.render("{{name|lower:}}", {"name": "HELLO"})   # hello
engine.render("{{name|title:}}", {"name": "hello"})   # Hello

# 自定义过滤器
engine.register_filter("reverse", lambda s: s[::-1])
engine.render("{{text|reverse:}}", {"text": "hello"})  # olleh
```

## 创建模板

### 基本模板

```python
from llm_api_router.templates import PromptTemplate

template = PromptTemplate(
    name="greeting",
    template="Hello, {{name}}!",
    description="简单的问候模板"
)
```

### 带系统提示的模板

```python
template = PromptTemplate(
    name="qa",
    template="问题: {{question}}",
    system_prompt="你是一个有帮助的助手。请准确回答用户的问题。",
    description="问答模板"
)
```

### 带默认值的模板

```python
template = PromptTemplate(
    name="translate",
    template="将以下{{source}}翻译成{{target}}: {{text}}",
    default_values={
        "source": "中文",
        "target": "英文"
    }
)
```

## 使用注册表

### 注册和获取模板

```python
from llm_api_router.templates import TemplateRegistry

registry = TemplateRegistry()

# 注册模板
registry.register(template)

# 从字典注册
registry.register_from_dict({
    "name": "custom",
    "template": "{{content}}",
    "description": "自定义模板"
})

# 获取模板
template = registry.get("custom")

# 列出所有模板
names = registry.list_templates()
```

### 渲染模板

```python
# 渲染为字符串
result = registry.render("greeting", name="World")

# 渲染为消息列表
messages = registry.render_messages("qa", question="什么是 AI?")
# [
#   {"role": "system", "content": "你是一个有帮助的助手..."},
#   {"role": "user", "content": "问题: 什么是 AI?"}
# ]
```

## 内置模板

LLM API Router 提供了 10+ 个常用内置模板：

| 模板名 | 描述 | 主要变量 |
|--------|------|----------|
| `summarize` | 文本摘要 | `text` |
| `translate` | 翻译 | `text`, `source_language`, `target_language` |
| `qa` | 问答 | `question`, `context`(可选) |
| `code_review` | 代码审查 | `code`, `language` |
| `explain_code` | 代码解释 | `code`, `language` |
| `rewrite` | 重写文本 | `text`, `style` |
| `grammar_fix` | 语法修正 | `text` |
| `extract` | 信息提取 | `text`, `extract_type` |
| `classify` | 文本分类 | `text`, `categories` |
| `sentiment` | 情感分析 | `text` |

### 使用内置模板

```python
from llm_api_router.templates import render_template, render_messages

# 摘要
summary = render_template("summarize", text="长文本...")

# 翻译
translation = render_template("translate", 
    text="Hello",
    source_language="英文",
    target_language="中文"
)

# 代码审查
messages = render_messages("code_review",
    code="def foo(): pass",
    language="python"
)
```

## 全局注册表

```python
from llm_api_router.templates import get_template_registry

# 获取全局注册表（已包含内置模板）
registry = get_template_registry()

# 添加自定义模板到全局注册表
registry.register(my_template)
```

## 与 Client 集成

```python
from llm_api_router import Client
from llm_api_router.templates import render_messages

# 创建客户端
client = Client(config)

# 使用模板生成消息
messages = render_messages("summarize", text="要总结的文本...")

# 发送请求
response = client.chat.completions.create(
    messages=messages,
    model="gpt-4"
)
```

## 严格模式

默认情况下，缺少的变量会保留原始占位符。启用严格模式会抛出错误：

```python
# 非严格模式（默认）
result = registry.render("greeting", strict=False)  # 保留 {{name}}

# 严格模式
result = registry.render("greeting", strict=True)   # 抛出 ValueError
```

## 最佳实践

1. **使用描述性变量名**：使模板意图清晰
2. **提供合理的默认值**：减少必需参数数量
3. **编写系统提示**：为模板提供上下文和角色设定
4. **组织模板注册表**：按用途分类管理模板
5. **利用内置模板**：从内置模板开始，根据需要自定义
