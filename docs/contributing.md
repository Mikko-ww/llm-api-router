# 贡献指南

感谢你有兴趣为 LLM API Router 做出贡献！本指南将帮助你了解如何参与项目开发。

## 行为准则

- 尊重所有参与者
- 建设性地讨论问题
- 专注于改进项目

## 如何贡献

### 报告 Bug

1. 检查 [Issue 列表](https://github.com/your-repo/llm-api-router/issues) 确认问题未被报告
2. 创建新 Issue，包含：
   - 清晰的标题和描述
   - 复现步骤
   - 预期行为 vs 实际行为
   - 环境信息（Python 版本、操作系统等）
   - 相关代码和错误信息

### 提出功能建议

1. 搜索现有 Issue 确认建议未被提出
2. 创建 Feature Request Issue，说明：
   - 功能描述
   - 使用场景
   - 可能的实现方式

### 提交代码

#### 1. Fork 仓库

```bash
git clone https://github.com/YOUR_USERNAME/llm-api-router.git
cd llm-api-router
```

#### 2. 创建开发环境

```bash
# 使用 uv（推荐）
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 或使用 pip
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

#### 3. 创建特性分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

#### 4. 编写代码

遵循项目代码风格：

- 使用 Python 3.10+ 特性
- 添加类型注解
- 编写 docstring
- 遵循 PEP 8

#### 5. 编写测试

```bash
# 运行测试
pytest tests/

# 运行特定测试
pytest tests/unit/test_your_feature.py -v

# 运行测试并查看覆盖率
pytest tests/ --cov=src/llm_api_router --cov-report=html
```

#### 6. 格式化代码

```bash
# 格式化代码
black src/ tests/

# 检查导入排序
isort src/ tests/

# 类型检查
mypy src/
```

#### 7. 提交更改

```bash
git add .
git commit -m "feat: add your feature description"
```

提交信息格式：

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `test:` 测试相关
- `refactor:` 代码重构
- `style:` 代码风格
- `chore:` 其他更改

#### 8. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

在 GitHub 上创建 Pull Request。

## 代码规范

### 类型注解

```python
def create_completion(
    self,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 1.0,
) -> ChatCompletion:
    """创建聊天完成。
    
    Args:
        messages: 消息列表
        model: 模型名称
        temperature: 温度参数
        
    Returns:
        聊天完成响应
    """
    ...
```

### 文档字符串

使用 Google 风格的 docstring：

```python
def function(param1: str, param2: int) -> bool:
    """简短描述。
    
    详细描述（如需要）。
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
        
    Returns:
        返回值描述
        
    Raises:
        ValueError: 错误描述
    """
```

### 测试规范

```python
import pytest
from llm_api_router import Client, ProviderConfig


class TestYourFeature:
    """测试你的功能"""
    
    def test_basic_usage(self):
        """测试基本用法"""
        # Arrange
        config = ProviderConfig(...)
        
        # Act
        result = some_function()
        
        # Assert
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_async_usage(self):
        """测试异步用法"""
        ...
```

## 项目结构

```
llm-api-router/
├── src/llm_api_router/     # 源代码
│   ├── __init__.py         # 包入口
│   ├── client.py           # 客户端实现
│   ├── providers/          # 提供商实现
│   └── ...
├── tests/                  # 测试代码
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试
├── docs/                   # 文档
├── examples/               # 示例代码
└── ...
```

## 发布流程

维护者会处理发布：

1. 更新版本号
2. 更新 CHANGELOG
3. 创建 Release Tag
4. 发布到 PyPI

## 获取帮助

- 查看 [文档](https://llm-api-router.readthedocs.io/)
- 在 GitHub Discussion 讨论
- 提交 Issue 寻求帮助

再次感谢你的贡献！🎉
