"""
Prompt template system for LLM API Router.

This module provides template management and rendering capabilities
for creating structured prompts.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from abc import ABC, abstractmethod


@dataclass
class PromptTemplate:
    """
    A prompt template with variable substitution support.
    
    Attributes:
        name: Unique template identifier
        template: Template string with {{variable}} placeholders
        description: Optional description of the template
        variables: List of variable names (auto-detected if not provided)
        system_prompt: Optional system prompt for this template
        default_values: Default values for variables
    """
    name: str
    template: str
    description: str = ""
    variables: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Auto-detect variables if not provided"""
        if self.variables is None:
            self.variables = self._extract_variables(self.template)
            if self.system_prompt:
                self.variables.extend(self._extract_variables(self.system_prompt))
            self.variables = list(set(self.variables))
    
    @staticmethod
    def _extract_variables(text: str) -> List[str]:
        """Extract variable names from template text"""
        # Match {{variable}} or {{ variable }}
        pattern = r'\{\{\s*(\w+)\s*\}\}'
        return re.findall(pattern, text)


class TemplateEngine:
    """
    Template engine for rendering templates with variable substitution.
    
    Supports:
    - Simple variable substitution: {{variable}}
    - Default values: {{variable|default_value}}
    - Conditional blocks: {% if condition %}...{% endif %}
    - Loop blocks: {% for item in items %}...{% endfor %}
    """
    
    def __init__(self):
        """Initialize template engine"""
        self._filters: Dict[str, Callable] = {
            'upper': str.upper,
            'lower': str.lower,
            'strip': str.strip,
            'title': str.title,
            'capitalize': str.capitalize,
        }
    
    def register_filter(self, name: str, func: Callable) -> None:
        """Register a custom filter function"""
        self._filters[name] = func
    
    def render(
        self,
        template: str,
        variables: Optional[Dict[str, Any]] = None,
        strict: bool = False
    ) -> str:
        """
        Render a template with variable substitution.
        
        Args:
            template: Template string
            variables: Dictionary of variable values
            strict: If True, raise error for missing variables
            
        Returns:
            Rendered string
        """
        if variables is None:
            variables = {}
        
        result = template
        
        # Process conditional blocks first
        result = self._process_conditionals(result, variables)
        
        # Process loop blocks
        result = self._process_loops(result, variables)
        
        # Process variable substitutions
        result = self._process_variables(result, variables, strict)
        
        return result
    
    def _process_variables(
        self,
        template: str,
        variables: Dict[str, Any],
        strict: bool
    ) -> str:
        """Process variable substitutions"""
        # Pattern: {{variable}}, {{variable|default}}, {{variable|filter:arg}}
        pattern = r'\{\{\s*(\w+)(?:\s*\|\s*([^}]+))?\s*\}\}'
        
        def replace(match):
            var_name = match.group(1)
            modifier = match.group(2)
            
            # Get value from variables
            if var_name in variables:
                value = variables[var_name]
            elif modifier and ':' not in modifier:
                # Use modifier as default value
                value = modifier
            elif strict:
                raise ValueError(f"Missing required variable: {var_name}")
            else:
                return match.group(0)  # Keep original
            
            # Apply filter if specified
            if modifier and ':' in modifier:
                parts = modifier.split(':')
                filter_name = parts[0].strip()
                if filter_name in self._filters:
                    value = self._filters[filter_name](str(value))
            
            return str(value)
        
        return re.sub(pattern, replace, template)
    
    def _process_conditionals(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """Process conditional blocks"""
        # Pattern: {% if condition %}...{% endif %}
        # Also supports: {% if condition %}...{% else %}...{% endif %}
        pattern = r'\{%\s*if\s+(\w+)\s*%\}(.*?)(?:\{%\s*else\s*%\}(.*?))?\{%\s*endif\s*%\}'
        
        def replace(match):
            condition_var = match.group(1)
            if_block = match.group(2)
            else_block = match.group(3) or ""
            
            # Evaluate condition
            condition_value = variables.get(condition_var)
            if condition_value:
                return if_block
            else:
                return else_block
        
        return re.sub(pattern, replace, template, flags=re.DOTALL)
    
    def _process_loops(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """Process loop blocks"""
        # Pattern: {% for item in items %}...{% endfor %}
        pattern = r'\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}'
        
        def replace(match):
            item_var = match.group(1)
            list_var = match.group(2)
            loop_body = match.group(3)
            
            items = variables.get(list_var, [])
            if not isinstance(items, (list, tuple)):
                return ""
            
            result = []
            for i, item in enumerate(items):
                loop_vars = variables.copy()
                loop_vars[item_var] = item
                loop_vars['loop_index'] = i
                loop_vars['loop_first'] = i == 0
                loop_vars['loop_last'] = i == len(items) - 1
                
                rendered = self._process_variables(loop_body, loop_vars, False)
                result.append(rendered)
            
            return ''.join(result)
        
        return re.sub(pattern, replace, template, flags=re.DOTALL)


class TemplateRegistry:
    """
    Registry for managing prompt templates.
    
    Provides methods to register, retrieve, and render templates.
    """
    
    def __init__(self):
        """Initialize template registry"""
        self._templates: Dict[str, PromptTemplate] = {}
        self._engine = TemplateEngine()
    
    @property
    def engine(self) -> TemplateEngine:
        """Get the template engine"""
        return self._engine
    
    def register(self, template: PromptTemplate) -> None:
        """
        Register a template.
        
        Args:
            template: PromptTemplate instance
        """
        self._templates[template.name] = template
    
    def register_from_dict(self, data: Dict[str, Any]) -> PromptTemplate:
        """
        Register a template from dictionary.
        
        Args:
            data: Dictionary with template data
            
        Returns:
            Created PromptTemplate
        """
        template = PromptTemplate(**data)
        self.register(template)
        return template
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate or None
        """
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all registered template names"""
        return list(self._templates.keys())
    
    def render(
        self,
        template_name: str,
        strict: bool = False,
        **kwargs
    ) -> str:
        """
        Render a template by name.
        
        Args:
            template_name: Template name
            strict: If True, raise error for missing variables
            **kwargs: Variable values
            
        Returns:
            Rendered string
            
        Raises:
            KeyError: If template not found
        """
        template = self._templates.get(template_name)
        if template is None:
            raise KeyError(f"Template not found: {template_name}")
        
        # Merge default values with provided kwargs
        variables = {**template.default_values, **kwargs}
        
        return self._engine.render(template.template, variables, strict)
    
    def render_messages(
        self,
        template_name: str,
        strict: bool = False,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Render a template as a list of messages.
        
        Args:
            template_name: Template name
            strict: If True, raise error for missing variables
            **kwargs: Variable values
            
        Returns:
            List of message dictionaries
        """
        template = self._templates.get(template_name)
        if template is None:
            raise KeyError(f"Template not found: {template_name}")
        
        variables = {**template.default_values, **kwargs}
        messages = []
        
        # Add system message if present
        if template.system_prompt:
            system_content = self._engine.render(
                template.system_prompt, variables, strict
            )
            messages.append({
                "role": "system",
                "content": system_content
            })
        
        # Add user message
        user_content = self._engine.render(template.template, variables, strict)
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def unregister(self, template_name: str) -> bool:
        """
        Remove a template from registry.
        
        Args:
            template_name: Template name
            
        Returns:
            True if template was removed, False if not found
        """
        if template_name in self._templates:
            del self._templates[template_name]
            return True
        return False
    
    def clear(self) -> None:
        """Remove all templates from registry"""
        self._templates.clear()


# --- Built-in Templates ---

class BuiltinTemplates:
    """Collection of built-in prompt templates"""
    
    @staticmethod
    def get_all() -> List[PromptTemplate]:
        """Get all built-in templates"""
        return [
            # Summarization template
            PromptTemplate(
                name="summarize",
                template="请对以下内容进行简明扼要的总结：\n\n{{text}}",
                description="对文本进行摘要",
                system_prompt="你是一个专业的内容摘要助手。请用简洁的语言总结用户提供的内容，保留关键信息。",
            ),
            
            # Translation template
            PromptTemplate(
                name="translate",
                template="请将以下{{source_language|中文}}文本翻译成{{target_language|英文}}：\n\n{{text}}",
                description="翻译文本",
                system_prompt="你是一个专业的翻译助手。请提供准确、自然的翻译，保持原文的语气和风格。",
            ),
            
            # Q&A template
            PromptTemplate(
                name="qa",
                template="{% if context %}基于以下背景信息：\n\n{{context}}\n\n{% endif %}请回答问题：{{question}}",
                description="问答模板（可选上下文）",
                system_prompt="你是一个知识丰富的助手。请基于提供的信息准确回答用户的问题。如果答案不确定，请明确说明。",
            ),
            
            # Code review template
            PromptTemplate(
                name="code_review",
                template="请对以下{{language|代码}}进行代码审查，指出潜在问题和改进建议：\n\n```{{language}}\n{{code}}\n```",
                description="代码审查",
                system_prompt="你是一个经验丰富的代码审查专家。请从代码质量、性能、安全性和可维护性等方面进行审查。",
            ),
            
            # Code explanation template
            PromptTemplate(
                name="explain_code",
                template="请解释以下{{language|代码}}的功能和工作原理：\n\n```{{language}}\n{{code}}\n```",
                description="代码解释",
                system_prompt="你是一个编程教育专家。请用清晰易懂的语言解释代码，必要时可以分步骤说明。",
            ),
            
            # Rewrite template
            PromptTemplate(
                name="rewrite",
                template="请按照以下要求重写文本：\n\n要求：{{style|更加简洁}}\n\n原文：\n{{text}}",
                description="重写文本",
                system_prompt="你是一个专业的写作助手。请按照用户的要求改写文本，保持原意的同时优化表达。",
            ),
            
            # Grammar correction template
            PromptTemplate(
                name="grammar_fix",
                template="请修正以下文本中的语法和拼写错误，并简要说明修改的内容：\n\n{{text}}",
                description="语法修正",
                system_prompt="你是一个语言专家。请仔细检查并修正文本中的错误，保持原文风格。",
            ),
            
            # Extraction template
            PromptTemplate(
                name="extract",
                template="请从以下文本中提取{{extract_type|关键信息}}：\n\n{{text}}",
                description="信息提取",
                system_prompt="你是一个信息提取专家。请准确提取用户指定的信息，以结构化的方式呈现。",
            ),
            
            # Classification template
            PromptTemplate(
                name="classify",
                template="请将以下内容分类到这些类别之一：{{categories}}\n\n内容：{{text}}",
                description="文本分类",
                system_prompt="你是一个分类专家。请根据内容的主题和特征进行准确分类，并简要说明分类依据。",
            ),
            
            # Sentiment analysis template
            PromptTemplate(
                name="sentiment",
                template="请分析以下文本的情感倾向（正面/负面/中性）并简要说明：\n\n{{text}}",
                description="情感分析",
                system_prompt="你是一个情感分析专家。请准确判断文本表达的情感，并提供分析依据。",
            ),
        ]
    
    @staticmethod
    def register_all(registry: TemplateRegistry) -> None:
        """Register all built-in templates to a registry"""
        for template in BuiltinTemplates.get_all():
            registry.register(template)


# --- Global registry instance ---

_global_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """
    Get the global template registry.
    
    Returns:
        Global TemplateRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TemplateRegistry()
        BuiltinTemplates.register_all(_global_registry)
    return _global_registry


def render_template(template_name: str, **kwargs) -> str:
    """
    Render a template from the global registry.
    
    Args:
        template_name: Template name
        **kwargs: Variable values
        
    Returns:
        Rendered string
    """
    return get_template_registry().render(template_name, **kwargs)


def render_messages(template_name: str, **kwargs) -> List[Dict[str, str]]:
    """
    Render a template as messages from the global registry.
    
    Args:
        template_name: Template name
        **kwargs: Variable values
        
    Returns:
        List of message dictionaries
    """
    return get_template_registry().render_messages(template_name, **kwargs)
