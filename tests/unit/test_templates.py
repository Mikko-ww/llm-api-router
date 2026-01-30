"""
Unit tests for templates module.
"""

import pytest
from llm_api_router.templates import (
    PromptTemplate,
    TemplateEngine,
    TemplateRegistry,
    BuiltinTemplates,
    get_template_registry,
    render_template,
    render_messages,
)


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass"""
    
    def test_basic_template(self):
        """Test basic template creation"""
        template = PromptTemplate(
            name="test",
            template="Hello, {{name}}!"
        )
        
        assert template.name == "test"
        assert template.template == "Hello, {{name}}!"
        assert "name" in template.variables
    
    def test_auto_detect_variables(self):
        """Test automatic variable detection"""
        template = PromptTemplate(
            name="test",
            template="{{greeting}}, {{name}}! How is {{topic}}?"
        )
        
        assert set(template.variables) == {"greeting", "name", "topic"}
    
    def test_system_prompt_variables(self):
        """Test variables in system prompt are also detected"""
        template = PromptTemplate(
            name="test",
            template="Question: {{question}}",
            system_prompt="You are an expert in {{domain}}"
        )
        
        assert "question" in template.variables
        assert "domain" in template.variables
    
    def test_explicit_variables(self):
        """Test explicit variable list"""
        template = PromptTemplate(
            name="test",
            template="{{a}} and {{b}}",
            variables=["x", "y", "z"]
        )
        
        assert template.variables == ["x", "y", "z"]
    
    def test_default_values(self):
        """Test default values"""
        template = PromptTemplate(
            name="test",
            template="{{name}}",
            default_values={"name": "World"}
        )
        
        assert template.default_values == {"name": "World"}


class TestTemplateEngine:
    """Tests for TemplateEngine"""
    
    def test_simple_substitution(self):
        """Test simple variable substitution"""
        engine = TemplateEngine()
        result = engine.render("Hello, {{name}}!", {"name": "World"})
        
        assert result == "Hello, World!"
    
    def test_multiple_variables(self):
        """Test multiple variable substitution"""
        engine = TemplateEngine()
        result = engine.render(
            "{{greeting}}, {{name}}!",
            {"greeting": "Hi", "name": "Alice"}
        )
        
        assert result == "Hi, Alice!"
    
    def test_variable_with_spaces(self):
        """Test variable with spaces in braces"""
        engine = TemplateEngine()
        result = engine.render("{{ name }}", {"name": "Test"})
        
        assert result == "Test"
    
    def test_default_value(self):
        """Test default value in template"""
        engine = TemplateEngine()
        result = engine.render("{{name|Default}}", {})
        
        assert result == "Default"
    
    def test_default_value_override(self):
        """Test default value is overridden by provided value"""
        engine = TemplateEngine()
        result = engine.render("{{name|Default}}", {"name": "Provided"})
        
        assert result == "Provided"
    
    def test_filter_upper(self):
        """Test upper filter"""
        engine = TemplateEngine()
        result = engine.render("{{name|upper:}}", {"name": "hello"})
        
        assert result == "HELLO"
    
    def test_filter_lower(self):
        """Test lower filter"""
        engine = TemplateEngine()
        result = engine.render("{{name|lower:}}", {"name": "HELLO"})
        
        assert result == "hello"
    
    def test_custom_filter(self):
        """Test custom filter registration"""
        engine = TemplateEngine()
        engine.register_filter("reverse", lambda s: s[::-1])
        
        result = engine.render("{{text|reverse:}}", {"text": "hello"})
        
        assert result == "olleh"
    
    def test_conditional_if_true(self):
        """Test conditional block when condition is true"""
        engine = TemplateEngine()
        result = engine.render(
            "{% if show %}Visible{% endif %}",
            {"show": True}
        )
        
        assert result == "Visible"
    
    def test_conditional_if_false(self):
        """Test conditional block when condition is false"""
        engine = TemplateEngine()
        result = engine.render(
            "{% if show %}Visible{% endif %}",
            {"show": False}
        )
        
        assert result == ""
    
    def test_conditional_if_else(self):
        """Test if-else conditional"""
        engine = TemplateEngine()
        
        result_true = engine.render(
            "{% if show %}Yes{% else %}No{% endif %}",
            {"show": True}
        )
        result_false = engine.render(
            "{% if show %}Yes{% else %}No{% endif %}",
            {"show": False}
        )
        
        assert result_true == "Yes"
        assert result_false == "No"
    
    def test_loop(self):
        """Test for loop"""
        engine = TemplateEngine()
        result = engine.render(
            "{% for item in items %}{{item}} {% endfor %}",
            {"items": ["a", "b", "c"]}
        )
        
        assert result == "a b c "
    
    def test_loop_empty(self):
        """Test for loop with empty list"""
        engine = TemplateEngine()
        result = engine.render(
            "{% for item in items %}{{item}}{% endfor %}",
            {"items": []}
        )
        
        assert result == ""
    
    def test_strict_mode_raises(self):
        """Test strict mode raises error for missing variables"""
        engine = TemplateEngine()
        
        with pytest.raises(ValueError, match="Missing required variable"):
            engine.render("{{missing}}", {}, strict=True)
    
    def test_non_strict_keeps_placeholder(self):
        """Test non-strict mode keeps placeholder for missing variables"""
        engine = TemplateEngine()
        result = engine.render("{{missing}}", {}, strict=False)
        
        assert result == "{{missing}}"


class TestTemplateRegistry:
    """Tests for TemplateRegistry"""
    
    def test_register_and_get(self):
        """Test template registration and retrieval"""
        registry = TemplateRegistry()
        template = PromptTemplate(name="test", template="Hello")
        
        registry.register(template)
        retrieved = registry.get("test")
        
        assert retrieved is not None
        assert retrieved.name == "test"
    
    def test_register_from_dict(self):
        """Test registration from dictionary"""
        registry = TemplateRegistry()
        
        template = registry.register_from_dict({
            "name": "test",
            "template": "Hello, {{name}}!",
            "description": "A test template"
        })
        
        assert template.name == "test"
        assert registry.get("test") is not None
    
    def test_list_templates(self):
        """Test listing registered templates"""
        registry = TemplateRegistry()
        registry.register(PromptTemplate(name="a", template="A"))
        registry.register(PromptTemplate(name="b", template="B"))
        
        templates = registry.list_templates()
        
        assert set(templates) == {"a", "b"}
    
    def test_render(self):
        """Test template rendering"""
        registry = TemplateRegistry()
        registry.register(PromptTemplate(
            name="greeting",
            template="Hello, {{name}}!"
        ))
        
        result = registry.render("greeting", name="World")
        
        assert result == "Hello, World!"
    
    def test_render_with_default_values(self):
        """Test rendering with default values"""
        registry = TemplateRegistry()
        registry.register(PromptTemplate(
            name="greeting",
            template="Hello, {{name}}!",
            default_values={"name": "Guest"}
        ))
        
        # Without override
        result1 = registry.render("greeting")
        assert result1 == "Hello, Guest!"
        
        # With override
        result2 = registry.render("greeting", name="Alice")
        assert result2 == "Hello, Alice!"
    
    def test_render_not_found(self):
        """Test render raises error for unknown template"""
        registry = TemplateRegistry()
        
        with pytest.raises(KeyError, match="Template not found"):
            registry.render("unknown")
    
    def test_render_messages(self):
        """Test rendering as messages"""
        registry = TemplateRegistry()
        registry.register(PromptTemplate(
            name="qa",
            template="Question: {{question}}",
            system_prompt="You are a helpful assistant."
        ))
        
        messages = registry.render_messages("qa", question="What is AI?")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Question: What is AI?"
    
    def test_render_messages_no_system(self):
        """Test rendering as messages without system prompt"""
        registry = TemplateRegistry()
        registry.register(PromptTemplate(
            name="simple",
            template="Hello, {{name}}!"
        ))
        
        messages = registry.render_messages("simple", name="World")
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
    
    def test_unregister(self):
        """Test template unregistration"""
        registry = TemplateRegistry()
        registry.register(PromptTemplate(name="test", template="Test"))
        
        result = registry.unregister("test")
        
        assert result is True
        assert registry.get("test") is None
    
    def test_unregister_not_found(self):
        """Test unregister returns False for unknown template"""
        registry = TemplateRegistry()
        
        result = registry.unregister("unknown")
        
        assert result is False
    
    def test_clear(self):
        """Test clearing all templates"""
        registry = TemplateRegistry()
        registry.register(PromptTemplate(name="a", template="A"))
        registry.register(PromptTemplate(name="b", template="B"))
        
        registry.clear()
        
        assert len(registry.list_templates()) == 0


class TestBuiltinTemplates:
    """Tests for BuiltinTemplates"""
    
    def test_get_all(self):
        """Test getting all built-in templates"""
        templates = BuiltinTemplates.get_all()
        
        assert len(templates) >= 10  # At least 10 built-in templates
        assert all(isinstance(t, PromptTemplate) for t in templates)
    
    def test_register_all(self):
        """Test registering all built-in templates"""
        registry = TemplateRegistry()
        BuiltinTemplates.register_all(registry)
        
        assert len(registry.list_templates()) >= 10
    
    def test_summarize_template(self):
        """Test summarize template"""
        templates = {t.name: t for t in BuiltinTemplates.get_all()}
        
        assert "summarize" in templates
        assert "text" in templates["summarize"].variables
    
    def test_translate_template(self):
        """Test translate template"""
        templates = {t.name: t for t in BuiltinTemplates.get_all()}
        
        assert "translate" in templates
        assert "text" in templates["translate"].variables


class TestGlobalRegistry:
    """Tests for global registry functions"""
    
    def test_get_template_registry(self):
        """Test getting global registry"""
        registry = get_template_registry()
        
        assert isinstance(registry, TemplateRegistry)
        # Should have built-in templates
        assert len(registry.list_templates()) >= 10
    
    def test_render_template(self):
        """Test global render_template function"""
        result = render_template("summarize", text="Some text to summarize")
        
        assert "Some text to summarize" in result
    
    def test_render_messages(self):
        """Test global render_messages function"""
        messages = render_messages("summarize", text="Some text")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestTemplateEdgeCases:
    """Edge case tests"""
    
    def test_nested_braces(self):
        """Test handling of nested braces"""
        engine = TemplateEngine()
        result = engine.render(
            "JSON: {\"key\": \"{{value}}\"}",
            {"value": "test"}
        )
        
        assert result == 'JSON: {"key": "test"}'
    
    def test_empty_template(self):
        """Test empty template"""
        engine = TemplateEngine()
        result = engine.render("", {})
        
        assert result == ""
    
    def test_no_variables(self):
        """Test template without variables"""
        engine = TemplateEngine()
        result = engine.render("Static text", {})
        
        assert result == "Static text"
    
    def test_multiline_template(self):
        """Test multiline template"""
        engine = TemplateEngine()
        template = """Line 1: {{a}}
Line 2: {{b}}
Line 3: {{c}}"""
        
        result = engine.render(template, {"a": "1", "b": "2", "c": "3"})
        
        expected = """Line 1: 1
Line 2: 2
Line 3: 3"""
        assert result == expected
    
    def test_special_characters_in_value(self):
        """Test special characters in variable values"""
        engine = TemplateEngine()
        result = engine.render(
            "{{text}}",
            {"text": "Special chars: <>&\"'"}
        )
        
        assert result == "Special chars: <>&\"'"
    
    def test_unicode_content(self):
        """Test unicode content"""
        engine = TemplateEngine()
        result = engine.render(
            "{{greeting}}, {{name}}!",
            {"greeting": "你好", "name": "世界"}
        )
        
        assert result == "你好, 世界!"
