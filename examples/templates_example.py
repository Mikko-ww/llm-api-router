"""
Prompt Template Example

This example demonstrates various ways to use the prompt template system
in LLM API Router.
"""

from llm_api_router.templates import (
    PromptTemplate,
    TemplateEngine,
    TemplateRegistry,
    BuiltinTemplates,
    get_template_registry,
    render_template,
    render_messages,
)


def basic_template_usage():
    """Basic template creation and rendering"""
    print("=== Basic Template Usage ===\n")
    
    # Create a simple template
    template = PromptTemplate(
        name="greeting",
        template="Hello, {{name}}! Welcome to {{place}}."
    )
    
    print(f"Template: {template.template}")
    print(f"Variables: {template.variables}")
    
    # Render using engine
    engine = TemplateEngine()
    result = engine.render(template.template, {
        "name": "Alice",
        "place": "the LLM API Router"
    })
    print(f"Rendered: {result}")
    print()


def template_with_defaults():
    """Templates with default values"""
    print("=== Templates with Default Values ===\n")
    
    # In-template defaults using |default syntax
    engine = TemplateEngine()
    result = engine.render(
        "Language: {{lang|Python}}, Framework: {{framework|FastAPI}}",
        {"lang": "TypeScript"}  # Only override lang
    )
    print(f"With partial override: {result}")
    
    # Default values in PromptTemplate
    registry = TemplateRegistry()
    registry.register(PromptTemplate(
        name="code_gen",
        template="Generate {{language}} code for: {{task}}",
        default_values={"language": "Python"}
    ))
    
    result = registry.render("code_gen", task="a REST API")
    print(f"With default language: {result}")
    print()


def conditional_templates():
    """Templates with conditional blocks"""
    print("=== Conditional Templates ===\n")
    
    engine = TemplateEngine()
    
    template = """
User: {{username}}
{% if is_admin %}Role: Administrator{% else %}Role: User{% endif %}
{% if email %}Email: {{email}}{% endif %}
""".strip()
    
    # Render for admin
    result1 = engine.render(template, {
        "username": "alice",
        "is_admin": True,
        "email": "alice@example.com"
    })
    print("Admin user:")
    print(result1)
    print()
    
    # Render for regular user without email
    result2 = engine.render(template, {
        "username": "bob",
        "is_admin": False,
        "email": ""
    })
    print("Regular user:")
    print(result2)
    print()


def loop_templates():
    """Templates with loop blocks"""
    print("=== Loop Templates ===\n")
    
    engine = TemplateEngine()
    
    template = """Requirements:
{% for item in requirements %}- {{item}}
{% endfor %}"""
    
    result = engine.render(template, {
        "requirements": [
            "Python 3.10+",
            "httpx library",
            "API key from provider"
        ]
    })
    print(result)
    print()


def filter_usage():
    """Using filters in templates"""
    print("=== Filter Usage ===\n")
    
    engine = TemplateEngine()
    
    # Built-in filters
    print("Built-in filters:")
    print(engine.render("Upper: {{name|upper:}}", {"name": "hello"}))
    print(engine.render("Lower: {{name|lower:}}", {"name": "HELLO"}))
    print(engine.render("Title: {{name|title:}}", {"name": "hello world"}))
    
    # Custom filter
    engine.register_filter("truncate", lambda s: s[:10] + "..." if len(s) > 10 else s)
    print(engine.render(
        "Truncated: {{text|truncate:}}",
        {"text": "This is a very long text that should be truncated"}
    ))
    print()


def using_registry():
    """Using template registry"""
    print("=== Using Template Registry ===\n")
    
    registry = TemplateRegistry()
    
    # Register templates
    registry.register(PromptTemplate(
        name="summarize",
        template="Please summarize the following text:\n\n{{text}}",
        system_prompt="You are a concise summarization expert."
    ))
    
    registry.register(PromptTemplate(
        name="translate",
        template="Translate from {{source}} to {{target}}:\n\n{{text}}",
        system_prompt="You are a professional translator."
    ))
    
    # List templates
    print(f"Registered templates: {registry.list_templates()}")
    
    # Render as string
    result = registry.render("summarize", text="Long text here...")
    print(f"\nRendered summarize template:\n{result}")
    
    # Render as messages (for LLM API)
    messages = registry.render_messages("translate",
        source="English",
        target="Chinese",
        text="Hello, World!"
    )
    print(f"\nRendered as messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    print()


def builtin_templates():
    """Using built-in templates"""
    print("=== Built-in Templates ===\n")
    
    # Get global registry with built-in templates
    registry = get_template_registry()
    
    print(f"Available built-in templates: {registry.list_templates()}")
    print()
    
    # Use summarize template
    result = render_template("summarize", 
        text="LLM API Router is a unified interface for various LLM providers..."
    )
    print("Summarize template:")
    print(result)
    print()
    
    # Use translate template
    result = render_template("translate",
        source_language="English",
        target_language="Japanese",
        text="Hello, World!"
    )
    print("Translate template:")
    print(result)
    print()
    
    # Use code_review template
    messages = render_messages("code_review",
        language="python",
        code="""
def add(a, b):
    return a + b
"""
    )
    print("Code review messages:")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content'][:80]}...")
    print()


def custom_template_workflow():
    """Complete workflow with custom templates"""
    print("=== Custom Template Workflow ===\n")
    
    # Create a registry for a specific use case
    qa_registry = TemplateRegistry()
    
    # Register a Q&A template with context
    qa_registry.register(PromptTemplate(
        name="qa_with_context",
        template="""Based on the following context:

{{context}}

Question: {{question}}

Please provide a detailed answer.""",
        system_prompt="You are a knowledgeable assistant. Answer questions accurately based on the provided context.",
        description="Q&A with context injection"
    ))
    
    # Register a follow-up template
    qa_registry.register(PromptTemplate(
        name="follow_up",
        template="""Previous Q&A:
Q: {{previous_question}}
A: {{previous_answer}}

Follow-up question: {{question}}""",
        system_prompt="Continue the conversation naturally, referencing previous context when relevant."
    ))
    
    # Use the templates
    context = """
The Python programming language was created by Guido van Rossum and first 
released in 1991. Python emphasizes code readability and simplicity.
"""
    
    messages = qa_registry.render_messages("qa_with_context",
        context=context,
        question="Who created Python and when?"
    )
    
    print("Q&A Messages:")
    for msg in messages:
        print(f"[{msg['role']}]")
        print(msg['content'][:200])
        print()


def main():
    """Run all examples"""
    basic_template_usage()
    template_with_defaults()
    conditional_templates()
    loop_templates()
    filter_usage()
    using_registry()
    builtin_templates()
    custom_template_workflow()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()
