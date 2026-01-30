"""
LLM API Router CLI - Main Entry Point.

Provides commands for testing providers, validating configurations,
running benchmarks, and interactive chat.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
except ImportError:
    print("CLI dependencies not installed. Run: pip install llm-api-router[cli]")
    sys.exit(1)


app = typer.Typer(
    name="llm-router",
    help="LLM API Router - Unified interface for multiple LLM providers",
    add_completion=True,
)

console = Console()


def get_version() -> str:
    """Get package version."""
    try:
        from llm_api_router import __version__
        return __version__
    except ImportError:
        return "unknown"


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"[bold blue]llm-api-router[/bold blue] version {get_version()}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """LLM API Router CLI - Unified interface for multiple LLM providers."""
    pass


@app.command()
def test(
    provider: str = typer.Argument(..., help="Provider to test (openai, anthropic, gemini, etc.)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key (or use env var)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-u", help="Custom base URL"),
    message: str = typer.Option("Hello! Please respond with 'OK' to confirm the connection.", "--message", "-msg", help="Test message to send"),
    timeout: int = typer.Option(30, "--timeout", "-t", help="Request timeout in seconds"),
) -> None:
    """
    Test connectivity to a provider.
    
    Examples:
        llm-router test openai --api-key sk-xxx
        llm-router test ollama --base-url http://localhost:11434
        llm-router test anthropic -m claude-3-haiku-20240307
    """
    from llm_api_router import Client, ProviderConfig
    from llm_api_router.exceptions import LLMRouterError
    
    # Resolve API key
    resolved_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not resolved_key and provider != "ollama":
        console.print(f"[red]Error:[/red] API key required. Use --api-key or set {provider.upper()}_API_KEY")
        raise typer.Exit(1)
    
    # Default models
    default_models = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash",
        "deepseek": "deepseek-chat",
        "zhipu": "glm-4-flash",
        "aliyun": "qwen-turbo",
        "ollama": "llama3.2",
        "openrouter": "openai/gpt-3.5-turbo",
        "xai": "grok-beta",
    }
    
    resolved_model = model or default_models.get(provider, "gpt-3.5-turbo")
    
    config = ProviderConfig(
        provider_type=provider,
        api_key=resolved_key or "not-required",
        default_model=resolved_model,
        base_url=base_url,
    )
    
    console.print(f"\n[bold]Testing {provider}...[/bold]")
    console.print(f"  Model: {resolved_model}")
    if base_url:
        console.print(f"  URL: {base_url}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Connecting...", total=None)
        
        try:
            start_time = time.time()
            
            with Client(config) as client:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": message}]
                )
            
            elapsed = time.time() - start_time
            progress.update(task, description="[green]Connected!")
            
        except LLMRouterError as e:
            progress.update(task, description="[red]Failed!")
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)
        except Exception as e:
            progress.update(task, description="[red]Failed!")
            console.print(f"\n[red]Unexpected error:[/red] {e}")
            raise typer.Exit(1)
    
    # Show results
    console.print("\n[green]✓ Connection successful![/green]\n")
    
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    
    table.add_row("Provider", provider)
    table.add_row("Model", response.model)
    table.add_row("Response Time", f"{elapsed:.2f}s")
    table.add_row("Tokens (prompt)", str(response.usage.prompt_tokens) if response.usage else "N/A")
    table.add_row("Tokens (completion)", str(response.usage.completion_tokens) if response.usage else "N/A")
    
    console.print(table)
    
    # Show response preview
    content = response.choices[0].message.content or ""
    preview = content[:200] + "..." if len(content) > 200 else content
    console.print(f"\n[dim]Response preview:[/dim]\n{preview}")


@app.command()
def validate(
    config_file: Path = typer.Argument(..., help="Path to configuration file (JSON/YAML)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation info"),
) -> None:
    """
    Validate a configuration file.
    
    Examples:
        llm-router validate config.json
        llm-router validate providers.yaml --verbose
    """
    if not config_file.exists():
        console.print(f"[red]Error:[/red] File not found: {config_file}")
        raise typer.Exit(1)
    
    suffix = config_file.suffix.lower()
    
    try:
        if suffix == ".json":
            with open(config_file) as f:
                config_data = json.load(f)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
            except ImportError:
                console.print("[red]Error:[/red] PyYAML not installed. Run: pip install pyyaml")
                raise typer.Exit(1)
        else:
            console.print(f"[red]Error:[/red] Unsupported file format: {suffix}")
            raise typer.Exit(1)
        
        # Validate structure
        errors = []
        warnings = []
        
        if isinstance(config_data, dict):
            # Single provider config
            configs = [config_data]
        elif isinstance(config_data, list):
            configs = config_data
        else:
            console.print("[red]Error:[/red] Invalid config format. Expected dict or list.")
            raise typer.Exit(1)
        
        valid_providers = ["openai", "anthropic", "gemini", "deepseek", "zhipu", 
                         "aliyun", "ollama", "openrouter", "xai"]
        
        for i, cfg in enumerate(configs):
            prefix = f"Config[{i}]" if len(configs) > 1 else "Config"
            
            if "provider_type" not in cfg:
                errors.append(f"{prefix}: Missing required field 'provider_type'")
            elif cfg["provider_type"] not in valid_providers:
                warnings.append(f"{prefix}: Unknown provider '{cfg['provider_type']}'")
            
            if "api_key" not in cfg and cfg.get("provider_type") != "ollama":
                warnings.append(f"{prefix}: No 'api_key' specified (will use env var)")
        
        # Show results
        if errors:
            console.print("\n[red]Validation failed![/red]\n")
            for error in errors:
                console.print(f"  [red]✗[/red] {error}")
            raise typer.Exit(1)
        
        console.print("\n[green]✓ Configuration is valid![/green]\n")
        
        if warnings:
            console.print("[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  [yellow]![/yellow] {warning}")
        
        if verbose:
            console.print("\n[dim]Configuration contents:[/dim]")
            syntax = Syntax(json.dumps(config_data, indent=2), "json", theme="monokai")
            console.print(syntax)
            
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    provider: str = typer.Argument(..., help="Provider to benchmark"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    requests: int = typer.Option(5, "--requests", "-n", help="Number of requests"),
    concurrent: int = typer.Option(1, "--concurrent", "-c", help="Concurrent requests"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
) -> None:
    """
    Run performance benchmark against a provider.
    
    Examples:
        llm-router benchmark openai -n 10
        llm-router benchmark anthropic -n 5 -c 2 -o results.json
    """
    import asyncio
    from llm_api_router import AsyncClient, ProviderConfig
    from llm_api_router.exceptions import LLMRouterError
    
    resolved_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not resolved_key and provider != "ollama":
        console.print(f"[red]Error:[/red] API key required")
        raise typer.Exit(1)
    
    default_models = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash",
        "ollama": "llama3.2",
    }
    resolved_model = model or default_models.get(provider, "gpt-3.5-turbo")
    
    config = ProviderConfig(
        provider_type=provider,
        api_key=resolved_key or "not-required",
        default_model=resolved_model,
    )
    
    console.print(f"\n[bold]Benchmarking {provider}[/bold]")
    console.print(f"  Model: {resolved_model}")
    console.print(f"  Requests: {requests}")
    console.print(f"  Concurrency: {concurrent}\n")
    
    results = {
        "provider": provider,
        "model": resolved_model,
        "total_requests": requests,
        "concurrency": concurrent,
        "latencies": [],
        "errors": 0,
        "total_tokens": 0,
    }
    
    async def run_request(client: AsyncClient, semaphore: asyncio.Semaphore) -> tuple:
        async with semaphore:
            start = time.time()
            try:
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": "Say 'benchmark' in one word."}]
                )
                latency = time.time() - start
                tokens = response.usage.total_tokens if response.usage else 0
                return (latency, tokens, None)
            except Exception as e:
                return (time.time() - start, 0, str(e))
    
    async def run_benchmark():
        semaphore = asyncio.Semaphore(concurrent)
        async with AsyncClient(config) as client:
            tasks = [run_request(client, semaphore) for _ in range(requests)]
            
            with Progress(console=console) as progress:
                task = progress.add_task("Running benchmark...", total=requests)
                
                for coro in asyncio.as_completed(tasks):
                    latency, tokens, error = await coro
                    results["latencies"].append(latency)
                    results["total_tokens"] += tokens
                    if error:
                        results["errors"] += 1
                    progress.advance(task)
    
    try:
        asyncio.run(run_benchmark())
    except LLMRouterError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    
    # Calculate stats
    latencies = results["latencies"]
    results["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0
    results["min_latency"] = min(latencies) if latencies else 0
    results["max_latency"] = max(latencies) if latencies else 0
    results["p50_latency"] = sorted(latencies)[len(latencies) // 2] if latencies else 0
    results["p95_latency"] = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    results["success_rate"] = (requests - results["errors"]) / requests * 100 if requests else 0
    results["requests_per_second"] = requests / sum(latencies) * concurrent if latencies else 0
    
    # Display results
    console.print("\n[bold green]Benchmark Results[/bold green]\n")
    
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Requests", str(requests))
    table.add_row("Successful", str(requests - results["errors"]))
    table.add_row("Failed", str(results["errors"]))
    table.add_row("Success Rate", f"{results['success_rate']:.1f}%")
    table.add_row("", "")
    table.add_row("Avg Latency", f"{results['avg_latency']*1000:.0f}ms")
    table.add_row("Min Latency", f"{results['min_latency']*1000:.0f}ms")
    table.add_row("Max Latency", f"{results['max_latency']*1000:.0f}ms")
    table.add_row("P50 Latency", f"{results['p50_latency']*1000:.0f}ms")
    table.add_row("P95 Latency", f"{results['p95_latency']*1000:.0f}ms")
    table.add_row("", "")
    table.add_row("Total Tokens", str(results["total_tokens"]))
    table.add_row("Req/sec", f"{results['requests_per_second']:.2f}")
    
    console.print(table)
    
    if output:
        # Remove raw latencies for cleaner output
        output_data = {k: v for k, v in results.items() if k != "latencies"}
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[dim]Results saved to {output}[/dim]")


@app.command()
def models(
    provider: str = typer.Argument(..., help="Provider to list models for"),
) -> None:
    """
    List available models for a provider.
    
    Note: This shows commonly used models, not all available models.
    
    Examples:
        llm-router models openai
        llm-router models anthropic
    """
    model_catalog = {
        "openai": [
            ("gpt-4o", "Most capable GPT-4 model, multimodal"),
            ("gpt-4o-mini", "Affordable small model for fast tasks"),
            ("gpt-4-turbo", "GPT-4 Turbo with vision"),
            ("gpt-4", "Original GPT-4 model"),
            ("gpt-3.5-turbo", "Fast and cost-effective"),
            ("text-embedding-3-small", "Embedding model (small)"),
            ("text-embedding-3-large", "Embedding model (large)"),
        ],
        "anthropic": [
            ("claude-3-5-sonnet-20241022", "Most intelligent Claude model"),
            ("claude-3-5-haiku-20241022", "Fast and affordable"),
            ("claude-3-opus-20240229", "Powerful for complex tasks"),
            ("claude-3-sonnet-20240229", "Balance of speed and intelligence"),
            ("claude-3-haiku-20240307", "Fastest Claude model"),
        ],
        "gemini": [
            ("gemini-1.5-pro", "Best performing Gemini model"),
            ("gemini-1.5-flash", "Fast and versatile"),
            ("gemini-1.5-flash-8b", "Smallest, most cost-effective"),
            ("gemini-1.0-pro", "Original Gemini Pro"),
            ("text-embedding-004", "Embedding model"),
        ],
        "deepseek": [
            ("deepseek-chat", "General chat model"),
            ("deepseek-coder", "Code generation specialist"),
        ],
        "zhipu": [
            ("glm-4", "Most capable GLM model"),
            ("glm-4-flash", "Fast and affordable"),
            ("glm-4v", "Vision capable"),
            ("embedding-2", "Embedding model"),
        ],
        "aliyun": [
            ("qwen-max", "Most capable Qwen model"),
            ("qwen-plus", "Enhanced Qwen model"),
            ("qwen-turbo", "Fast and efficient"),
            ("text-embedding-v2", "Embedding model"),
        ],
        "ollama": [
            ("llama3.2", "Latest Llama 3.2 model"),
            ("llama3.1", "Llama 3.1 model"),
            ("mistral", "Mistral 7B"),
            ("codellama", "Code Llama"),
            ("phi3", "Microsoft Phi-3"),
        ],
        "xai": [
            ("grok-beta", "Grok beta model"),
            ("grok-vision-beta", "Grok with vision"),
        ],
    }
    
    if provider not in model_catalog:
        console.print(f"[yellow]No model catalog available for '{provider}'[/yellow]")
        console.print(f"[dim]Supported providers: {', '.join(model_catalog.keys())}[/dim]")
        raise typer.Exit(1)
    
    table = Table(title=f"{provider.title()} Models")
    table.add_column("Model", style="cyan")
    table.add_column("Description")
    
    for model_name, description in model_catalog[provider]:
        table.add_row(model_name, description)
    
    console.print(table)
    console.print(f"\n[dim]Note: This is not a complete list. Check provider docs for all models.[/dim]")


@app.command()
def chat(
    provider: str = typer.Argument(..., help="Provider to chat with"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
) -> None:
    """
    Start an interactive chat session.
    
    Commands during chat:
        /clear  - Clear conversation history
        /model  - Show current model
        /exit   - Exit chat (or Ctrl+C)
    
    Examples:
        llm-router chat openai
        llm-router chat anthropic -m claude-3-sonnet --system "You are a helpful assistant"
    """
    from llm_api_router import Client, ProviderConfig
    from llm_api_router.exceptions import LLMRouterError
    
    resolved_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY", "")
    if not resolved_key and provider != "ollama":
        console.print(f"[red]Error:[/red] API key required")
        raise typer.Exit(1)
    
    default_models = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash",
        "ollama": "llama3.2",
    }
    resolved_model = model or default_models.get(provider, "gpt-3.5-turbo")
    
    config = ProviderConfig(
        provider_type=provider,
        api_key=resolved_key or "not-required",
        default_model=resolved_model,
    )
    
    console.print(Panel.fit(
        f"[bold]LLM Router Chat[/bold]\n"
        f"Provider: {provider} | Model: {resolved_model}\n"
        f"[dim]Type /exit to quit, /clear to reset[/dim]",
        border_style="blue",
    ))
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
        console.print(f"[dim]System: {system[:50]}{'...' if len(system) > 50 else ''}[/dim]\n")
    
    try:
        with Client(config) as client:
            while True:
                try:
                    user_input = console.input("[bold cyan]You:[/bold cyan] ")
                except EOFError:
                    break
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.strip().startswith("/"):
                    cmd = user_input.strip().lower()
                    if cmd in ("/exit", "/quit", "/q"):
                        break
                    elif cmd == "/clear":
                        messages = messages[:1] if system else []
                        console.print("[dim]Conversation cleared.[/dim]\n")
                        continue
                    elif cmd == "/model":
                        console.print(f"[dim]Model: {resolved_model}[/dim]\n")
                        continue
                    else:
                        console.print(f"[yellow]Unknown command: {cmd}[/yellow]\n")
                        continue
                
                messages.append({"role": "user", "content": user_input})
                
                try:
                    console.print("[bold green]Assistant:[/bold green] ", end="")
                    
                    # Stream response
                    response_text = ""
                    stream = client.chat.completions.create(
                        messages=messages,
                        stream=True,
                    )
                    
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            console.print(content, end="")
                            response_text += content
                    
                    console.print("\n")
                    messages.append({"role": "assistant", "content": response_text})
                    
                except LLMRouterError as e:
                    console.print(f"\n[red]Error:[/red] {e}\n")
                    messages.pop()  # Remove failed user message
                    
    except KeyboardInterrupt:
        pass
    
    console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    app()
