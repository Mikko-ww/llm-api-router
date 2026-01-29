"""
Example: Using Performance Metrics Collection

This example demonstrates how to use the metrics collection system
to track performance, analyze provider behavior, and export metrics
to Prometheus.
"""

import time
from llm_api_router import Client, ProviderConfig
from llm_api_router.metrics import get_metrics_collector


def main():
    # Create a client with metrics enabled (enabled by default)
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        metrics_enabled=True,  # This is the default
    )
    
    client = Client(config)
    
    # Make some requests
    print("Making sample requests...")
    
    messages = [
        {"role": "user", "content": "What is Python?"}
    ]
    
    try:
        # Successful request
        response = client.chat.completions.create(
            messages=messages,
            temperature=0.7,
        )
        print(f"Response: {response.choices[0].message.content[:50]}...")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Simulate multiple requests
    for i in range(3):
        try:
            messages = [{"role": "user", "content": f"Count to {i+1}"}]
            response = client.chat.completions.create(messages=messages)
            time.sleep(0.5)  # Add some delay between requests
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    # Get metrics from the client
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    
    # Get aggregated metrics
    aggregated = client.get_aggregated_metrics()
    
    for agg in aggregated:
        print(f"\nProvider: {agg.provider}")
        print(f"Model: {agg.model}")
        print(f"Total Requests: {agg.total_requests}")
        print(f"Successful: {agg.successful_requests}")
        print(f"Failed: {agg.failed_requests}")
        print(f"Success Rate: {agg.success_rate:.2%}")
        print(f"\nLatency Statistics:")
        if agg.min_latency_ms is not None:
            print(f"  Min: {agg.min_latency_ms:.2f}ms")
            print(f"  Max: {agg.max_latency_ms:.2f}ms")
            print(f"  Avg: {agg.avg_latency_ms:.2f}ms")
            print(f"  P50: {agg.p50_latency_ms:.2f}ms")
            print(f"  P95: {agg.p95_latency_ms:.2f}ms")
            print(f"  P99: {agg.p99_latency_ms:.2f}ms")
        else:
            print("  No latency data available")
        print(f"\nToken Usage:")
        print(f"  Total Prompt Tokens: {agg.total_prompt_tokens}")
        print(f"  Total Completion Tokens: {agg.total_completion_tokens}")
        print(f"  Total Tokens: {agg.total_tokens}")
        if agg.avg_prompt_tokens is not None:
            print(f"  Avg Prompt Tokens: {agg.avg_prompt_tokens:.1f}")
            print(f"  Avg Completion Tokens: {agg.avg_completion_tokens:.1f}")
        
        if agg.error_counts:
            print(f"\nError Breakdown:")
            for error_type, count in agg.error_counts.items():
                print(f"  {error_type}: {count}")
    
    # Get raw metrics
    print("\n" + "="*60)
    print("RAW METRICS (last 5)")
    print("="*60)
    
    raw_metrics = client.get_metrics()
    for metric in raw_metrics[-5:]:
        print(f"\nRequest ID: {metric.request_id or 'N/A'}")
        print(f"  Provider: {metric.provider}")
        print(f"  Model: {metric.model}")
        print(f"  Latency: {metric.latency_ms:.2f}ms")
        print(f"  Success: {metric.success}")
        print(f"  Tokens: {metric.total_tokens or 'N/A'}")
    
    # Export to Prometheus format
    print("\n" + "="*60)
    print("PROMETHEUS EXPORT")
    print("="*60)
    
    prometheus_text = client.export_metrics_prometheus()
    print(prometheus_text)
    
    # You can write this to a file for Prometheus to scrape
    with open("/tmp/metrics.prom", "w") as f:
        f.write(prometheus_text)
    print("Metrics written to /tmp/metrics.prom")
    
    # Compare providers (useful when using multiple providers)
    print("\n" + "="*60)
    print("PROVIDER COMPARISON")
    print("="*60)
    
    comparison = client.compare_providers()
    for comp in comparison:
        print(f"\nProvider: {comp['provider']}")
        print(f"  Model: {comp['model']}")
        print(f"  Total Requests: {comp['total_requests']}")
        print(f"  Success Rate: {comp['success_rate']:.2%}")
        if comp['avg_latency_ms'] is not None:
            print(f"  Avg Latency: {comp['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {comp['p95_latency_ms']:.2f}ms")
        print(f"  Total Tokens: {comp['total_tokens']}")
    
    client.close()


def example_with_multiple_providers():
    """Example showing metrics with multiple providers"""
    from llm_api_router.metrics import MetricsCollector
    
    # Create a shared metrics collector
    shared_metrics = MetricsCollector(enabled=True)
    
    # Create clients for different providers
    openai_config = ProviderConfig(
        provider_type="openai",
        api_key="your-openai-key",
        default_model="gpt-3.5-turbo",
        metrics_collector=shared_metrics,
    )
    
    anthropic_config = ProviderConfig(
        provider_type="anthropic",
        api_key="your-anthropic-key",
        default_model="claude-3-sonnet",
        metrics_collector=shared_metrics,
    )
    
    openai_client = Client(openai_config)
    anthropic_client = Client(anthropic_config)
    
    # Make requests with both clients
    messages = [{"role": "user", "content": "Hello!"}]
    
    try:
        openai_client.chat.completions.create(messages=messages)
    except Exception as e:
        print(f"OpenAI request failed: {e}")
    
    try:
        anthropic_client.chat.completions.create(messages=messages)
    except Exception as e:
        print(f"Anthropic request failed: {e}")
    
    # Compare performance
    comparison = shared_metrics.compare_providers()
    
    print("\n" + "="*60)
    print("CROSS-PROVIDER COMPARISON")
    print("="*60)
    
    for comp in comparison:
        print(f"\nProvider: {comp['provider']} ({comp['model']})")
        print(f"  Success Rate: {comp['success_rate']:.2%}")
        if comp['avg_latency_ms'] is not None:
            print(f"  Avg Latency: {comp['avg_latency_ms']:.2f}ms")
        print(f"  Total Tokens: {comp['total_tokens']}")
    
    openai_client.close()
    anthropic_client.close()


def example_with_disabled_metrics():
    """Example showing how to disable metrics"""
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        metrics_enabled=False,  # Disable metrics collection
    )
    
    client = Client(config)
    
    # Make a request
    messages = [{"role": "user", "content": "Hello!"}]
    try:
        response = client.chat.completions.create(messages=messages)
        print("Request successful")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # No metrics will be collected
    metrics = client.get_metrics()
    print(f"Collected metrics: {len(metrics)} (should be 0)")
    
    client.close()


if __name__ == "__main__":
    print("="*60)
    print("BASIC METRICS EXAMPLE")
    print("="*60)
    main()
    
    print("\n\n")
    print("="*60)
    print("MULTIPLE PROVIDERS EXAMPLE")
    print("="*60)
    example_with_multiple_providers()
    
    print("\n\n")
    print("="*60)
    print("DISABLED METRICS EXAMPLE")
    print("="*60)
    example_with_disabled_metrics()
