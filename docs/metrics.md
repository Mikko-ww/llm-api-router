# Performance Metrics and Monitoring

The LLM API Router includes a comprehensive metrics collection system to help you monitor and optimize your LLM API usage. This document explains how to use the metrics features.

## Overview

The metrics system automatically collects:
- **Request latency statistics** (min, max, avg, p50, p95, p99)
- **Token usage** (prompt, completion, total)
- **Success/failure rates** and error types
- **Provider performance comparison**
- **Prometheus-compatible export format**

## Quick Start

Metrics collection is **enabled by default**. No configuration needed:

```python
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="gpt-3.5-turbo",
)

client = Client(config)

# Make requests
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)

# Get aggregated metrics
metrics = client.get_aggregated_metrics()
for m in metrics:
    print(f"Provider: {m.provider}, Success Rate: {m.success_rate:.2%}")
    print(f"Avg Latency: {m.avg_latency_ms:.2f}ms")
```

## Configuration

### Enable/Disable Metrics

```python
# Enable metrics (default)
config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    metrics_enabled=True,
)

# Disable metrics
config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    metrics_enabled=False,
)
```

### Custom Metrics Collector

Use a shared metrics collector across multiple clients:

```python
from llm_api_router.metrics import MetricsCollector

# Create a shared collector
collector = MetricsCollector(enabled=True)

# Use it with multiple clients
openai_config = ProviderConfig(
    provider_type="openai",
    api_key="openai-key",
    metrics_collector=collector,
)

anthropic_config = ProviderConfig(
    provider_type="anthropic",
    api_key="anthropic-key",
    metrics_collector=collector,
)

openai_client = Client(openai_config)
anthropic_client = Client(anthropic_config)

# Both clients will record to the same collector
# This allows cross-provider comparison
```

## Accessing Metrics

### Raw Metrics

Get individual request metrics:

```python
raw_metrics = client.get_metrics()

for metric in raw_metrics:
    print(f"Request ID: {metric.request_id}")
    print(f"Provider: {metric.provider}")
    print(f"Model: {metric.model}")
    print(f"Latency: {metric.latency_ms}ms")
    print(f"Success: {metric.success}")
    print(f"Tokens: {metric.total_tokens}")
```

### Aggregated Metrics

Get aggregated statistics by provider/model:

```python
aggregated = client.get_aggregated_metrics()

for agg in aggregated:
    print(f"\nProvider: {agg.provider} - Model: {agg.model}")
    print(f"Total Requests: {agg.total_requests}")
    print(f"Success Rate: {agg.success_rate:.2%}")
    
    # Latency statistics
    if agg.min_latency_ms is not None:
        print(f"Latency - Min: {agg.min_latency_ms:.2f}ms")
        print(f"Latency - Max: {agg.max_latency_ms:.2f}ms")
        print(f"Latency - Avg: {agg.avg_latency_ms:.2f}ms")
        print(f"Latency - P50: {agg.p50_latency_ms:.2f}ms")
        print(f"Latency - P95: {agg.p95_latency_ms:.2f}ms")
        print(f"Latency - P99: {agg.p99_latency_ms:.2f}ms")
    
    # Token usage
    print(f"Total Tokens: {agg.total_tokens}")
    if agg.avg_prompt_tokens is not None:
        print(f"Avg Prompt Tokens: {agg.avg_prompt_tokens:.1f}")
        print(f"Avg Completion Tokens: {agg.avg_completion_tokens:.1f}")
    
    # Error breakdown
    if agg.error_counts:
        print("Errors:")
        for error_type, count in agg.error_counts.items():
            print(f"  {error_type}: {count}")
```

### Provider Comparison

Compare performance across providers:

```python
comparison = client.compare_providers()

# Results are sorted by success rate (desc) and latency (asc)
for comp in comparison:
    print(f"{comp['provider']} ({comp['model']})")
    print(f"  Success Rate: {comp['success_rate']:.2%}")
    if comp['avg_latency_ms'] is not None:
        print(f"  Avg Latency: {comp['avg_latency_ms']:.2f}ms")
        print(f"  P95 Latency: {comp['p95_latency_ms']:.2f}ms")
    print(f"  Total Tokens: {comp['total_tokens']}")
```

### Filtered Metrics

Get metrics for specific providers or models:

```python
# Filter by provider
openai_metrics = client.get_metrics(provider="openai")

# Filter by model
gpt4_metrics = client.get_metrics(model="gpt-4")

# Filter by both
specific_metrics = client.get_metrics(provider="openai", model="gpt-4")
```

## Prometheus Export

Export metrics in Prometheus text format:

```python
prometheus_text = client.export_metrics_prometheus()
print(prometheus_text)

# Write to file for Prometheus to scrape
with open("/var/metrics/llm_router.prom", "w") as f:
    f.write(prometheus_text)
```

The exported metrics include:

- `llm_router_requests_total` - Total requests (counter)
- `llm_router_requests_success` - Successful requests (counter)
- `llm_router_requests_failed` - Failed requests (counter)
- `llm_router_success_rate` - Success rate 0-1 (gauge)
- `llm_router_latency_ms` - Latency quantiles (summary)
- `llm_router_tokens_total` - Token usage (counter)

All metrics are labeled with:
- `provider` - The LLM provider name
- `model` - The model name

### Prometheus Configuration

Configure Prometheus to scrape your metrics endpoint:

```yaml
scrape_configs:
  - job_name: 'llm_api_router'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## Grafana Dashboard

A pre-built Grafana dashboard is available at `examples/grafana_dashboard.json`.

### Import Dashboard

1. Open Grafana
2. Click "+" → "Import"
3. Upload `examples/grafana_dashboard.json`
4. Select your Prometheus data source
5. Click "Import"

### Dashboard Panels

The dashboard includes:

- **Request Rate** - Requests per second over time
- **Success Rate** - Current success rate gauge
- **Response Latency** - P50, P95, P99 latency percentiles
- **Token Usage Rate** - Token consumption over time
- **Total Requests/Success/Failed** - Cumulative statistics
- **Total Tokens Used** - Cumulative token usage

## Advanced Usage

### Direct Metrics Collector Access

Access the underlying metrics collector:

```python
collector = client.get_metrics_collector()

# Get summary
summary = collector.get_summary()
print(f"Total Requests: {summary['total_requests']}")
print(f"Success Rate: {summary['success_rate']:.2%}")
print(f"Providers: {summary['providers']}")

# Reset metrics
collector.reset()
```

### Custom Recording

Manually record metrics (advanced use case):

```python
from llm_api_router.metrics import get_metrics_collector

collector = get_metrics_collector()

collector.record_request(
    provider="custom",
    model="custom-model",
    latency_ms=125.5,
    success=True,
    status_code=200,
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
)
```

## Metrics Data Structures

### RequestMetrics

Individual request metrics:

```python
@dataclass
class RequestMetrics:
    provider: str
    model: Optional[str]
    timestamp: datetime
    latency_ms: float
    success: bool
    status_code: Optional[int]
    error_type: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    stream: bool
    request_id: Optional[str]
```

### AggregatedMetrics

Aggregated statistics:

```python
@dataclass
class AggregatedMetrics:
    provider: str
    model: Optional[str]
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    min_latency_ms: Optional[float]
    max_latency_ms: Optional[float]
    avg_latency_ms: Optional[float]
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_prompt_tokens: Optional[float]
    avg_completion_tokens: Optional[float]
    error_counts: Dict[str, int]
    first_request_time: Optional[datetime]
    last_request_time: Optional[datetime]
```

## Performance Considerations

### Memory Usage

Metrics are stored in memory. For long-running applications with many requests:

```python
# Reset metrics periodically
collector = client.get_metrics_collector()
if collector:
    collector.reset()
```

Or disable metrics if not needed:

```python
config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    metrics_enabled=False,
)
```

### Thread Safety

The metrics collector is thread-safe and can be used in multi-threaded environments without additional synchronization.

## Examples

See `examples/metrics_example.py` for complete working examples:

```bash
python examples/metrics_example.py
```

## Best Practices

1. **Enable metrics in development and staging** to understand performance characteristics
2. **Use shared collectors** when comparing multiple providers
3. **Export to Prometheus** for production monitoring and alerting
4. **Set up Grafana dashboards** for visual monitoring
5. **Reset metrics periodically** in long-running applications to manage memory
6. **Monitor success rates and latencies** to detect issues early
7. **Compare providers** to optimize cost and performance

## Troubleshooting

### No metrics collected

Check if metrics are enabled:

```python
collector = client.get_metrics_collector()
if collector:
    print(f"Metrics enabled: {collector.enabled}")
else:
    print("Metrics disabled")
```

### Metrics not updating

Ensure requests are actually completing:

```python
try:
    response = client.chat.completions.create(messages=messages)
    print("Request successful")
except Exception as e:
    print(f"Request failed: {e}")

# Check metrics
metrics = client.get_metrics()
print(f"Total metrics: {len(metrics)}")
```

### High memory usage

Reset metrics periodically:

```python
import time

collector = client.get_metrics_collector()

while True:
    # Your application logic
    make_requests()
    
    # Reset every hour
    time.sleep(3600)
    if collector:
        collector.reset()
```

## Related Documentation

- [Error Handling](error-handling.md) - Understanding error types in metrics
- [Logging](logging.md) - Complementary logging system
- [Examples](../examples/) - Working code examples
