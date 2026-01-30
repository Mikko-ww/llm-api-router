# HTTP Connection Pool Optimization

This document describes the HTTP connection pool optimization features in llm-api-router, including configuration options and best practices.

## Overview

The llm-api-router uses httpx as its HTTP client library, which provides excellent support for connection pooling and reuse. By properly configuring the connection pool, you can significantly improve the performance and resource efficiency of your LLM API interactions.

## Benefits of Connection Pool Optimization

1. **Reduced Latency**: Reusing existing connections eliminates TCP handshake overhead
2. **Better Throughput**: More efficient handling of concurrent requests
3. **Resource Efficiency**: Controlled resource usage through connection limits
4. **Improved Reliability**: Proper timeout configuration prevents hanging requests

## Configuration Classes

### ConnectionPoolConfig

Controls the HTTP connection pool behavior:

```python
from llm_api_router import ConnectionPoolConfig

pool_config = ConnectionPoolConfig(
    max_connections=100,           # Maximum total connections
    max_keepalive_connections=20,  # Maximum idle connections to keep alive
    keepalive_expiry=300.0,        # Seconds to keep idle connections alive
    stream_buffer_size=65536       # Buffer size for streaming responses (bytes)
)
```

**Parameters:**

- `max_connections` (int, default=100): Maximum number of connections that can exist in the pool at once. This limits concurrent requests.
- `max_keepalive_connections` (int, default=20): Maximum number of idle connections to keep alive for reuse. Connections beyond this limit are closed immediately after use.
- `keepalive_expiry` (float, default=300.0): Time in seconds that an idle connection is kept alive before being closed.
- `stream_buffer_size` (int, default=65536): Size of the buffer used for streaming responses in bytes (64KB default).

### TimeoutConfig

Provides fine-grained control over various timeout aspects:

```python
from llm_api_router import TimeoutConfig

timeout_config = TimeoutConfig(
    connect=10.0,  # Connection establishment timeout
    read=60.0,     # Read timeout for response data
    write=10.0,    # Write timeout for request data
    pool=10.0      # Timeout for acquiring a connection from the pool
)
```

**Parameters:**

- `connect` (float, default=10.0): Maximum time in seconds to wait for a connection to be established.
- `read` (float, default=60.0): Maximum time in seconds to wait between consecutive read operations.
- `write` (float, default=10.0): Maximum time in seconds to wait for write operations.
- `pool` (float, default=10.0): Maximum time in seconds to wait for acquiring a connection from the pool.

## Usage

### Basic Configuration

Using default settings (recommended for most use cases):

```python
from llm_api_router import Client, ProviderConfig

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    default_model="gpt-3.5-turbo"
)

client = Client(config)
```

### Custom Connection Pool Configuration

```python
from llm_api_router import Client, ProviderConfig, ConnectionPoolConfig

pool_config = ConnectionPoolConfig(
    max_connections=50,
    max_keepalive_connections=10,
    keepalive_expiry=120.0
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    connection_pool_config=pool_config
)

client = Client(config)
```

### Custom Timeout Configuration

```python
from llm_api_router import Client, ProviderConfig, TimeoutConfig

timeout_config = TimeoutConfig(
    connect=5.0,
    read=30.0,
    write=5.0,
    pool=5.0
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    timeout_config=timeout_config
)

client = Client(config)
```

### Combined Configuration

```python
from llm_api_router import (
    Client, ProviderConfig,
    ConnectionPoolConfig, TimeoutConfig
)

timeout_config = TimeoutConfig(
    connect=5.0,
    read=60.0,
    write=5.0,
    pool=5.0
)

pool_config = ConnectionPoolConfig(
    max_connections=100,
    max_keepalive_connections=30,
    keepalive_expiry=600.0
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    timeout_config=timeout_config,
    connection_pool_config=pool_config
)

client = Client(config)
```

## Configuration Scenarios

### High Concurrency Workloads

When handling many concurrent requests:

```python
pool_config = ConnectionPoolConfig(
    max_connections=200,        # Allow more concurrent connections
    max_keepalive_connections=50,  # Keep more connections alive
    keepalive_expiry=600.0,     # Keep connections alive longer
    stream_buffer_size=131072   # Larger buffer for streaming (128KB)
)
```

### Low Latency Requirements

When minimizing latency is critical:

```python
timeout_config = TimeoutConfig(
    connect=5.0,   # Quick connection timeout
    read=30.0,     # Moderate read timeout
    write=5.0,     # Quick write timeout
    pool=5.0       # Fast pool acquisition
)

pool_config = ConnectionPoolConfig(
    max_connections=50,
    max_keepalive_connections=20,
    keepalive_expiry=120.0  # Shorter expiry to avoid stale connections
)
```

### Resource-Constrained Environments

When running with limited resources:

```python
pool_config = ConnectionPoolConfig(
    max_connections=20,         # Lower connection limit
    max_keepalive_connections=5,   # Fewer persistent connections
    keepalive_expiry=60.0,      # Shorter expiry
    stream_buffer_size=32768    # Smaller buffer (32KB)
)
```

### Long-Running Streaming Applications

When dealing with long streaming responses:

```python
timeout_config = TimeoutConfig(
    connect=10.0,
    read=300.0,    # Longer read timeout for streaming
    write=10.0,
    pool=10.0
)

pool_config = ConnectionPoolConfig(
    max_connections=50,
    max_keepalive_connections=20,
    keepalive_expiry=600.0,
    stream_buffer_size=131072  # Larger buffer for streaming
)
```

## Best Practices

### 1. Connection Pool Sizing

- **max_connections**: Set based on your expected peak concurrent requests. A good starting point is 100.
- **max_keepalive_connections**: Keep this at 10-30% of max_connections for most workloads.
- Rule of thumb: `max_keepalive_connections` ≤ `max_connections`

### 2. Keepalive Expiry

- Default 300s (5 minutes) works well for most cases
- Increase to 600s-3600s for applications with consistent traffic patterns
- Decrease to 60s-120s for sporadic usage to free up resources

### 3. Timeout Configuration

- **connect**: Should be short (5-10s) to fail fast on network issues
- **read**: Should account for typical API response times plus buffer (30-120s)
- **write**: Usually short (5-10s) as request bodies are typically small
- **pool**: Should be short (5-10s) to detect pool exhaustion quickly

### 4. Buffer Sizing

- Default 64KB is suitable for most use cases
- Increase to 128KB-256KB for high-throughput streaming applications
- Decrease to 32KB for memory-constrained environments

### 5. Monitoring and Tuning

Monitor these metrics to tune your configuration:
- Request latency (P50, P95, P99)
- Connection pool exhaustion events
- Timeout errors
- Memory usage

## Backward Compatibility

The simple `timeout` parameter is still supported for backward compatibility:

```python
config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    timeout=30.0  # Simple timeout (seconds)
)
```

When both `timeout` and `timeout_config` are provided, `timeout_config` takes precedence.

## Async Client

All configuration options work identically with AsyncClient:

```python
from llm_api_router import AsyncClient, ProviderConfig, ConnectionPoolConfig

pool_config = ConnectionPoolConfig(
    max_connections=100,
    max_keepalive_connections=30
)

config = ProviderConfig(
    provider_type="openai",
    api_key="your-api-key",
    connection_pool_config=pool_config
)

async with AsyncClient(config) as client:
    # Use async client
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

## Performance Impact

Proper connection pool configuration can provide significant performance improvements:

- **Connection Reuse**: Up to 50% reduction in request latency by eliminating TCP handshake
- **Concurrent Throughput**: 2-5x improvement in handling concurrent requests with proper sizing
- **Resource Efficiency**: 30-50% reduction in memory usage with appropriate keepalive settings

## Troubleshooting

### Pool Exhaustion

If you see timeout errors when acquiring connections:
- Increase `max_connections`
- Check if requests are being properly closed/released
- Verify that `keepalive_expiry` isn't too short

### Memory Issues

If memory usage is too high:
- Decrease `max_keepalive_connections`
- Reduce `keepalive_expiry`
- Lower `stream_buffer_size`

### Connection Timeouts

If you experience frequent connection timeouts:
- Increase `timeout_config.connect`
- Check network stability
- Verify API endpoint availability

### Read Timeouts

If you experience read timeouts during responses:
- Increase `timeout_config.read`
- For streaming, ensure buffer size is adequate
- Check API response times

## Examples

See `examples/connection_pool_optimization.py` for complete working examples demonstrating various configuration scenarios.

## References

- [httpx Documentation](https://www.python-httpx.org/)
- [Connection Pooling Best Practices](https://www.python-httpx.org/advanced/#pool-limit-configuration)
- [Timeout Configuration Guide](https://www.python-httpx.org/advanced/#timeout-configuration)
