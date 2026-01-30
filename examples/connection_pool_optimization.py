"""
Connection Pool Optimization Examples

This example demonstrates how to configure HTTP connection pool settings
for optimal performance in different scenarios.
"""

from llm_api_router import (
    Client, AsyncClient, ProviderConfig,
    ConnectionPoolConfig, TimeoutConfig
)
import asyncio
import time


def example_default_configuration():
    """Example 1: Using default connection pool configuration"""
    print("\n=== Example 1: Default Configuration ===")
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo"
    )
    
    # Default settings:
    # - max_connections: 100
    # - max_keepalive_connections: 20
    # - keepalive_expiry: 300 seconds (5 minutes)
    # - timeout: 60 seconds
    
    with Client(config) as client:
        print("Client created with default connection pool settings")
        print("- Max connections: 100")
        print("- Max keepalive connections: 20")
        print("- Keepalive expiry: 300s (5 minutes)")


def example_high_concurrency_configuration():
    """Example 2: Configuration for high concurrency scenarios"""
    print("\n=== Example 2: High Concurrency Configuration ===")
    
    # Optimize for high concurrency workloads
    pool_config = ConnectionPoolConfig(
        max_connections=200,  # Higher connection limit
        max_keepalive_connections=50,  # More persistent connections
        keepalive_expiry=600.0,  # Keep connections alive longer (10 minutes)
        stream_buffer_size=131072  # Larger buffer for streaming (128KB)
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        connection_pool_config=pool_config
    )
    
    with Client(config) as client:
        print("Client configured for high concurrency:")
        print(f"- Max connections: {pool_config.max_connections}")
        print(f"- Max keepalive connections: {pool_config.max_keepalive_connections}")
        print(f"- Keepalive expiry: {pool_config.keepalive_expiry}s")
        print(f"- Stream buffer size: {pool_config.stream_buffer_size} bytes")


def example_low_latency_configuration():
    """Example 3: Configuration for low latency requirements"""
    print("\n=== Example 3: Low Latency Configuration ===")
    
    # Optimize for low latency
    timeout_config = TimeoutConfig(
        connect=5.0,  # Quick connection timeout
        read=30.0,  # Reasonable read timeout
        write=5.0,  # Quick write timeout
        pool=5.0  # Fast pool acquisition
    )
    
    pool_config = ConnectionPoolConfig(
        max_connections=50,
        max_keepalive_connections=20,
        keepalive_expiry=120.0  # Shorter expiry to avoid stale connections
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        timeout_config=timeout_config,
        connection_pool_config=pool_config
    )
    
    with Client(config) as client:
        print("Client configured for low latency:")
        print(f"- Connect timeout: {timeout_config.connect}s")
        print(f"- Read timeout: {timeout_config.read}s")
        print(f"- Write timeout: {timeout_config.write}s")
        print(f"- Pool timeout: {timeout_config.pool}s")


def example_resource_constrained_configuration():
    """Example 4: Configuration for resource-constrained environments"""
    print("\n=== Example 4: Resource-Constrained Configuration ===")
    
    # Minimize resource usage
    pool_config = ConnectionPoolConfig(
        max_connections=20,  # Lower connection limit
        max_keepalive_connections=5,  # Fewer persistent connections
        keepalive_expiry=60.0,  # Shorter expiry (1 minute)
        stream_buffer_size=32768  # Smaller buffer (32KB)
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        connection_pool_config=pool_config
    )
    
    with Client(config) as client:
        print("Client configured for resource-constrained environment:")
        print(f"- Max connections: {pool_config.max_connections}")
        print(f"- Max keepalive connections: {pool_config.max_keepalive_connections}")
        print(f"- Keepalive expiry: {pool_config.keepalive_expiry}s")
        print(f"- Stream buffer size: {pool_config.stream_buffer_size} bytes")


async def example_async_client_configuration():
    """Example 5: Async client with custom configuration"""
    print("\n=== Example 5: Async Client Configuration ===")
    
    # Configuration optimized for async operations
    timeout_config = TimeoutConfig(
        connect=10.0,
        read=120.0,  # Longer for async operations
        write=10.0,
        pool=10.0
    )
    
    pool_config = ConnectionPoolConfig(
        max_connections=100,
        max_keepalive_connections=30,
        keepalive_expiry=300.0
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        timeout_config=timeout_config,
        connection_pool_config=pool_config
    )
    
    async with AsyncClient(config) as client:
        print("Async client configured with custom settings:")
        print(f"- Max connections: {pool_config.max_connections}")
        print(f"- Read timeout: {timeout_config.read}s")
        print("Ready for concurrent async operations")


def example_backward_compatibility():
    """Example 6: Backward compatibility with simple timeout"""
    print("\n=== Example 6: Backward Compatibility ===")
    
    # Old style configuration still works
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        timeout=30.0  # Simple timeout configuration
    )
    
    with Client(config) as client:
        print("Client created with simple timeout configuration:")
        print("- Timeout: 30s (backward compatible)")
        print("- Connection pool uses default settings")


def example_connection_reuse_demonstration():
    """Example 7: Demonstrating connection reuse benefits"""
    print("\n=== Example 7: Connection Reuse Demonstration ===")
    
    # Configuration that encourages connection reuse
    pool_config = ConnectionPoolConfig(
        max_connections=10,
        max_keepalive_connections=10,  # Keep all connections alive
        keepalive_expiry=3600.0  # Keep connections for 1 hour
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        connection_pool_config=pool_config
    )
    
    with Client(config) as client:
        print("Making multiple requests to demonstrate connection reuse:")
        print("- All connections kept alive for efficient reuse")
        print("- Reduces TCP handshake overhead")
        print("- Improves throughput for repeated requests")
        
        # In real usage, you would make multiple requests here
        # The HTTP client will automatically reuse connections


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("HTTP Connection Pool Optimization Examples")
    print("="*60)
    
    # Run synchronous examples
    example_default_configuration()
    example_high_concurrency_configuration()
    example_low_latency_configuration()
    example_resource_constrained_configuration()
    example_backward_compatibility()
    example_connection_reuse_demonstration()
    
    # Run async example
    asyncio.run(example_async_client_configuration())
    
    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("1. Default settings work well for most use cases")
    print("2. Increase max_connections for high concurrency")
    print("3. Adjust timeouts based on your latency requirements")
    print("4. Keep connections alive longer for better performance")
    print("5. Fine-tune for resource-constrained environments")
    print("6. Backward compatible with simple timeout configuration")
    print("\n")


if __name__ == "__main__":
    main()
