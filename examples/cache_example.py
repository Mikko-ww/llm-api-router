"""
Example: Using Response Caching

This example demonstrates how to use the response caching feature
to reduce redundant API calls and improve performance.
"""

from llm_api_router import Client, ProviderConfig
from llm_api_router.cache import CacheConfig
import time


def example_basic_caching():
    """Basic caching example"""
    print("=" * 60)
    print("Example 1: Basic Caching")
    print("=" * 60)
    
    # Configure cache: memory backend, 1-hour TTL, max 1000 items
    cache_config = CacheConfig(
        enabled=True,
        backend="memory",
        ttl=3600,  # 1 hour in seconds
        max_size=1000
    )
    
    # Create client with cache enabled
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        cache_config=cache_config
    )
    
    client = Client(config)
    
    # First request - will hit the API
    print("\nFirst request (API call)...")
    start = time.time()
    response1 = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is Python?"}],
        model="gpt-3.5-turbo"
    )
    elapsed1 = time.time() - start
    print(f"Response: {response1.choices[0].message.content[:100]}...")
    print(f"Time: {elapsed1:.3f}s")
    
    # Second identical request - will hit the cache
    print("\nSecond request (cached)...")
    start = time.time()
    response2 = client.chat.completions.create(
        messages=[{"role": "user", "content": "What is Python?"}],
        model="gpt-3.5-turbo"
    )
    elapsed2 = time.time() - start
    print(f"Response: {response2.choices[0].message.content[:100]}...")
    print(f"Time: {elapsed2:.3f}s")
    print(f"Speed improvement: {elapsed1/elapsed2:.1f}x faster")
    
    # Get cache statistics
    stats = client.get_cache_stats()
    print(f"\nCache stats: {stats}")
    
    client.close()


def example_cache_stats():
    """Example showing cache statistics"""
    print("\n" + "=" * 60)
    print("Example 2: Cache Statistics")
    print("=" * 60)
    
    cache_config = CacheConfig(
        enabled=True,
        backend="memory",
        ttl=3600,
        max_size=100
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        cache_config=cache_config
    )
    
    client = Client(config)
    
    # Make several requests
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Duplicate - will be cached
        "What is Go?",
        "What is Python?",  # Duplicate - will be cached
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nRequest {i}: {question}")
        client.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="gpt-3.5-turbo"
        )
    
    # Display cache statistics
    stats = client.get_cache_stats()
    print("\n" + "-" * 60)
    print("Cache Statistics:")
    print(f"  Backend: {stats['backend']}")
    print(f"  Cached items: {stats['size']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    
    client.close()


def example_redis_cache():
    """Example using Redis cache backend"""
    print("\n" + "=" * 60)
    print("Example 3: Redis Cache Backend")
    print("=" * 60)
    
    # Configure Redis cache
    # Note: Requires 'redis' package: pip install redis
    cache_config = CacheConfig(
        enabled=True,
        backend="redis",
        ttl=3600,
        redis_url="redis://localhost:6379/0",  # Redis connection URL
        redis_prefix="myapp:"  # Prefix for all cache keys
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        cache_config=cache_config
    )
    
    try:
        client = Client(config)
        
        print("\nUsing Redis cache backend...")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}],
            model="gpt-3.5-turbo"
        )
        
        print(f"Response: {response.choices[0].message.content}")
        
        stats = client.get_cache_stats()
        print(f"\nCache stats: {stats}")
        
        client.close()
        
    except ImportError:
        print("\nRedis package not installed!")
        print("Install it with: pip install redis")
    except Exception as e:
        print(f"\nError connecting to Redis: {e}")
        print("Make sure Redis is running on localhost:6379")


def example_custom_ttl():
    """Example with custom TTL for different use cases"""
    print("\n" + "=" * 60)
    print("Example 4: Custom TTL Configuration")
    print("=" * 60)
    
    # Short TTL for frequently changing data
    cache_config = CacheConfig(
        enabled=True,
        backend="memory",
        ttl=300,  # 5 minutes
        max_size=500
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        cache_config=cache_config
    )
    
    client = Client(config)
    
    print("\nCache configured with 5-minute TTL")
    print("Responses will expire after 5 minutes")
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather today?"}],
        model="gpt-3.5-turbo"
    )
    
    print(f"Response: {response.choices[0].message.content[:100]}...")
    
    client.close()


def example_cache_management():
    """Example showing cache management operations"""
    print("\n" + "=" * 60)
    print("Example 5: Cache Management")
    print("=" * 60)
    
    cache_config = CacheConfig(
        enabled=True,
        backend="memory",
        ttl=3600,
        max_size=100
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        cache_config=cache_config
    )
    
    client = Client(config)
    
    # Make some requests
    print("\nMaking requests...")
    for i in range(3):
        client.chat.completions.create(
            messages=[{"role": "user", "content": f"Question {i}"}],
            model="gpt-3.5-turbo"
        )
    
    stats = client.get_cache_stats()
    print(f"\nCache size: {stats['size']} items")
    
    # Clear cache
    print("\nClearing cache...")
    client.clear_cache()
    
    stats = client.get_cache_stats()
    print(f"Cache size after clear: {stats['size']} items")
    
    client.close()


def example_streaming_no_cache():
    """Example showing that streaming is not cached"""
    print("\n" + "=" * 60)
    print("Example 6: Streaming Requests (Not Cached)")
    print("=" * 60)
    
    cache_config = CacheConfig(
        enabled=True,
        backend="memory",
        ttl=3600
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        cache_config=cache_config
    )
    
    client = Client(config)
    
    print("\nNote: Streaming responses are NOT cached")
    print("Each streaming request will hit the API\n")
    
    # Streaming request
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "Count to 5"}],
        model="gpt-3.5-turbo",
        stream=True
    )
    
    print("Streaming response:")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    
    client.close()


if __name__ == "__main__":
    # Note: Replace 'your-api-key-here' with your actual API key
    # or set it via environment variable
    
    print("\nLLM API Router - Response Caching Examples")
    print("=" * 60)
    print("\nThese examples demonstrate caching functionality.")
    print("To run them, you need a valid API key.")
    print("\nExamples:")
    
    # Uncomment to run examples (requires valid API key)
    # example_basic_caching()
    # example_cache_stats()
    # example_redis_cache()
    # example_custom_ttl()
    # example_cache_management()
    # example_streaming_no_cache()
    
    print("\nCache Configuration Options:")
    print("  - enabled: Enable/disable caching (default: False)")
    print("  - backend: 'memory' or 'redis' (default: 'memory')")
    print("  - ttl: Time-to-live in seconds (default: 3600)")
    print("  - max_size: Max cached items for memory backend (default: 1000)")
    print("  - redis_url: Redis connection URL (required for redis backend)")
    print("  - redis_prefix: Prefix for Redis keys (default: 'llm_cache:')")
