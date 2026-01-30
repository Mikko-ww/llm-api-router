"""
Rate Limiter Example

This example demonstrates various ways to use the rate limiter
in LLM API Router.
"""

import time
import asyncio
from llm_api_router.rate_limiter import (
    RateLimiterConfig,
    RateLimiter,
    RateLimitContext,
)


def basic_rate_limiting():
    """Basic rate limiting example"""
    print("=== Basic Rate Limiting ===\n")
    
    # Create a rate limiter with 5 requests per minute
    config = RateLimiterConfig(
        enabled=True,
        backend="token_bucket",
        requests_per_minute=5,
        burst_size=3  # Allow up to 3 requests immediately
    )
    limiter = RateLimiter(config)
    
    # Make several requests
    for i in range(6):
        allowed, wait_time = limiter.acquire()
        if allowed:
            print(f"Request {i+1}: Allowed (remaining: {limiter.get_remaining()})")
        else:
            print(f"Request {i+1}: Rate limited (wait {wait_time:.2f}s)")
    
    print()


def wait_for_rate_limit():
    """Example of waiting for rate limit to clear"""
    print("=== Wait for Rate Limit ===\n")
    
    config = RateLimiterConfig(
        enabled=True,
        backend="token_bucket",
        requests_per_minute=60,  # 1 per second
        burst_size=2,
        wait_timeout=5.0
    )
    limiter = RateLimiter(config)
    
    # Make requests that will require waiting
    for i in range(4):
        start = time.time()
        acquired = limiter.wait_and_acquire()
        elapsed = time.time() - start
        if acquired:
            print(f"Request {i+1}: Acquired after {elapsed:.3f}s")
        else:
            print(f"Request {i+1}: Timed out")
    
    print()


def sliding_window_example():
    """Example using sliding window algorithm"""
    print("=== Sliding Window Rate Limiting ===\n")
    
    config = RateLimiterConfig(
        enabled=True,
        backend="sliding_window",
        requests_per_minute=3  # 3 requests per minute
    )
    limiter = RateLimiter(config)
    
    # Make requests
    for i in range(5):
        allowed, wait_time = limiter.acquire()
        status = "Allowed" if allowed else f"Rejected (wait {wait_time:.1f}s)"
        print(f"Request {i+1}: {status}")
    
    # Show stats
    stats = limiter.get_stats()
    print(f"\nStatistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Rejected: {stats['rejected_requests']}")
    print(f"  Rejection rate: {stats['rejection_rate']:.1%}")
    print()


def context_manager_example():
    """Example using context manager"""
    print("=== Context Manager Usage ===\n")
    
    config = RateLimiterConfig(
        enabled=True,
        backend="token_bucket",
        requests_per_minute=60,
        burst_size=3
    )
    limiter = RateLimiter(config)
    
    # Using context manager
    for i in range(5):
        with RateLimitContext(limiter, key="api_calls", wait=False) as acquired:
            if acquired:
                print(f"Request {i+1}: Making API call...")
                # Your API call here
            else:
                print(f"Request {i+1}: Rate limited, skipping")
    
    print()


def multiple_keys_example():
    """Example with different rate limits for different providers"""
    print("=== Multiple Keys (Provider-based) ===\n")
    
    config = RateLimiterConfig(
        enabled=True,
        backend="token_bucket",
        requests_per_minute=60,
        burst_size=5
    )
    limiter = RateLimiter(config)
    
    # Different providers can have separate rate limits
    providers = ["openai", "anthropic", "gemini"]
    
    for provider in providers:
        key = f"provider:{provider}"
        for i in range(3):
            allowed, _ = limiter.acquire(key=key)
            status = "✓" if allowed else "✗"
            print(f"  {provider} request {i+1}: {status}")
    
    print()


async def async_rate_limiting():
    """Async rate limiting example"""
    print("=== Async Rate Limiting ===\n")
    
    config = RateLimiterConfig(
        enabled=True,
        backend="token_bucket",
        requests_per_minute=60,
        burst_size=3
    )
    limiter = RateLimiter(config)
    
    async def make_request(request_id: int):
        acquired = await limiter.wait_and_acquire_async(timeout=2.0)
        if acquired:
            print(f"Request {request_id}: Processing...")
            await asyncio.sleep(0.1)  # Simulate API call
            print(f"Request {request_id}: Done")
        else:
            print(f"Request {request_id}: Timed out")
    
    # Make concurrent requests
    tasks = [make_request(i) for i in range(5)]
    await asyncio.gather(*tasks)
    print()


async def async_context_manager_example():
    """Async context manager example"""
    print("=== Async Context Manager ===\n")
    
    config = RateLimiterConfig(
        enabled=True,
        backend="token_bucket",
        requests_per_minute=60,
        burst_size=3
    )
    limiter = RateLimiter(config)
    
    async def api_call(request_id: int):
        async with RateLimitContext(limiter, timeout=1.0) as acquired:
            if acquired:
                print(f"Request {request_id}: Executing...")
                await asyncio.sleep(0.1)
            else:
                print(f"Request {request_id}: Rate limited")
    
    await asyncio.gather(*[api_call(i) for i in range(5)])
    print()


def main():
    """Run all examples"""
    basic_rate_limiting()
    wait_for_rate_limit()
    sliding_window_example()
    context_manager_example()
    multiple_keys_example()
    
    # Run async examples
    print("Running async examples...\n")
    asyncio.run(async_rate_limiting())
    asyncio.run(async_context_manager_example())
    
    print("All examples completed!")


if __name__ == "__main__":
    main()
