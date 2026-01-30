"""
Load Balancer Example for LLM API Router.

Demonstrates load balancing strategies for distributing requests
across multiple provider endpoints with failover support.
"""

import random
import time
from llm_api_router.load_balancer import (
    LoadBalancer,
    Endpoint,
    LoadBalancerConfig,
    create_load_balancer,
)


def simulate_request(endpoint: Endpoint) -> tuple[bool, float]:
    """
    Simulate an API request with random success/failure.
    
    Returns:
        Tuple of (success, latency)
    """
    # Simulate variable latency
    latency = random.uniform(0.1, 0.5)
    time.sleep(0.01)  # Small delay
    
    # Simulate occasional failures (10% chance)
    success = random.random() > 0.1
    return success, latency


def example_round_robin():
    """Demonstrate round-robin load balancing."""
    print("\n" + "=" * 50)
    print("Round Robin Load Balancing")
    print("=" * 50)
    
    endpoints = [
        Endpoint(name="server-1", provider="openai"),
        Endpoint(name="server-2", provider="openai"),
        Endpoint(name="server-3", provider="openai"),
    ]
    
    lb = LoadBalancer(endpoints=endpoints, strategy="round_robin")
    
    print("\nSelecting endpoints in round-robin order:")
    for i in range(6):
        endpoint = lb.get_endpoint()
        print(f"  Request {i+1}: {endpoint.name}")


def example_weighted():
    """Demonstrate weighted load balancing."""
    print("\n" + "=" * 50)
    print("Weighted Load Balancing")
    print("=" * 50)
    
    endpoints = [
        Endpoint(name="primary", provider="openai", weight=5),
        Endpoint(name="secondary", provider="anthropic", weight=2),
        Endpoint(name="backup", provider="gemini", weight=1),
    ]
    
    lb = LoadBalancer(endpoints=endpoints, strategy="weighted")
    
    # Count selections over many requests
    counts = {ep.name: 0 for ep in endpoints}
    for _ in range(100):
        endpoint = lb.get_endpoint()
        counts[endpoint.name] += 1
    
    print("\nSelection distribution (100 requests):")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}% (weight: "
              f"{next(ep.weight for ep in endpoints if ep.name == name)})")


def example_least_latency():
    """Demonstrate least latency load balancing."""
    print("\n" + "=" * 50)
    print("Least Latency Load Balancing")
    print("=" * 50)
    
    endpoints = [
        Endpoint(name="fast-server", provider="openai"),
        Endpoint(name="medium-server", provider="openai"),
        Endpoint(name="slow-server", provider="openai"),
    ]
    
    lb = LoadBalancer(endpoints=endpoints, strategy="least_latency")
    
    # Simulate different latencies for each server
    latencies = {
        "fast-server": 0.1,
        "medium-server": 0.3,
        "slow-server": 0.5,
    }
    
    # Prime the load balancer with latency data
    print("\nPriming with latency data:")
    for ep in endpoints:
        for _ in range(3):
            lb.mark_success(ep, latency=latencies[ep.name])
        print(f"  {ep.name}: {latencies[ep.name]}s avg latency")
    
    # Now fast-server should be selected most often
    counts = {ep.name: 0 for ep in endpoints}
    for _ in range(10):
        endpoint = lb.get_endpoint()
        counts[endpoint.name] += 1
    
    print("\nNext 10 selections (should prefer fast-server):")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} selections")


def example_failover():
    """Demonstrate failover load balancing."""
    print("\n" + "=" * 50)
    print("Failover Load Balancing")
    print("=" * 50)
    
    endpoints = [
        Endpoint(name="primary", provider="openai", priority=0),
        Endpoint(name="secondary", provider="anthropic", priority=1),
        Endpoint(name="backup", provider="gemini", priority=2),
    ]
    
    config = LoadBalancerConfig(
        failure_threshold=2,
        recovery_time=5.0,  # Short for demo
    )
    
    lb = LoadBalancer(
        endpoints=endpoints,
        strategy="failover",
        config=config,
    )
    
    print("\nInitial state - primary should be selected:")
    endpoint = lb.get_endpoint()
    print(f"  Selected: {endpoint.name}")
    
    # Simulate primary failures
    print("\nSimulating primary server failures...")
    lb.mark_failure(lb.endpoints[0])
    lb.mark_failure(lb.endpoints[0])
    
    print("After 2 failures (threshold met):")
    endpoint = lb.get_endpoint()
    print(f"  Selected: {endpoint.name} (should be secondary)")
    
    # Show stats
    print("\nEndpoint stats:")
    for ep in endpoints:
        stats = lb.get_stats(ep.name)
        print(f"  {ep.name}: status={stats['status']}, "
              f"failures={stats['consecutive_failures']}")


def example_with_client():
    """Demonstrate load balancer with retry logic."""
    print("\n" + "=" * 50)
    print("Load Balancer with Retry Logic")
    print("=" * 50)
    
    endpoints = [
        Endpoint(name="primary", provider="openai", priority=0),
        Endpoint(name="backup", provider="anthropic", priority=1),
    ]
    
    config = LoadBalancerConfig(
        failure_threshold=2,
        recovery_time=30.0,
    )
    
    lb = LoadBalancer(
        endpoints=endpoints,
        strategy="failover",
        config=config,
    )
    
    def make_request_with_failover():
        """Make a request with automatic failover."""
        tried = []
        
        while True:
            endpoint = lb.get_endpoint(exclude=tried)
            if not endpoint:
                raise Exception("All endpoints exhausted")
            
            print(f"  Trying {endpoint.name}...")
            tried.append(endpoint.name)
            
            success, latency = simulate_request(endpoint)
            
            if success:
                lb.mark_success(endpoint, latency=latency)
                return f"Success via {endpoint.name}"
            else:
                lb.mark_failure(endpoint)
                print(f"    Failed, trying next...")
    
    print("\nMaking 5 requests with failover:")
    for i in range(5):
        print(f"\nRequest {i+1}:")
        try:
            result = make_request_with_failover()
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Show final stats
    print("\nFinal statistics:")
    stats = lb.get_stats()
    for name, s in stats.items():
        print(f"  {name}: {s['total_requests']} requests, "
              f"{s['success_rate']:.0%} success rate")


def example_dynamic_endpoints():
    """Demonstrate adding/removing endpoints at runtime."""
    print("\n" + "=" * 50)
    print("Dynamic Endpoint Management")
    print("=" * 50)
    
    lb = create_load_balancer(
        endpoints=[
            {"name": "server-1", "provider": "openai"},
        ],
        strategy="round_robin"
    )
    
    print(f"\nInitial endpoints: {[ep.name for ep in lb.endpoints]}")
    
    # Add new endpoint
    lb.add_endpoint(Endpoint(name="server-2", provider="anthropic"))
    print(f"After adding server-2: {[ep.name for ep in lb.endpoints]}")
    
    # Remove endpoint
    lb.remove_endpoint("server-1")
    print(f"After removing server-1: {[ep.name for ep in lb.endpoints]}")
    
    # Change strategy
    lb.set_strategy("weighted")
    print(f"Changed strategy to: {lb.strategy}")


def example_health_monitoring():
    """Demonstrate health monitoring and recovery."""
    print("\n" + "=" * 50)
    print("Health Monitoring and Recovery")
    print("=" * 50)
    
    endpoints = [
        Endpoint(name="server-1"),
        Endpoint(name="server-2"),
    ]
    
    config = LoadBalancerConfig(
        failure_threshold=3,
        recovery_time=2.0,  # Short for demo
        latency_window=5,
    )
    
    lb = LoadBalancer(
        endpoints=endpoints,
        strategy="round_robin",
        config=config,
    )
    
    # Simulate mixed results
    print("\nSimulating requests...")
    for i in range(10):
        endpoint = lb.get_endpoint()
        if endpoint:
            if random.random() > 0.3:  # 70% success
                latency = random.uniform(0.1, 0.3)
                lb.mark_success(endpoint, latency=latency)
                print(f"  {endpoint.name}: success ({latency:.3f}s)")
            else:
                lb.mark_failure(endpoint)
                print(f"  {endpoint.name}: failed")
        else:
            print("  No healthy endpoints!")
    
    # Show health status
    print("\nHealth status:")
    for ep in endpoints:
        stats = lb.get_stats(ep.name)
        print(f"  {ep.name}:")
        print(f"    Status: {stats['status']}")
        print(f"    Success rate: {stats['success_rate']:.0%}")
        print(f"    Avg latency: {stats['avg_latency']:.3f}s")
    
    healthy = lb.get_healthy_endpoints()
    print(f"\nHealthy endpoints: {[ep.name for ep in healthy]}")


def main():
    """Run all examples."""
    print("LLM API Router - Load Balancer Examples")
    print("=" * 50)
    
    example_round_robin()
    example_weighted()
    example_least_latency()
    example_failover()
    example_with_client()
    example_dynamic_endpoints()
    example_health_monitoring()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
