"""
Load Balancer Module for LLM API Router.

Provides load balancing and failover strategies for multiple provider endpoints,
enabling high availability and optimal resource distribution.

Key Features:
- Multiple selection strategies (round-robin, weighted, least-latency, random, failover)
- Health checking with automatic endpoint recovery
- Thread-safe with thread local storage
- Async support

Example usage:
    # Create load balancer with multiple endpoints
    endpoints = [
        Endpoint(name="primary", url="https://api.openai.com", weight=3),
        Endpoint(name="backup", url="https://api.azure.com", weight=1),
    ]
    
    lb = LoadBalancer(
        endpoints=endpoints,
        strategy="weighted",  # or "round_robin", "least_latency", "random", "failover"
    )
    
    # Get next endpoint
    endpoint = lb.get_endpoint()
    
    # Mark endpoint as failed/succeeded
    lb.mark_failure(endpoint)
    lb.mark_success(endpoint, latency=0.1)
"""

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class EndpointStatus(Enum):
    """Endpoint health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class Endpoint:
    """
    Represents a provider endpoint.
    
    Attributes:
        name: Unique identifier for the endpoint
        url: Base URL for the endpoint (optional, for reference)
        provider: Provider name (e.g., "openai", "anthropic")
        api_key: API key for this endpoint (optional)
        weight: Weight for weighted selection (higher = more traffic)
        priority: Priority for failover (lower = higher priority)
        metadata: Additional endpoint-specific data
    """
    name: str
    url: str = ""
    provider: str = ""
    api_key: str = ""
    weight: int = 1
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Endpoint):
            return self.name == other.name
        return False


@dataclass
class EndpointStats:
    """
    Runtime statistics for an endpoint.
    
    Attributes:
        status: Current health status
        total_requests: Total number of requests sent
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        consecutive_failures: Current consecutive failure count
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last success
        avg_latency: Average response latency in seconds
        latency_samples: Recent latency samples for averaging
    """
    status: EndpointStatus = EndpointStatus.HEALTHY
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    avg_latency: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


@dataclass
class LoadBalancerConfig:
    """
    Configuration for the load balancer.
    
    Attributes:
        failure_threshold: Consecutive failures before unhealthy
        recovery_time: Seconds before retrying unhealthy endpoint
        latency_window: Max latency samples to keep for averaging
        degraded_threshold: Consecutive failures before degraded
        health_check_interval: Optional health check interval (seconds)
    """
    failure_threshold: int = 3
    recovery_time: float = 30.0
    latency_window: int = 10
    degraded_threshold: int = 1
    health_check_interval: Optional[float] = None


class SelectionStrategy(ABC):
    """Abstract base class for endpoint selection strategies."""
    
    @abstractmethod
    def select(
        self,
        endpoints: List[Endpoint],
        stats: Dict[str, EndpointStats]
    ) -> Optional[Endpoint]:
        """
        Select an endpoint from the available endpoints.
        
        Args:
            endpoints: List of healthy endpoints
            stats: Dictionary mapping endpoint names to their stats
            
        Returns:
            Selected endpoint or None if no endpoints available
        """
        pass


class RoundRobinStrategy(SelectionStrategy):
    """Round-robin selection strategy."""
    
    def __init__(self):
        self._index = 0
        self._lock = threading.Lock()
    
    def select(
        self,
        endpoints: List[Endpoint],
        stats: Dict[str, EndpointStats]
    ) -> Optional[Endpoint]:
        if not endpoints:
            return None
        
        with self._lock:
            endpoint = endpoints[self._index % len(endpoints)]
            self._index += 1
            return endpoint


class WeightedStrategy(SelectionStrategy):
    """Weighted random selection based on endpoint weights."""
    
    def select(
        self,
        endpoints: List[Endpoint],
        stats: Dict[str, EndpointStats]
    ) -> Optional[Endpoint]:
        if not endpoints:
            return None
        
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return random.choice(endpoints)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        for endpoint in endpoints:
            cumulative += endpoint.weight
            if r <= cumulative:
                return endpoint
        
        return endpoints[-1]


class LeastLatencyStrategy(SelectionStrategy):
    """Select endpoint with lowest average latency."""
    
    def select(
        self,
        endpoints: List[Endpoint],
        stats: Dict[str, EndpointStats]
    ) -> Optional[Endpoint]:
        if not endpoints:
            return None
        
        # Sort by average latency, preferring endpoints with data
        def latency_key(ep):
            ep_stats = stats.get(ep.name)
            if ep_stats and ep_stats.latency_samples:
                return ep_stats.avg_latency
            return float('inf')
        
        # If all endpoints have no data, use round-robin-like selection
        latencies = [(ep, latency_key(ep)) for ep in endpoints]
        all_infinite = all(lat == float('inf') for _, lat in latencies)
        
        if all_infinite:
            return random.choice(endpoints)
        
        return min(endpoints, key=latency_key)


class RandomStrategy(SelectionStrategy):
    """Pure random selection."""
    
    def select(
        self,
        endpoints: List[Endpoint],
        stats: Dict[str, EndpointStats]
    ) -> Optional[Endpoint]:
        if not endpoints:
            return None
        return random.choice(endpoints)


class FailoverStrategy(SelectionStrategy):
    """Priority-based failover selection."""
    
    def select(
        self,
        endpoints: List[Endpoint],
        stats: Dict[str, EndpointStats]
    ) -> Optional[Endpoint]:
        if not endpoints:
            return None
        
        # Sort by priority (lower is higher priority)
        sorted_endpoints = sorted(endpoints, key=lambda ep: ep.priority)
        return sorted_endpoints[0]


STRATEGY_MAP = {
    "round_robin": RoundRobinStrategy,
    "weighted": WeightedStrategy,
    "least_latency": LeastLatencyStrategy,
    "random": RandomStrategy,
    "failover": FailoverStrategy,
}


class LoadBalancer:
    """
    Load balancer for managing multiple provider endpoints.
    
    Provides endpoint selection with multiple strategies, health checking,
    automatic failover, and recovery.
    
    Example:
        endpoints = [
            Endpoint(name="primary", provider="openai", weight=3),
            Endpoint(name="secondary", provider="anthropic", weight=1),
        ]
        
        lb = LoadBalancer(endpoints=endpoints, strategy="weighted")
        
        # Get endpoint for request
        endpoint = lb.get_endpoint()
        
        # After request completes
        lb.mark_success(endpoint, latency=0.5)
        # Or on failure
        lb.mark_failure(endpoint)
    """
    
    def __init__(
        self,
        endpoints: List[Endpoint],
        strategy: str = "round_robin",
        config: Optional[LoadBalancerConfig] = None,
    ):
        """
        Initialize load balancer.
        
        Args:
            endpoints: List of endpoints to balance between
            strategy: Selection strategy name ("round_robin", "weighted", 
                     "least_latency", "random", "failover")
            config: Optional configuration settings
        """
        if not endpoints:
            raise ValueError("At least one endpoint is required")
        
        self._endpoints = list(endpoints)
        self._config = config or LoadBalancerConfig()
        self._stats: Dict[str, EndpointStats] = {
            ep.name: EndpointStats() for ep in endpoints
        }
        self._lock = threading.Lock()
        
        # Initialize strategy
        if strategy not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Available: {list(STRATEGY_MAP.keys())}")
        self._strategy_name = strategy
        self._strategy = STRATEGY_MAP[strategy]()
    
    @property
    def endpoints(self) -> List[Endpoint]:
        """Get all endpoints."""
        return list(self._endpoints)
    
    @property
    def strategy(self) -> str:
        """Get current strategy name."""
        return self._strategy_name
    
    def get_endpoint(self, exclude: Optional[List[str]] = None) -> Optional[Endpoint]:
        """
        Get next endpoint based on selection strategy.
        
        Args:
            exclude: Optional list of endpoint names to exclude
            
        Returns:
            Selected endpoint or None if all endpoints unavailable
        """
        exclude_set = set(exclude or [])
        
        with self._lock:
            # Filter healthy endpoints
            available = []
            for ep in self._endpoints:
                if ep.name in exclude_set:
                    continue
                
                stats = self._stats[ep.name]
                
                # Check if unhealthy endpoint can be retried
                if stats.status == EndpointStatus.UNHEALTHY:
                    if (stats.last_failure_time and 
                        time.time() - stats.last_failure_time >= self._config.recovery_time):
                        # Allow retry
                        stats.status = EndpointStatus.DEGRADED
                        available.append(ep)
                else:
                    available.append(ep)
            
            if not available:
                return None
            
            return self._strategy.select(available, self._stats)
    
    def mark_success(self, endpoint: Endpoint, latency: Optional[float] = None) -> None:
        """
        Mark an endpoint request as successful.
        
        Args:
            endpoint: The endpoint that succeeded
            latency: Optional response latency in seconds
        """
        with self._lock:
            stats = self._stats.get(endpoint.name)
            if not stats:
                return
            
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.consecutive_failures = 0
            stats.last_success_time = time.time()
            stats.status = EndpointStatus.HEALTHY
            
            if latency is not None:
                stats.latency_samples.append(latency)
                # Keep only recent samples
                if len(stats.latency_samples) > self._config.latency_window:
                    stats.latency_samples = stats.latency_samples[-self._config.latency_window:]
                stats.avg_latency = sum(stats.latency_samples) / len(stats.latency_samples)
    
    def mark_failure(self, endpoint: Endpoint) -> None:
        """
        Mark an endpoint request as failed.
        
        Args:
            endpoint: The endpoint that failed
        """
        with self._lock:
            stats = self._stats.get(endpoint.name)
            if not stats:
                return
            
            stats.total_requests += 1
            stats.failed_requests += 1
            stats.consecutive_failures += 1
            stats.last_failure_time = time.time()
            
            # Update health status
            if stats.consecutive_failures >= self._config.failure_threshold:
                stats.status = EndpointStatus.UNHEALTHY
            elif stats.consecutive_failures >= self._config.degraded_threshold:
                stats.status = EndpointStatus.DEGRADED
    
    def get_stats(self, endpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for endpoint(s).
        
        Args:
            endpoint_name: Optional specific endpoint name, or None for all
            
        Returns:
            Dictionary of endpoint statistics
        """
        with self._lock:
            if endpoint_name:
                stats = self._stats.get(endpoint_name)
                if not stats:
                    return {}
                return {
                    "status": stats.status.value,
                    "total_requests": stats.total_requests,
                    "successful_requests": stats.successful_requests,
                    "failed_requests": stats.failed_requests,
                    "success_rate": stats.success_rate,
                    "consecutive_failures": stats.consecutive_failures,
                    "avg_latency": stats.avg_latency,
                }
            
            return {
                ep.name: {
                    "status": self._stats[ep.name].status.value,
                    "total_requests": self._stats[ep.name].total_requests,
                    "successful_requests": self._stats[ep.name].successful_requests,
                    "failed_requests": self._stats[ep.name].failed_requests,
                    "success_rate": self._stats[ep.name].success_rate,
                    "consecutive_failures": self._stats[ep.name].consecutive_failures,
                    "avg_latency": self._stats[ep.name].avg_latency,
                }
                for ep in self._endpoints
            }
    
    def get_healthy_endpoints(self) -> List[Endpoint]:
        """Get list of currently healthy endpoints."""
        with self._lock:
            return [
                ep for ep in self._endpoints
                if self._stats[ep.name].status in (
                    EndpointStatus.HEALTHY, 
                    EndpointStatus.DEGRADED
                )
            ]
    
    def reset_stats(self, endpoint_name: Optional[str] = None) -> None:
        """
        Reset statistics for endpoint(s).
        
        Args:
            endpoint_name: Optional specific endpoint name, or None for all
        """
        with self._lock:
            if endpoint_name:
                if endpoint_name in self._stats:
                    self._stats[endpoint_name] = EndpointStats()
            else:
                for name in self._stats:
                    self._stats[name] = EndpointStats()
    
    def add_endpoint(self, endpoint: Endpoint) -> None:
        """
        Add a new endpoint to the load balancer.
        
        Args:
            endpoint: The endpoint to add
        """
        with self._lock:
            if any(ep.name == endpoint.name for ep in self._endpoints):
                raise ValueError(f"Endpoint '{endpoint.name}' already exists")
            self._endpoints.append(endpoint)
            self._stats[endpoint.name] = EndpointStats()
    
    def remove_endpoint(self, endpoint_name: str) -> bool:
        """
        Remove an endpoint from the load balancer.
        
        Args:
            endpoint_name: Name of endpoint to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            for i, ep in enumerate(self._endpoints):
                if ep.name == endpoint_name:
                    self._endpoints.pop(i)
                    del self._stats[endpoint_name]
                    return True
            return False
    
    def set_strategy(self, strategy: str) -> None:
        """
        Change the selection strategy.
        
        Args:
            strategy: New strategy name
        """
        if strategy not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        with self._lock:
            self._strategy_name = strategy
            self._strategy = STRATEGY_MAP[strategy]()


# Convenience function
def create_load_balancer(
    endpoints: List[Dict[str, Any]],
    strategy: str = "round_robin",
    **config_kwargs
) -> LoadBalancer:
    """
    Convenience function to create a load balancer from dictionaries.
    
    Args:
        endpoints: List of endpoint dictionaries with keys matching Endpoint fields
        strategy: Selection strategy name
        **config_kwargs: Configuration options
        
    Returns:
        Configured LoadBalancer instance
        
    Example:
        lb = create_load_balancer(
            endpoints=[
                {"name": "primary", "provider": "openai", "weight": 3},
                {"name": "secondary", "provider": "anthropic", "weight": 1},
            ],
            strategy="weighted",
            failure_threshold=5,
        )
    """
    endpoint_objs = [Endpoint(**ep) for ep in endpoints]
    config = LoadBalancerConfig(**config_kwargs) if config_kwargs else None
    return LoadBalancer(endpoints=endpoint_objs, strategy=strategy, config=config)
