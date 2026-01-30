"""Tests for load balancer module."""

import time
import threading
import pytest

from llm_api_router.load_balancer import (
    Endpoint,
    EndpointStats,
    EndpointStatus,
    LoadBalancerConfig,
    LoadBalancer,
    RoundRobinStrategy,
    WeightedStrategy,
    LeastLatencyStrategy,
    RandomStrategy,
    FailoverStrategy,
    create_load_balancer,
)


class TestEndpoint:
    """Tests for Endpoint dataclass."""
    
    def test_default_values(self):
        ep = Endpoint(name="test")
        assert ep.name == "test"
        assert ep.url == ""
        assert ep.provider == ""
        assert ep.weight == 1
        assert ep.priority == 0
        assert ep.metadata == {}
    
    def test_custom_values(self):
        ep = Endpoint(
            name="primary",
            url="https://api.example.com",
            provider="openai",
            weight=5,
            priority=1,
            metadata={"region": "us-east"}
        )
        assert ep.name == "primary"
        assert ep.url == "https://api.example.com"
        assert ep.provider == "openai"
        assert ep.weight == 5
        assert ep.priority == 1
        assert ep.metadata == {"region": "us-east"}
    
    def test_hash_and_equality(self):
        ep1 = Endpoint(name="test", weight=1)
        ep2 = Endpoint(name="test", weight=2)
        ep3 = Endpoint(name="other")
        
        # Same name = equal
        assert ep1 == ep2
        assert hash(ep1) == hash(ep2)
        
        # Different name = not equal
        assert ep1 != ep3


class TestEndpointStats:
    """Tests for EndpointStats dataclass."""
    
    def test_default_values(self):
        stats = EndpointStats()
        assert stats.status == EndpointStatus.HEALTHY
        assert stats.total_requests == 0
        assert stats.consecutive_failures == 0
        assert stats.avg_latency == 0.0
    
    def test_success_rate(self):
        stats = EndpointStats(total_requests=10, successful_requests=8)
        assert stats.success_rate == 0.8
        
        # Zero requests
        stats2 = EndpointStats()
        assert stats2.success_rate == 1.0


class TestLoadBalancerConfig:
    """Tests for LoadBalancerConfig dataclass."""
    
    def test_default_values(self):
        config = LoadBalancerConfig()
        assert config.failure_threshold == 3
        assert config.recovery_time == 30.0
        assert config.latency_window == 10
    
    def test_custom_values(self):
        config = LoadBalancerConfig(
            failure_threshold=5,
            recovery_time=60.0,
            latency_window=20
        )
        assert config.failure_threshold == 5
        assert config.recovery_time == 60.0
        assert config.latency_window == 20


class TestRoundRobinStrategy:
    """Tests for round robin selection strategy."""
    
    def test_cycles_through_endpoints(self):
        strategy = RoundRobinStrategy()
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
            Endpoint(name="c"),
        ]
        stats = {}
        
        # Should cycle through in order
        assert strategy.select(endpoints, stats).name == "a"
        assert strategy.select(endpoints, stats).name == "b"
        assert strategy.select(endpoints, stats).name == "c"
        assert strategy.select(endpoints, stats).name == "a"
    
    def test_empty_endpoints(self):
        strategy = RoundRobinStrategy()
        assert strategy.select([], {}) is None


class TestWeightedStrategy:
    """Tests for weighted selection strategy."""
    
    def test_respects_weights(self):
        strategy = WeightedStrategy()
        endpoints = [
            Endpoint(name="heavy", weight=100),
            Endpoint(name="light", weight=1),
        ]
        stats = {}
        
        # Run many selections, heavy should be selected more often
        selections = [strategy.select(endpoints, stats).name for _ in range(100)]
        heavy_count = selections.count("heavy")
        
        # With 100:1 weight ratio, heavy should be selected most times
        assert heavy_count > 50
    
    def test_zero_weights(self):
        strategy = WeightedStrategy()
        endpoints = [
            Endpoint(name="a", weight=0),
            Endpoint(name="b", weight=0),
        ]
        
        # Should still return something (random choice)
        result = strategy.select(endpoints, {})
        assert result is not None
    
    def test_empty_endpoints(self):
        strategy = WeightedStrategy()
        assert strategy.select([], {}) is None


class TestLeastLatencyStrategy:
    """Tests for least latency selection strategy."""
    
    def test_selects_lowest_latency(self):
        strategy = LeastLatencyStrategy()
        endpoints = [
            Endpoint(name="slow"),
            Endpoint(name="fast"),
            Endpoint(name="medium"),
        ]
        stats = {
            "slow": EndpointStats(avg_latency=1.0, latency_samples=[1.0]),
            "fast": EndpointStats(avg_latency=0.1, latency_samples=[0.1]),
            "medium": EndpointStats(avg_latency=0.5, latency_samples=[0.5]),
        }
        
        result = strategy.select(endpoints, stats)
        assert result.name == "fast"
    
    def test_no_latency_data(self):
        strategy = LeastLatencyStrategy()
        endpoints = [Endpoint(name="a"), Endpoint(name="b")]
        stats = {}
        
        # Should return something when no data
        result = strategy.select(endpoints, stats)
        assert result is not None


class TestRandomStrategy:
    """Tests for random selection strategy."""
    
    def test_returns_endpoint(self):
        strategy = RandomStrategy()
        endpoints = [Endpoint(name="a"), Endpoint(name="b")]
        
        result = strategy.select(endpoints, {})
        assert result in endpoints
    
    def test_empty_endpoints(self):
        strategy = RandomStrategy()
        assert strategy.select([], {}) is None


class TestFailoverStrategy:
    """Tests for failover selection strategy."""
    
    def test_selects_by_priority(self):
        strategy = FailoverStrategy()
        endpoints = [
            Endpoint(name="backup", priority=2),
            Endpoint(name="primary", priority=0),
            Endpoint(name="secondary", priority=1),
        ]
        
        # Should select lowest priority (0)
        result = strategy.select(endpoints, {})
        assert result.name == "primary"
    
    def test_empty_endpoints(self):
        strategy = FailoverStrategy()
        assert strategy.select([], {}) is None


class TestLoadBalancer:
    """Tests for LoadBalancer class."""
    
    def test_initialization(self):
        endpoints = [Endpoint(name="test")]
        lb = LoadBalancer(endpoints=endpoints)
        
        assert lb.strategy == "round_robin"
        assert len(lb.endpoints) == 1
    
    def test_no_endpoints_raises(self):
        with pytest.raises(ValueError, match="At least one endpoint"):
            LoadBalancer(endpoints=[])
    
    def test_unknown_strategy_raises(self):
        endpoints = [Endpoint(name="test")]
        with pytest.raises(ValueError, match="Unknown strategy"):
            LoadBalancer(endpoints=endpoints, strategy="invalid")
    
    def test_get_endpoint_round_robin(self):
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
        ]
        lb = LoadBalancer(endpoints=endpoints, strategy="round_robin")
        
        names = [lb.get_endpoint().name for _ in range(4)]
        assert names == ["a", "b", "a", "b"]
    
    def test_get_endpoint_with_exclude(self):
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
            Endpoint(name="c"),
        ]
        lb = LoadBalancer(endpoints=endpoints, strategy="round_robin")
        
        result = lb.get_endpoint(exclude=["a", "b"])
        assert result.name == "c"
    
    def test_mark_success(self):
        endpoints = [Endpoint(name="test")]
        lb = LoadBalancer(endpoints=endpoints)
        ep = endpoints[0]
        
        lb.mark_success(ep, latency=0.5)
        
        stats = lb.get_stats("test")
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["avg_latency"] == 0.5
        assert stats["status"] == "healthy"
    
    def test_mark_failure(self):
        endpoints = [Endpoint(name="test")]
        config = LoadBalancerConfig(failure_threshold=2)
        lb = LoadBalancer(endpoints=endpoints, config=config)
        ep = endpoints[0]
        
        # First failure -> degraded
        lb.mark_failure(ep)
        stats = lb.get_stats("test")
        assert stats["status"] == "degraded"
        assert stats["consecutive_failures"] == 1
        
        # Second failure -> unhealthy
        lb.mark_failure(ep)
        stats = lb.get_stats("test")
        assert stats["status"] == "unhealthy"
        assert stats["consecutive_failures"] == 2
    
    def test_unhealthy_endpoint_excluded(self):
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
        ]
        config = LoadBalancerConfig(failure_threshold=1, recovery_time=1000)
        lb = LoadBalancer(endpoints=endpoints, config=config)
        
        # Make 'a' unhealthy
        lb.mark_failure(endpoints[0])
        
        # Should only return 'b'
        for _ in range(5):
            result = lb.get_endpoint()
            assert result.name == "b"
    
    def test_unhealthy_endpoint_recovery(self):
        endpoints = [Endpoint(name="test")]
        config = LoadBalancerConfig(failure_threshold=1, recovery_time=0.1)
        lb = LoadBalancer(endpoints=endpoints, config=config)
        
        # Make unhealthy
        lb.mark_failure(endpoints[0])
        assert lb.get_endpoint() is None  # No healthy endpoints
        
        # Wait for recovery
        time.sleep(0.15)
        
        # Should be available again (as degraded)
        result = lb.get_endpoint()
        assert result is not None
        assert result.name == "test"
    
    def test_latency_window(self):
        endpoints = [Endpoint(name="test")]
        config = LoadBalancerConfig(latency_window=3)
        lb = LoadBalancer(endpoints=endpoints, config=config)
        ep = endpoints[0]
        
        # Add more samples than window
        for lat in [1.0, 2.0, 3.0, 4.0, 5.0]:
            lb.mark_success(ep, latency=lat)
        
        stats = lb.get_stats("test")
        # Should only keep last 3: [3.0, 4.0, 5.0]
        assert stats["avg_latency"] == 4.0
    
    def test_get_healthy_endpoints(self):
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
            Endpoint(name="c"),
        ]
        config = LoadBalancerConfig(failure_threshold=1)
        lb = LoadBalancer(endpoints=endpoints, config=config)
        
        # Make 'a' unhealthy
        lb.mark_failure(endpoints[0])
        
        healthy = lb.get_healthy_endpoints()
        names = [ep.name for ep in healthy]
        assert "a" not in names
        assert "b" in names
        assert "c" in names
    
    def test_reset_stats(self):
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
        ]
        lb = LoadBalancer(endpoints=endpoints)
        
        # Add some stats
        lb.mark_success(endpoints[0], latency=0.5)
        lb.mark_failure(endpoints[1])
        
        # Reset specific endpoint
        lb.reset_stats("a")
        assert lb.get_stats("a")["total_requests"] == 0
        assert lb.get_stats("b")["total_requests"] == 1
        
        # Reset all
        lb.reset_stats()
        assert lb.get_stats("b")["total_requests"] == 0
    
    def test_add_endpoint(self):
        endpoints = [Endpoint(name="a")]
        lb = LoadBalancer(endpoints=endpoints)
        
        lb.add_endpoint(Endpoint(name="b"))
        
        assert len(lb.endpoints) == 2
    
    def test_add_duplicate_endpoint_raises(self):
        endpoints = [Endpoint(name="a")]
        lb = LoadBalancer(endpoints=endpoints)
        
        with pytest.raises(ValueError, match="already exists"):
            lb.add_endpoint(Endpoint(name="a"))
    
    def test_remove_endpoint(self):
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
        ]
        lb = LoadBalancer(endpoints=endpoints)
        
        result = lb.remove_endpoint("a")
        assert result is True
        assert len(lb.endpoints) == 1
        assert lb.endpoints[0].name == "b"
        
        # Remove non-existent
        result = lb.remove_endpoint("xyz")
        assert result is False
    
    def test_set_strategy(self):
        endpoints = [Endpoint(name="a")]
        lb = LoadBalancer(endpoints=endpoints, strategy="round_robin")
        
        lb.set_strategy("weighted")
        assert lb.strategy == "weighted"
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            lb.set_strategy("invalid")
    
    def test_get_stats_all(self):
        endpoints = [
            Endpoint(name="a"),
            Endpoint(name="b"),
        ]
        lb = LoadBalancer(endpoints=endpoints)
        
        lb.mark_success(endpoints[0])
        
        stats = lb.get_stats()
        assert "a" in stats
        assert "b" in stats
        assert stats["a"]["total_requests"] == 1
        assert stats["b"]["total_requests"] == 0


class TestCreateLoadBalancer:
    """Tests for create_load_balancer convenience function."""
    
    def test_basic_creation(self):
        lb = create_load_balancer(
            endpoints=[
                {"name": "primary", "provider": "openai"},
                {"name": "secondary", "provider": "anthropic"},
            ],
            strategy="weighted"
        )
        
        assert lb.strategy == "weighted"
        assert len(lb.endpoints) == 2
    
    def test_with_config(self):
        lb = create_load_balancer(
            endpoints=[{"name": "test"}],
            failure_threshold=10,
            recovery_time=60.0
        )
        
        assert lb._config.failure_threshold == 10
        assert lb._config.recovery_time == 60.0


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_mark_operations(self):
        endpoints = [Endpoint(name="test")]
        lb = LoadBalancer(endpoints=endpoints)
        ep = endpoints[0]
        
        errors = []
        
        def mark_ops():
            try:
                for _ in range(100):
                    lb.mark_success(ep, latency=0.1)
                    lb.mark_failure(ep)
                    lb.get_endpoint()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=mark_ops) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Stats should be consistent
        stats = lb.get_stats("test")
        assert stats["total_requests"] == 1000  # 5 threads * 100 * 2 ops
