"""Unit tests for metrics collection system"""

import pytest
import time
from datetime import datetime, timezone
from llm_api_router.metrics import (
    MetricsCollector,
    RequestMetrics,
    AggregatedMetrics,
    get_metrics_collector,
    set_metrics_collector,
)


class TestMetricsCollector:
    """Test MetricsCollector class"""
    
    def test_init(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector(enabled=True)
        assert collector.enabled is True
        assert len(collector._metrics) == 0
        
        disabled_collector = MetricsCollector(enabled=False)
        assert disabled_collector.enabled is False
    
    def test_record_request_enabled(self):
        """Test recording metrics when enabled"""
        collector = MetricsCollector(enabled=True)
        
        collector.record_request(
            provider="openai",
            model="gpt-4",
            latency_ms=150.5,
            success=True,
            status_code=200,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            request_id="test-123",
        )
        
        metrics = collector.get_metrics()
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.provider == "openai"
        assert metric.model == "gpt-4"
        assert metric.latency_ms == 150.5
        assert metric.success is True
        assert metric.status_code == 200
        assert metric.prompt_tokens == 100
        assert metric.completion_tokens == 50
        assert metric.total_tokens == 150
        assert metric.request_id == "test-123"
    
    def test_record_request_disabled(self):
        """Test that metrics are not recorded when disabled"""
        collector = MetricsCollector(enabled=False)
        
        collector.record_request(
            provider="openai",
            model="gpt-4",
            latency_ms=150.5,
            success=True,
        )
        
        metrics = collector.get_metrics()
        assert len(metrics) == 0
    
    def test_record_multiple_requests(self):
        """Test recording multiple requests"""
        collector = MetricsCollector(enabled=True)
        
        # Record 5 successful requests
        for i in range(5):
            collector.record_request(
                provider="openai",
                model="gpt-4",
                latency_ms=100 + i * 10,
                success=True,
                status_code=200,
                total_tokens=100 + i * 10,
            )
        
        # Record 2 failed requests
        for i in range(2):
            collector.record_request(
                provider="openai",
                model="gpt-4",
                latency_ms=200 + i * 10,
                success=False,
                status_code=500,
                error_type="ServerError",
            )
        
        metrics = collector.get_metrics()
        assert len(metrics) == 7
        
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]
        
        assert len(successful) == 5
        assert len(failed) == 2
    
    def test_get_metrics_filtered(self):
        """Test filtering metrics by provider and model"""
        collector = MetricsCollector(enabled=True)
        
        # Record metrics for different providers and models
        collector.record_request(provider="openai", model="gpt-4", latency_ms=100, success=True)
        collector.record_request(provider="openai", model="gpt-3.5-turbo", latency_ms=80, success=True)
        collector.record_request(provider="anthropic", model="claude-3-opus", latency_ms=120, success=True)
        collector.record_request(provider="anthropic", model="claude-3-opus", latency_ms=130, success=True)
        
        # Filter by provider
        openai_metrics = collector.get_metrics(provider="openai")
        assert len(openai_metrics) == 2
        assert all(m.provider == "openai" for m in openai_metrics)
        
        anthropic_metrics = collector.get_metrics(provider="anthropic")
        assert len(anthropic_metrics) == 2
        assert all(m.provider == "anthropic" for m in anthropic_metrics)
        
        # Filter by model
        gpt4_metrics = collector.get_metrics(model="gpt-4")
        assert len(gpt4_metrics) == 1
        assert gpt4_metrics[0].model == "gpt-4"
        
        # Filter by both
        specific_metrics = collector.get_metrics(provider="anthropic", model="claude-3-opus")
        assert len(specific_metrics) == 2
        assert all(m.provider == "anthropic" and m.model == "claude-3-opus" for m in specific_metrics)
    
    def test_get_aggregated_metrics(self):
        """Test aggregated metrics calculation"""
        collector = MetricsCollector(enabled=True)
        
        # Record multiple requests with varying latencies
        latencies = [100, 120, 110, 130, 115]
        for lat in latencies:
            collector.record_request(
                provider="openai",
                model="gpt-4",
                latency_ms=lat,
                success=True,
                status_code=200,
                prompt_tokens=50,
                completion_tokens=25,
                total_tokens=75,
            )
        
        # Record one failure
        collector.record_request(
            provider="openai",
            model="gpt-4",
            latency_ms=150,
            success=False,
            status_code=500,
            error_type="ServerError",
        )
        
        aggregated = collector.get_aggregated_metrics()
        assert len(aggregated) == 1
        
        agg = aggregated[0]
        assert agg.provider == "openai"
        assert agg.model == "gpt-4"
        assert agg.total_requests == 6
        assert agg.successful_requests == 5
        assert agg.failed_requests == 1
        assert abs(agg.success_rate - 5/6) < 0.001
        
        # Check latency stats
        assert agg.min_latency_ms == 100
        assert agg.max_latency_ms == 150
        assert agg.avg_latency_ms == pytest.approx(120.833, rel=0.01)
        assert agg.p50_latency_ms == pytest.approx(117.5, rel=0.01)  # Median of 6 values: [100, 110, 115, 120, 130, 150]
        
        # Check token stats
        assert agg.total_prompt_tokens == 250
        assert agg.total_completion_tokens == 125
        assert agg.total_tokens == 375
        assert agg.avg_prompt_tokens == 50
        assert agg.avg_completion_tokens == 25
        
        # Check error counts
        assert agg.error_counts == {"ServerError": 1}
    
    def test_aggregated_metrics_multiple_groups(self):
        """Test aggregated metrics with multiple provider/model combinations"""
        collector = MetricsCollector(enabled=True)
        
        # Record for different providers and models
        collector.record_request(provider="openai", model="gpt-4", latency_ms=100, success=True)
        collector.record_request(provider="openai", model="gpt-4", latency_ms=110, success=True)
        collector.record_request(provider="openai", model="gpt-3.5-turbo", latency_ms=80, success=True)
        collector.record_request(provider="anthropic", model="claude-3-opus", latency_ms=120, success=True)
        
        aggregated = collector.get_aggregated_metrics()
        assert len(aggregated) == 3
        
        # Check each group
        providers_models = {(agg.provider, agg.model) for agg in aggregated}
        assert ("openai", "gpt-4") in providers_models
        assert ("openai", "gpt-3.5-turbo") in providers_models
        assert ("anthropic", "claude-3-opus") in providers_models
    
    def test_export_prometheus(self):
        """Test Prometheus export format"""
        collector = MetricsCollector(enabled=True)
        
        # Record some metrics
        collector.record_request(
            provider="openai",
            model="gpt-4",
            latency_ms=100,
            success=True,
            status_code=200,
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
        )
        collector.record_request(
            provider="openai",
            model="gpt-4",
            latency_ms=120,
            success=True,
            status_code=200,
            prompt_tokens=60,
            completion_tokens=30,
            total_tokens=90,
        )
        
        prometheus_text = collector.export_prometheus()
        
        # Check that it contains expected content
        assert "# HELP llm_router_requests_total" in prometheus_text
        assert "# TYPE llm_router_requests_total counter" in prometheus_text
        assert 'llm_router_requests_total{provider="openai",model="gpt-4"} 2' in prometheus_text
        
        assert "# HELP llm_router_success_rate" in prometheus_text
        assert "# TYPE llm_router_success_rate gauge" in prometheus_text
        assert 'llm_router_success_rate{provider="openai",model="gpt-4"} 1.0000' in prometheus_text
        
        assert "# HELP llm_router_latency_ms" in prometheus_text
        assert "# TYPE llm_router_latency_ms summary" in prometheus_text
        
        assert "# HELP llm_router_tokens_total" in prometheus_text
        assert "# TYPE llm_router_tokens_total counter" in prometheus_text
        assert 'llm_router_tokens_total{provider="openai",model="gpt-4",type="prompt"} 110' in prometheus_text
        assert 'llm_router_tokens_total{provider="openai",model="gpt-4",type="completion"} 55' in prometheus_text
        assert 'llm_router_tokens_total{provider="openai",model="gpt-4",type="total"} 165' in prometheus_text
    
    def test_compare_providers(self):
        """Test provider comparison"""
        collector = MetricsCollector(enabled=True)
        
        # Record metrics for OpenAI
        for i in range(5):
            collector.record_request(
                provider="openai",
                model="gpt-4",
                latency_ms=100 + i * 10,
                success=True,
                total_tokens=100,
            )
        
        # Record metrics for Anthropic (faster, but one failure)
        for i in range(5):
            collector.record_request(
                provider="anthropic",
                model="claude-3-opus",
                latency_ms=80 + i * 5,
                success=True,
                total_tokens=90,
            )
        collector.record_request(
            provider="anthropic",
            model="claude-3-opus",
            latency_ms=200,
            success=False,
            error_type="RateLimitError",
        )
        
        comparison = collector.compare_providers()
        
        # Should have 2 providers
        assert len(comparison) == 2
        
        # First should be openai (100% success rate)
        assert comparison[0]["provider"] == "openai"
        assert comparison[0]["success_rate"] == 1.0
        
        # Second should be anthropic (lower success rate)
        assert comparison[1]["provider"] == "anthropic"
        assert comparison[1]["success_rate"] == pytest.approx(5/6, rel=0.01)
    
    def test_reset(self):
        """Test resetting metrics"""
        collector = MetricsCollector(enabled=True)
        
        collector.record_request(provider="openai", model="gpt-4", latency_ms=100, success=True)
        collector.record_request(provider="openai", model="gpt-4", latency_ms=110, success=True)
        
        assert len(collector.get_metrics()) == 2
        
        collector.reset()
        
        assert len(collector.get_metrics()) == 0
        assert len(collector.get_aggregated_metrics()) == 0
    
    def test_get_summary(self):
        """Test getting metrics summary"""
        collector = MetricsCollector(enabled=True)
        
        # Empty summary
        summary = collector.get_summary()
        assert summary["total_requests"] == 0
        assert summary["successful_requests"] == 0
        assert summary["failed_requests"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["providers"] == []
        
        # Add some metrics
        collector.record_request(provider="openai", model="gpt-4", latency_ms=100, success=True)
        collector.record_request(provider="openai", model="gpt-4", latency_ms=110, success=True)
        collector.record_request(provider="anthropic", model="claude-3-opus", latency_ms=120, success=False)
        
        summary = collector.get_summary()
        assert summary["total_requests"] == 3
        assert summary["successful_requests"] == 2
        assert summary["failed_requests"] == 1
        assert abs(summary["success_rate"] - 2/3) < 0.001
        assert set(summary["providers"]) == {"openai", "anthropic"}
        assert len(summary["aggregated"]) == 2
    
    def test_percentile_calculation(self):
        """Test percentile calculation"""
        collector = MetricsCollector(enabled=True)
        
        # Test with simple case
        values = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        for val in values:
            collector.record_request(
                provider="test",
                model="test-model",
                latency_ms=val,
                success=True,
            )
        
        aggregated = collector.get_aggregated_metrics()[0]
        
        # p50 should be around 145 (50th percentile)
        assert aggregated.p50_latency_ms == pytest.approx(145, rel=0.1)
        
        # p95 should be around 186 (95th percentile)
        assert aggregated.p95_latency_ms == pytest.approx(186, rel=0.1)
        
        # p99 should be around 189.6 (99th percentile)
        assert aggregated.p99_latency_ms == pytest.approx(189.6, rel=0.1)


class TestGlobalMetricsCollector:
    """Test global metrics collector functions"""
    
    def test_get_metrics_collector(self):
        """Test getting global metrics collector"""
        # Reset to clean state
        set_metrics_collector(None)
        
        collector = get_metrics_collector(enabled=True)
        assert collector is not None
        assert collector.enabled is True
        
        # Should return same instance
        collector2 = get_metrics_collector()
        assert collector2 is collector
    
    def test_set_metrics_collector(self):
        """Test setting global metrics collector"""
        custom_collector = MetricsCollector(enabled=False)
        set_metrics_collector(custom_collector)
        
        collector = get_metrics_collector()
        assert collector is custom_collector
        assert collector.enabled is False
        
        # Clean up
        set_metrics_collector(None)


class TestRequestMetrics:
    """Test RequestMetrics dataclass"""
    
    def test_creation(self):
        """Test creating RequestMetrics"""
        now = datetime.now(timezone.utc)
        
        metrics = RequestMetrics(
            provider="openai",
            model="gpt-4",
            timestamp=now,
            latency_ms=150.5,
            success=True,
            status_code=200,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            stream=False,
            request_id="test-123",
        )
        
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"
        assert metrics.timestamp == now
        assert metrics.latency_ms == 150.5
        assert metrics.success is True
        assert metrics.status_code == 200
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.stream is False
        assert metrics.request_id == "test-123"


class TestAggregatedMetrics:
    """Test AggregatedMetrics dataclass"""
    
    def test_creation(self):
        """Test creating AggregatedMetrics"""
        now = datetime.now(timezone.utc)
        
        metrics = AggregatedMetrics(
            provider="openai",
            model="gpt-4",
            total_requests=10,
            successful_requests=9,
            failed_requests=1,
            success_rate=0.9,
            min_latency_ms=100.0,
            max_latency_ms=200.0,
            avg_latency_ms=150.0,
            p50_latency_ms=145.0,
            p95_latency_ms=190.0,
            p99_latency_ms=198.0,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            error_counts={"RateLimitError": 1},
            first_request_time=now,
            last_request_time=now,
        )
        
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"
        assert metrics.total_requests == 10
        assert metrics.successful_requests == 9
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 0.9
        assert metrics.error_counts == {"RateLimitError": 1}
