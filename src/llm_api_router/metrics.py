"""Performance monitoring and metrics collection for LLM API Router"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime, timezone
import statistics


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    
    provider: str
    model: Optional[str]
    timestamp: datetime
    latency_ms: float
    success: bool
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    
    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Additional metadata
    stream: bool = False
    request_id: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a provider/model combination"""
    
    provider: str
    model: Optional[str]
    
    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Success rate
    success_rate: float = 0.0
    
    # Latency statistics (in milliseconds)
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    
    # Token usage statistics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_prompt_tokens: Optional[float] = None
    avg_completion_tokens: Optional[float] = None
    
    # Error breakdown
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    # Time range
    first_request_time: Optional[datetime] = None
    last_request_time: Optional[datetime] = None


class MetricsCollector:
    """
    Collects and aggregates performance metrics for LLM API requests.
    
    Thread-safe metrics collection with support for:
    - Request latency tracking
    - Token usage statistics
    - Success/failure rates
    - Provider performance comparison
    - Prometheus export format
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize metrics collector
        
        Args:
            enabled: Whether to collect metrics (default: True)
        """
        self.enabled = enabled
        self._lock = threading.Lock()
        self._metrics: List[RequestMetrics] = []
        self._request_counts: Dict[str, int] = defaultdict(int)
    
    def record_request(
        self,
        provider: str,
        model: Optional[str],
        latency_ms: float,
        success: bool,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        stream: bool = False,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Record metrics for a single request
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3-opus")
            latency_ms: Request latency in milliseconds
            success: Whether the request was successful
            status_code: HTTP status code
            error_type: Error type if request failed
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total number of tokens
            stream: Whether the request was streaming
            request_id: Unique request ID
        """
        if not self.enabled:
            return
        
        metrics = RequestMetrics(
            provider=provider,
            model=model,
            timestamp=datetime.now(timezone.utc),
            latency_ms=latency_ms,
            success=success,
            status_code=status_code,
            error_type=error_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            stream=stream,
            request_id=request_id,
        )
        
        with self._lock:
            self._metrics.append(metrics)
            key = f"{provider}:{model or 'default'}"
            self._request_counts[key] += 1
    
    def get_metrics(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[RequestMetrics]:
        """
        Get raw metrics, optionally filtered by provider and/or model
        
        Args:
            provider: Filter by provider (optional)
            model: Filter by model (optional)
            
        Returns:
            List of RequestMetrics matching the filters
        """
        with self._lock:
            metrics = self._metrics.copy()
        
        if provider:
            metrics = [m for m in metrics if m.provider == provider]
        if model:
            metrics = [m for m in metrics if m.model == model]
        
        return metrics
    
    def get_aggregated_metrics(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[AggregatedMetrics]:
        """
        Get aggregated metrics grouped by provider and model
        
        Args:
            provider: Filter by provider (optional)
            model: Filter by model (optional)
            
        Returns:
            List of AggregatedMetrics for each provider/model combination
        """
        metrics = self.get_metrics(provider, model)
        
        if not metrics:
            return []
        
        # Group by provider and model
        groups: Dict[tuple, List[RequestMetrics]] = defaultdict(list)
        for m in metrics:
            key = (m.provider, m.model)
            groups[key].append(m)
        
        # Aggregate each group
        aggregated = []
        for (prov, mdl), group_metrics in groups.items():
            agg = self._aggregate_metrics_group(prov, mdl, group_metrics)
            aggregated.append(agg)
        
        return aggregated
    
    def _aggregate_metrics_group(
        self,
        provider: str,
        model: Optional[str],
        metrics: List[RequestMetrics],
    ) -> AggregatedMetrics:
        """Aggregate a group of metrics for a specific provider/model"""
        
        total_requests = len(metrics)
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]
        
        successful_requests = len(successful)
        failed_requests = len(failed)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
        
        # Latency statistics
        latencies = [m.latency_ms for m in metrics]
        latencies_sorted = sorted(latencies)
        
        min_latency = min(latencies) if latencies else None
        max_latency = max(latencies) if latencies else None
        avg_latency = statistics.mean(latencies) if latencies else None
        
        # Percentiles
        p50_latency = self._percentile(latencies_sorted, 0.50) if latencies_sorted else None
        p95_latency = self._percentile(latencies_sorted, 0.95) if latencies_sorted else None
        p99_latency = self._percentile(latencies_sorted, 0.99) if latencies_sorted else None
        
        # Token usage
        prompt_tokens_list = [m.prompt_tokens for m in metrics if m.prompt_tokens is not None]
        completion_tokens_list = [m.completion_tokens for m in metrics if m.completion_tokens is not None]
        total_tokens_list = [m.total_tokens for m in metrics if m.total_tokens is not None]
        
        total_prompt_tokens = sum(prompt_tokens_list)
        total_completion_tokens = sum(completion_tokens_list)
        total_tokens_sum = sum(total_tokens_list)
        
        avg_prompt_tokens = statistics.mean(prompt_tokens_list) if prompt_tokens_list else None
        avg_completion_tokens = statistics.mean(completion_tokens_list) if completion_tokens_list else None
        
        # Error breakdown
        error_counts: Dict[str, int] = defaultdict(int)
        for m in failed:
            if m.error_type:
                error_counts[m.error_type] += 1
        
        # Time range
        timestamps = [m.timestamp for m in metrics]
        first_request_time = min(timestamps) if timestamps else None
        last_request_time = max(timestamps) if timestamps else None
        
        return AggregatedMetrics(
            provider=provider,
            model=model,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens_sum,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_completion_tokens=avg_completion_tokens,
            error_counts=dict(error_counts),
            first_request_time=first_request_time,
            last_request_time=last_request_time,
        )
    
    @staticmethod
    def _percentile(sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted list of values"""
        if not sorted_values:
            return 0.0
        
        index = percentile * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format
        
        Returns:
            Prometheus-formatted metrics string
        """
        aggregated = self.get_aggregated_metrics()
        
        lines = [
            "# HELP llm_router_requests_total Total number of LLM API requests",
            "# TYPE llm_router_requests_total counter",
        ]
        
        for agg in aggregated:
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}"'
            lines.append(f"llm_router_requests_total{{{labels}}} {agg.total_requests}")
        
        lines.extend([
            "",
            "# HELP llm_router_requests_success Total number of successful requests",
            "# TYPE llm_router_requests_success counter",
        ])
        
        for agg in aggregated:
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}"'
            lines.append(f"llm_router_requests_success{{{labels}}} {agg.successful_requests}")
        
        lines.extend([
            "",
            "# HELP llm_router_requests_failed Total number of failed requests",
            "# TYPE llm_router_requests_failed counter",
        ])
        
        for agg in aggregated:
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}"'
            lines.append(f"llm_router_requests_failed{{{labels}}} {agg.failed_requests}")
        
        lines.extend([
            "",
            "# HELP llm_router_success_rate Request success rate (0-1)",
            "# TYPE llm_router_success_rate gauge",
        ])
        
        for agg in aggregated:
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}"'
            lines.append(f"llm_router_success_rate{{{labels}}} {agg.success_rate:.4f}")
        
        lines.extend([
            "",
            "# HELP llm_router_latency_ms Request latency in milliseconds",
            "# TYPE llm_router_latency_ms summary",
        ])
        
        for agg in aggregated:
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}"'
            if agg.avg_latency_ms is not None:
                lines.append(f"llm_router_latency_ms{{{labels},quantile=\"0.5\"}} {agg.p50_latency_ms:.2f}")
                lines.append(f"llm_router_latency_ms{{{labels},quantile=\"0.95\"}} {agg.p95_latency_ms:.2f}")
                lines.append(f"llm_router_latency_ms{{{labels},quantile=\"0.99\"}} {agg.p99_latency_ms:.2f}")
                lines.append(f"llm_router_latency_ms_sum{{{labels}}} {agg.avg_latency_ms * agg.total_requests:.2f}")
                lines.append(f"llm_router_latency_ms_count{{{labels}}} {agg.total_requests}")
        
        lines.extend([
            "",
            "# HELP llm_router_tokens_total Total tokens used",
            "# TYPE llm_router_tokens_total counter",
        ])
        
        for agg in aggregated:
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}",type="prompt"'
            lines.append(f"llm_router_tokens_total{{{labels}}} {agg.total_prompt_tokens}")
            
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}",type="completion"'
            lines.append(f"llm_router_tokens_total{{{labels}}} {agg.total_completion_tokens}")
            
            labels = f'provider="{agg.provider}",model="{agg.model or "default"}",type="total"'
            lines.append(f"llm_router_tokens_total{{{labels}}} {agg.total_tokens}")
        
        return "\n".join(lines) + "\n"
    
    def compare_providers(self) -> List[Dict[str, Any]]:
        """
        Compare performance across providers
        
        Returns:
            List of provider comparison data, sorted by success rate and latency
        """
        aggregated = self.get_aggregated_metrics()
        
        comparisons = []
        for agg in aggregated:
            comparisons.append({
                "provider": agg.provider,
                "model": agg.model,
                "total_requests": agg.total_requests,
                "success_rate": agg.success_rate,
                "avg_latency_ms": agg.avg_latency_ms,
                "p95_latency_ms": agg.p95_latency_ms,
                "total_tokens": agg.total_tokens,
                "avg_prompt_tokens": agg.avg_prompt_tokens,
                "avg_completion_tokens": agg.avg_completion_tokens,
            })
        
        # Sort by success rate (descending), then by avg latency (ascending)
        comparisons.sort(key=lambda x: (-x["success_rate"], x["avg_latency_ms"] or float("inf")))
        
        return comparisons
    
    def reset(self) -> None:
        """Clear all collected metrics"""
        with self._lock:
            self._metrics.clear()
            self._request_counts.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics
        
        Returns:
            Dictionary with overall metrics summary
        """
        with self._lock:
            metrics = self._metrics.copy()
        
        if not metrics:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "providers": [],
            }
        
        total_requests = len(metrics)
        successful = len([m for m in metrics if m.success])
        failed = total_requests - successful
        success_rate = successful / total_requests if total_requests > 0 else 0.0
        
        # Get unique providers
        providers = list(set(m.provider for m in metrics))
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": success_rate,
            "providers": providers,
            "aggregated": self.get_aggregated_metrics(),
        }


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(enabled: bool = True) -> MetricsCollector:
    """
    Get the global metrics collector instance
    
    Args:
        enabled: Whether to enable metrics collection
        
    Returns:
        MetricsCollector instance
    """
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector(enabled=enabled)
    
    return _global_metrics_collector


def set_metrics_collector(collector: Optional[MetricsCollector]) -> None:
    """
    Set the global metrics collector instance
    
    Args:
        collector: MetricsCollector instance or None to disable
    """
    global _global_metrics_collector
    _global_metrics_collector = collector
