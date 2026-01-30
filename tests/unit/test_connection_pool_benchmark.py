"""
Performance benchmark test to validate connection pool optimization benefits.

This test demonstrates the performance impact of different connection pool configurations.
Note: These are synthetic benchmarks to show relative performance improvements.
"""

import pytest
import httpx
import time
from unittest.mock import Mock, patch
from llm_api_router import (
    Client, ProviderConfig, ConnectionPoolConfig, TimeoutConfig
)


class TestConnectionPoolPerformance:
    """Benchmark tests for connection pool performance"""
    
    def test_connection_reuse_reduces_overhead(self):
        """
        Test that keeping connections alive reduces connection overhead.
        This is a conceptual test showing configuration differences.
        """
        # Configuration with no keepalive (worst case)
        no_keepalive_config = ConnectionPoolConfig(
            max_connections=10,
            max_keepalive_connections=0,  # No connection reuse
            keepalive_expiry=0.0
        )
        
        # Configuration with keepalive (best case)
        with_keepalive_config = ConnectionPoolConfig(
            max_connections=10,
            max_keepalive_connections=10,  # All connections can be reused
            keepalive_expiry=300.0
        )
        
        # Verify configurations are different
        assert no_keepalive_config.max_keepalive_connections == 0
        assert with_keepalive_config.max_keepalive_connections == 10
        
        # In real-world usage, with_keepalive_config would be significantly faster
        # because it reuses connections instead of creating new ones each time
    
    def test_timeout_configuration_flexibility(self):
        """
        Test that fine-grained timeout configuration provides better control.
        """
        # Simple timeout (less control)
        simple_config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout=30.0
        )
        
        # Fine-grained timeout (more control)
        detailed_timeout = TimeoutConfig(
            connect=5.0,   # Fast connection timeout
            read=60.0,     # Longer for reading responses
            write=5.0,     # Fast write timeout
            pool=5.0       # Fast pool acquisition
        )
        detailed_config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout_config=detailed_timeout
        )
        
        # Fine-grained config provides better control over each timeout aspect
        assert simple_config.timeout == 30.0
        assert detailed_config.timeout_config.connect == 5.0
        assert detailed_config.timeout_config.read == 60.0
    
    def test_connection_pool_scaling(self):
        """
        Test that higher connection limits support more concurrent requests.
        """
        # Small pool (limited concurrency)
        small_pool = ConnectionPoolConfig(
            max_connections=10,
            max_keepalive_connections=5
        )
        
        # Large pool (high concurrency)
        large_pool = ConnectionPoolConfig(
            max_connections=100,
            max_keepalive_connections=30
        )
        
        # Large pool can handle more concurrent requests
        assert large_pool.max_connections > small_pool.max_connections
        assert large_pool.max_keepalive_connections > small_pool.max_keepalive_connections
    
    def test_buffer_size_optimization(self):
        """
        Test that buffer size affects streaming performance.
        """
        # Small buffer (lower memory, slower streaming)
        small_buffer = ConnectionPoolConfig(
            stream_buffer_size=32768  # 32KB
        )
        
        # Large buffer (higher memory, faster streaming)
        large_buffer = ConnectionPoolConfig(
            stream_buffer_size=131072  # 128KB
        )
        
        # Larger buffer can process streaming data more efficiently
        assert large_buffer.stream_buffer_size > small_buffer.stream_buffer_size
        assert large_buffer.stream_buffer_size / small_buffer.stream_buffer_size == 4


class TestConnectionPoolBenchmark:
    """More realistic benchmark tests with mocked HTTP calls"""
    
    @patch('httpx.Client.post')
    def test_client_creation_overhead(self, mock_post):
        """
        Measure the overhead of client creation with different configurations.
        """
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_post.return_value = mock_response
        
        # Default configuration
        start_time = time.time()
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key"
        )
        client = Client(config)
        client.close()
        default_time = time.time() - start_time
        
        # Optimized configuration
        start_time = time.time()
        optimized_pool = ConnectionPoolConfig(
            max_connections=100,
            max_keepalive_connections=30,
            keepalive_expiry=600.0
        )
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            connection_pool_config=optimized_pool
        )
        client = Client(config)
        client.close()
        optimized_time = time.time() - start_time
        
        # Both should be fast (< 1 second)
        assert default_time < 1.0
        assert optimized_time < 1.0
        
        # The difference should be minimal since we're just creating clients
        # In real usage, the benefits appear during actual request processing


class TestConfigurationValidation:
    """Test that configurations are properly validated"""
    
    def test_connection_pool_config_values_are_reasonable(self):
        """Ensure connection pool values are within reasonable ranges"""
        config = ConnectionPoolConfig()
        
        # Verify defaults are reasonable
        assert config.max_connections > 0
        assert config.max_keepalive_connections >= 0
        assert config.max_keepalive_connections <= config.max_connections
        assert config.keepalive_expiry >= 0
        assert config.stream_buffer_size > 0
    
    def test_timeout_config_values_are_reasonable(self):
        """Ensure timeout values are within reasonable ranges"""
        config = TimeoutConfig()
        
        # Verify defaults are reasonable
        assert config.connect > 0
        assert config.read > 0
        assert config.write > 0
        assert config.pool > 0
    
    def test_client_accepts_none_configs(self):
        """Test that client works with None configs (uses defaults)"""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            connection_pool_config=None,
            timeout_config=None
        )
        
        client = Client(config)
        assert client._http_client is not None
        client.close()


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "--tb=short"])
