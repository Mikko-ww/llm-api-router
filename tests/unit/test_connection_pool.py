"""Tests for HTTP connection pool configuration and optimization."""

import pytest
import httpx
from llm_api_router import (
    Client, AsyncClient, ProviderConfig,
    ConnectionPoolConfig, TimeoutConfig
)


class TestTimeoutConfig:
    """Test TimeoutConfig dataclass"""
    
    def test_timeout_config_defaults(self):
        """Test TimeoutConfig with default values"""
        config = TimeoutConfig()
        assert config.connect == 10.0
        assert config.read == 60.0
        assert config.write == 10.0
        assert config.pool == 10.0
    
    def test_timeout_config_custom_values(self):
        """Test TimeoutConfig with custom values"""
        config = TimeoutConfig(
            connect=5.0,
            read=30.0,
            write=5.0,
            pool=5.0
        )
        assert config.connect == 5.0
        assert config.read == 30.0
        assert config.write == 5.0
        assert config.pool == 5.0


class TestConnectionPoolConfig:
    """Test ConnectionPoolConfig dataclass"""
    
    def test_connection_pool_config_defaults(self):
        """Test ConnectionPoolConfig with default values"""
        config = ConnectionPoolConfig()
        assert config.max_connections == 100
        assert config.max_keepalive_connections == 20
        assert config.keepalive_expiry == 300.0
        assert config.stream_buffer_size == 65536
    
    def test_connection_pool_config_custom_values(self):
        """Test ConnectionPoolConfig with custom values"""
        config = ConnectionPoolConfig(
            max_connections=50,
            max_keepalive_connections=10,
            keepalive_expiry=120.0,
            stream_buffer_size=32768
        )
        assert config.max_connections == 50
        assert config.max_keepalive_connections == 10
        assert config.keepalive_expiry == 120.0
        assert config.stream_buffer_size == 32768


class TestClientConnectionPool:
    """Test Client with connection pool configuration"""
    
    def test_client_with_default_connection_pool(self):
        """Test Client uses default connection pool configuration"""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key"
        )
        client = Client(config)
        
        # Verify the client is created successfully
        assert client._http_client is not None
        assert isinstance(client._http_client, httpx.Client)
        
        # Verify the connection pool config is applied by checking transport existence
        assert hasattr(client._http_client, '_transport')
        
        client.close()
    
    def test_client_with_custom_connection_pool(self):
        """Test Client with custom connection pool configuration"""
        pool_config = ConnectionPoolConfig(
            max_connections=50,
            max_keepalive_connections=10,
            keepalive_expiry=120.0
        )
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            connection_pool_config=pool_config
        )
        client = Client(config)
        
        # Verify the client is created successfully with custom config
        assert client._http_client is not None
        assert isinstance(client._http_client, httpx.Client)
        assert client.config.connection_pool_config == pool_config
        
        client.close()
    
    def test_client_with_default_timeout(self):
        """Test Client with default simple timeout"""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout=30.0
        )
        client = Client(config)
        
        # Verify timeout is set
        assert client._http_client.timeout == httpx.Timeout(30.0)
        
        client.close()
    
    def test_client_with_timeout_config(self):
        """Test Client with fine-grained timeout configuration"""
        timeout_config = TimeoutConfig(
            connect=5.0,
            read=30.0,
            write=5.0,
            pool=5.0
        )
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout_config=timeout_config
        )
        client = Client(config)
        
        # Verify fine-grained timeout is set
        timeout = client._http_client.timeout
        assert timeout.connect == 5.0
        assert timeout.read == 30.0
        assert timeout.write == 5.0
        assert timeout.pool == 5.0
        
        client.close()
    
    def test_client_timeout_config_overrides_simple_timeout(self):
        """Test that timeout_config takes precedence over simple timeout"""
        timeout_config = TimeoutConfig(
            connect=5.0,
            read=30.0,
            write=5.0,
            pool=5.0
        )
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout=60.0,  # This should be ignored
            timeout_config=timeout_config
        )
        client = Client(config)
        
        # Verify timeout_config is used, not simple timeout
        timeout = client._http_client.timeout
        assert timeout.connect == 5.0
        assert timeout.read == 30.0
        
        client.close()


@pytest.mark.asyncio
class TestAsyncClientConnectionPool:
    """Test AsyncClient with connection pool configuration"""
    
    async def test_async_client_with_default_connection_pool(self):
        """Test AsyncClient uses default connection pool configuration"""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key"
        )
        async with AsyncClient(config) as client:
            # Verify the client is created successfully
            assert client._http_client is not None
            assert isinstance(client._http_client, httpx.AsyncClient)
            
            # Verify the connection pool config is applied by checking transport existence
            assert hasattr(client._http_client, '_transport')
    
    async def test_async_client_with_custom_connection_pool(self):
        """Test AsyncClient with custom connection pool configuration"""
        pool_config = ConnectionPoolConfig(
            max_connections=50,
            max_keepalive_connections=10,
            keepalive_expiry=120.0
        )
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            connection_pool_config=pool_config
        )
        async with AsyncClient(config) as client:
            # Verify the client is created successfully with custom config
            assert client._http_client is not None
            assert isinstance(client._http_client, httpx.AsyncClient)
            assert client.config.connection_pool_config == pool_config
    
    async def test_async_client_with_timeout_config(self):
        """Test AsyncClient with fine-grained timeout configuration"""
        timeout_config = TimeoutConfig(
            connect=5.0,
            read=30.0,
            write=5.0,
            pool=5.0
        )
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout_config=timeout_config
        )
        async with AsyncClient(config) as client:
            # Verify fine-grained timeout is set
            timeout = client._http_client.timeout
            assert timeout.connect == 5.0
            assert timeout.read == 30.0
            assert timeout.write == 5.0
            assert timeout.pool == 5.0


class TestProviderConfigWithNewOptions:
    """Test ProviderConfig with new connection pool and timeout options"""
    
    def test_provider_config_with_all_options(self):
        """Test ProviderConfig with all connection pool and timeout options"""
        timeout_config = TimeoutConfig(connect=5.0, read=30.0)
        pool_config = ConnectionPoolConfig(max_connections=50)
        
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout_config=timeout_config,
            connection_pool_config=pool_config
        )
        
        assert config.timeout_config == timeout_config
        assert config.connection_pool_config == pool_config
        assert config.timeout == 60.0  # Default value maintained
    
    def test_provider_config_backward_compatible(self):
        """Test ProviderConfig is backward compatible with simple timeout"""
        config = ProviderConfig(
            provider_type="openai",
            api_key="test-key",
            timeout=30.0
        )
        
        assert config.timeout == 30.0
        assert config.timeout_config is None
        assert config.connection_pool_config is None
