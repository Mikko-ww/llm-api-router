"""
Tests for caching functionality
"""
import pytest
import time
from unittest.mock import Mock, patch
from llm_api_router.cache import (
    CacheConfig,
    CacheManager,
    MemoryCacheBackend,
    generate_cache_key,
)


class TestCacheConfig:
    """Test CacheConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = CacheConfig()
        assert config.enabled is False
        assert config.backend == "memory"
        assert config.ttl == 3600
        assert config.max_size == 1000
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CacheConfig(
            enabled=True,
            backend="redis",
            ttl=7200,
            max_size=500,
            redis_url="redis://localhost:6379/0"
        )
        assert config.enabled is True
        assert config.backend == "redis"
        assert config.ttl == 7200
        assert config.max_size == 500
        assert config.redis_url == "redis://localhost:6379/0"


class TestMemoryCacheBackend:
    """Test MemoryCacheBackend"""
    
    def test_basic_set_get(self):
        """Test basic set and get operations"""
        cache = MemoryCacheBackend(max_size=10, default_ttl=60)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_get_nonexistent_key(self):
        """Test getting a non-existent key"""
        cache = MemoryCacheBackend()
        assert cache.get("nonexistent") is None
    
    def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = MemoryCacheBackend(default_ttl=1)
        
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when max_size is reached"""
        cache = MemoryCacheBackend(max_size=3, default_ttl=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # All should be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
        # Add a fourth item, should evict key1 (oldest)
        cache.set("key4", "value4")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_lru_ordering(self):
        """Test that accessing items updates LRU order"""
        cache = MemoryCacheBackend(max_size=3, default_ttl=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1, making it most recently used
        cache.get("key1")
        
        # Add a fourth item, should evict key2 (now oldest)
        cache.set("key4", "value4")
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_update_existing_key(self):
        """Test updating an existing key"""
        cache = MemoryCacheBackend()
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"
    
    def test_delete(self):
        """Test deleting a key"""
        cache = MemoryCacheBackend()
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_clear(self):
        """Test clearing all cache"""
        cache = MemoryCacheBackend()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_get_stats(self):
        """Test getting cache statistics"""
        cache = MemoryCacheBackend(max_size=10)
        
        stats = cache.get_stats()
        assert stats['backend'] == 'memory'
        assert stats['size'] == 0
        assert stats['max_size'] == 10
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        
        # Add some items and access them
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3


class TestGenerateCacheKey:
    """Test generate_cache_key function"""
    
    def test_basic_key_generation(self):
        """Test basic key generation"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "temperature": 1.0
        }
        
        key = generate_cache_key(request_data)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 produces 64 hex characters
    
    def test_same_data_same_key(self):
        """Test that same data produces same key"""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "temperature": 1.0
        }
        
        key1 = generate_cache_key(request_data)
        key2 = generate_cache_key(request_data)
        assert key1 == key2
    
    def test_different_data_different_key(self):
        """Test that different data produces different key"""
        request_data1 = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
        }
        request_data2 = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "gpt-3.5-turbo",
        }
        
        key1 = generate_cache_key(request_data1)
        key2 = generate_cache_key(request_data2)
        assert key1 != key2
    
    def test_exclude_keys(self):
        """Test that excluded keys don't affect the hash"""
        request_data1 = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "request_id": "id1",
            "stream": False
        }
        request_data2 = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "request_id": "id2",  # Different request_id
            "stream": True  # Different stream value
        }
        
        # Default exclusions include 'request_id' and 'stream'
        key1 = generate_cache_key(request_data1)
        key2 = generate_cache_key(request_data2)
        assert key1 == key2
    
    def test_custom_exclude_keys(self):
        """Test custom exclude keys"""
        request_data1 = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "custom_field": "value1"
        }
        request_data2 = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "custom_field": "value2"
        }
        
        # Without exclusion, should be different
        key1 = generate_cache_key(request_data1, exclude_keys=[])
        key2 = generate_cache_key(request_data2, exclude_keys=[])
        assert key1 != key2
        
        # With exclusion, should be same
        key1 = generate_cache_key(request_data1, exclude_keys=['custom_field'])
        key2 = generate_cache_key(request_data2, exclude_keys=['custom_field'])
        assert key1 == key2


class TestCacheManager:
    """Test CacheManager"""
    
    def test_disabled_cache(self):
        """Test that disabled cache doesn't store anything"""
        config = CacheConfig(enabled=False)
        manager = CacheManager(config)
        
        assert not manager.enabled
        
        manager.set("key1", "value1")
        assert manager.get("key1") is None
    
    def test_memory_backend(self):
        """Test memory backend"""
        config = CacheConfig(enabled=True, backend="memory", ttl=60, max_size=10)
        manager = CacheManager(config)
        
        assert manager.enabled
        
        manager.set("key1", "value1")
        assert manager.get("key1") == "value1"
    
    def test_invalid_backend(self):
        """Test that invalid backend raises error"""
        config = CacheConfig(enabled=True, backend="invalid")
        
        with pytest.raises(ValueError, match="Unsupported cache backend"):
            CacheManager(config)
    
    def test_redis_backend_without_url(self):
        """Test that Redis backend without URL raises error"""
        config = CacheConfig(enabled=True, backend="redis")
        
        with pytest.raises(ValueError, match="redis_url is required"):
            CacheManager(config)
    
    def test_get_stats(self):
        """Test getting cache statistics"""
        config = CacheConfig(enabled=True, backend="memory", ttl=60)
        manager = CacheManager(config)
        
        manager.set("key1", "value1")
        manager.get("key1")
        
        stats = manager.get_stats()
        assert stats['enabled'] is True
        assert stats['backend'] == 'memory'
        assert stats['ttl'] == 60
        assert stats['size'] >= 1
    
    def test_clear_cache(self):
        """Test clearing cache"""
        config = CacheConfig(enabled=True, backend="memory")
        manager = CacheManager(config)
        
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        
        manager.clear()
        assert manager.get("key1") is None
        assert manager.get("key2") is None
    
    def test_delete_key(self):
        """Test deleting a specific key"""
        config = CacheConfig(enabled=True, backend="memory")
        manager = CacheManager(config)
        
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        
        manager.delete("key1")
        assert manager.get("key1") is None
        assert manager.get("key2") == "value2"


class TestRedisCacheBackend:
    """Test RedisCacheBackend (requires redis package)"""
    
    def test_redis_not_installed(self):
        """Test that missing redis package raises ImportError"""
        with patch.dict('sys.modules', {'redis': None}):
            from llm_api_router.cache import RedisCacheBackend
            
            with pytest.raises(ImportError, match="redis"):
                RedisCacheBackend("redis://localhost:6379/0")
