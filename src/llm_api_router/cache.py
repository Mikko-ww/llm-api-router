"""
Response caching module for LLM API Router.

This module provides caching capabilities to reduce redundant API calls
and improve performance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = False  # Whether caching is enabled
    backend: str = "memory"  # Cache backend: "memory" or "redis"
    ttl: int = 3600  # Time-to-live in seconds (default: 1 hour)
    max_size: int = 1000  # Maximum number of cached items (for memory backend)
    # Redis configuration (optional, only used when backend="redis")
    redis_url: Optional[str] = None  # Redis connection URL, e.g., "redis://localhost:6379/0"
    redis_prefix: str = "llm_cache:"  # Redis key prefix


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend with LRU eviction and TTL support.
    
    This implementation uses OrderedDict for LRU and stores expiration
    timestamps with each value.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize memory cache backend.
        
        Args:
            max_size: Maximum number of items to store
            default_ttl: Default time-to-live in seconds
        """
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry['expires_at'] < time.time():
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL"""
        with self._lock:
            if ttl is None:
                ttl = self._default_ttl
            
            expires_at = time.time() + ttl
            
            # Update existing key or add new one
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at
            }
            
            # Evict oldest item if cache is full
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)  # Remove oldest (FIFO/LRU)
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cached values"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            # Clean up expired entries for accurate count
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry['expires_at'] < current_time
            ]
            for key in expired_keys:
                del self._cache[key]
            
            return {
                'backend': 'memory',
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
            }


class RedisCacheBackend(CacheBackend):
    """
    Redis cache backend with TTL support.
    
    Requires redis package to be installed.
    """
    
    def __init__(self, redis_url: str, prefix: str = "llm_cache:", default_ttl: int = 3600):
        """
        Initialize Redis cache backend.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            prefix: Key prefix for all cache keys
            default_ttl: Default time-to-live in seconds
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis cache backend requires 'redis' package. "
                "Install it with: pip install redis"
            )
        
        self._redis = redis.from_url(redis_url, decode_responses=False)
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._lock = Lock()
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self._prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        try:
            full_key = self._make_key(key)
            value = self._redis.get(full_key)
            
            with self._lock:
                if value is None:
                    self._misses += 1
                    return None
                
                self._hits += 1
                return json.loads(value)
        except Exception:
            with self._lock:
                self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL"""
        try:
            if ttl is None:
                ttl = self._default_ttl
            
            full_key = self._make_key(key)
            serialized = json.dumps(value)
            self._redis.setex(full_key, ttl, serialized)
        except Exception:
            # Silently fail to avoid breaking the main flow
            pass
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        try:
            full_key = self._make_key(key)
            self._redis.delete(full_key)
        except Exception:
            pass
    
    def clear(self) -> None:
        """Clear all cached values with the prefix"""
        try:
            # Find all keys with our prefix
            pattern = f"{self._prefix}*"
            for key in self._redis.scan_iter(match=pattern):
                self._redis.delete(key)
            
            with self._lock:
                self._hits = 0
                self._misses = 0
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self._lock:
                total_requests = self._hits + self._misses
                hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
                
                # Count keys with our prefix
                pattern = f"{self._prefix}*"
                size = sum(1 for _ in self._redis.scan_iter(match=pattern))
                
                return {
                    'backend': 'redis',
                    'size': size,
                    'hits': self._hits,
                    'misses': self._misses,
                    'hit_rate': hit_rate,
                }
        except Exception:
            return {
                'backend': 'redis',
                'error': 'Unable to retrieve stats'
            }


def generate_cache_key(request_data: Dict[str, Any], exclude_keys: Optional[List[str]] = None) -> str:
    """
    Generate a cache key based on request content hash.
    
    Args:
        request_data: Request data dictionary
        exclude_keys: Keys to exclude from hashing (e.g., 'request_id', 'stream')
        
    Returns:
        SHA256 hash of the request data
    """
    if exclude_keys is None:
        exclude_keys = ['request_id', 'stream']  # Exclude non-content fields
    
    # Create a copy without excluded keys
    filtered_data = {k: v for k, v in request_data.items() if k not in exclude_keys}
    
    # Sort and serialize to ensure consistent hashing
    serialized = json.dumps(filtered_data, sort_keys=True)
    
    # Generate SHA256 hash
    return hashlib.sha256(serialized.encode()).hexdigest()


class CacheManager:
    """
    Cache manager that handles caching of LLM responses.
    
    This class provides a simple interface for caching and supports
    multiple backends (memory, Redis).
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._backend: Optional[CacheBackend] = None
        
        if config.enabled:
            if config.backend == "memory":
                self._backend = MemoryCacheBackend(
                    max_size=config.max_size,
                    default_ttl=config.ttl
                )
            elif config.backend == "redis":
                if not config.redis_url:
                    raise ValueError("redis_url is required when backend='redis'")
                self._backend = RedisCacheBackend(
                    redis_url=config.redis_url,
                    prefix=config.redis_prefix,
                    default_ttl=config.ttl
                )
            else:
                raise ValueError(f"Unsupported cache backend: {config.backend}")
    
    @property
    def enabled(self) -> bool:
        """Check if caching is enabled"""
        return self.config.enabled and self._backend is not None
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        if not self.enabled:
            return None
        return self._backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache"""
        if not self.enabled:
            return
        self._backend.set(key, value, ttl)
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        if not self.enabled:
            return
        self._backend.delete(key)
    
    def clear(self) -> None:
        """Clear all cached values"""
        if not self.enabled:
            return
        self._backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        stats = self._backend.get_stats()
        stats['enabled'] = True
        stats['ttl'] = self.config.ttl
        return stats
