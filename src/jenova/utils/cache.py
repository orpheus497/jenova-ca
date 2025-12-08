##Script function and purpose: Caching System for The JENOVA Cognitive Architecture
##This module provides efficient caching for expensive operations like centrality scores and semantic searches

import time
from typing import Dict, Any, Optional, Callable, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta

##Class purpose: LRU cache with TTL (Time-To-Live) for expensive operations
class TTLCache:
    ##Function purpose: Initialize TTL cache with size limit and expiration time
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        """
        Initialize TTL cache.
        
        Args:
            max_size: Maximum number of items to cache (LRU eviction)
            ttl_seconds: Time-to-live in seconds (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    ##Function purpose: Get item from cache if not expired
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache if it exists and hasn't expired."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        value, timestamp = self.cache[key]
        
        ##Block purpose: Check if item has expired
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None
        
        ##Block purpose: Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return value
    
    ##Function purpose: Store item in cache with current timestamp
    def set(self, key: str, value: Any) -> None:
        """Store item in cache with current timestamp."""
        ##Block purpose: Remove oldest item if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)  # Remove oldest (first) item
        
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)  # Mark as recently used
    
    ##Function purpose: Clear all cached items
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    ##Function purpose: Get cache statistics
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }
    
    ##Function purpose: Remove expired items from cache
    def cleanup_expired(self) -> int:
        """Remove all expired items from cache. Returns number of items removed."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

##Class purpose: Cache manager for coordinating multiple caches
class CacheManager:
    ##Function purpose: Initialize cache manager with configuration
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        cache_config = config.get('performance', {}).get('caching', {})
        
        ##Block purpose: Initialize caches with configuration
        self.centrality_cache = TTLCache(
            max_size=cache_config.get('centrality_cache_size', 500),
            ttl_seconds=cache_config.get('centrality_cache_ttl', 300.0)
        )
        
        self.node_search_cache = TTLCache(
            max_size=cache_config.get('node_search_cache_size', 1000),
            ttl_seconds=cache_config.get('node_search_cache_ttl', 600.0)
        )
        
        self.recency_cache = TTLCache(
            max_size=cache_config.get('recency_cache_size', 500),
            ttl_seconds=cache_config.get('recency_cache_ttl', 180.0)
        )
        
        self.similarity_cache = TTLCache(
            max_size=cache_config.get('similarity_cache_size', 2000),
            ttl_seconds=cache_config.get('similarity_cache_ttl', 600.0)
        )
        
        self.last_cleanup = time.time()
        self.cleanup_interval = cache_config.get('cleanup_interval', 300.0)  # 5 minutes
    
    ##Function purpose: Get or compute value using cache
    def get_or_compute(self, cache: TTLCache, key: str, compute_func: Callable[[], Any]) -> Any:
        """Get value from cache or compute it using provided function."""
        cached_value = cache.get(key)
        if cached_value is not None:
            return cached_value
        
        ##Block purpose: Compute value and cache it
        value = compute_func()
        cache.set(key, value)
        return value
    
    ##Function purpose: Periodic cleanup of expired cache entries
    def cleanup_if_needed(self) -> None:
        """Clean up expired cache entries if cleanup interval has passed."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.centrality_cache.cleanup_expired()
            self.node_search_cache.cleanup_expired()
            self.recency_cache.cleanup_expired()
            self.similarity_cache.cleanup_expired()
            self.last_cleanup = current_time
    
    ##Function purpose: Clear all caches
    def clear_all(self) -> None:
        """Clear all caches."""
        self.centrality_cache.clear()
        self.node_search_cache.clear()
        self.recency_cache.clear()
        self.similarity_cache.clear()
    
    ##Function purpose: Get performance statistics for all caches
    def get_all_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all caches."""
        return {
            'centrality_cache': self.centrality_cache.get_stats(),
            'node_search_cache': self.node_search_cache.get_stats(),
            'recency_cache': self.recency_cache.get_stats(),
            'similarity_cache': self.similarity_cache.get_stats()
        }
    
    ##Function purpose: Generate cache key from multiple components
    @staticmethod
    def make_key(*components: Any) -> str:
        """Generate a cache key from multiple components."""
        return '|'.join(str(c) for c in components)
