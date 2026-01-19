##Script function and purpose: TTL Cache and Cache Manager - Performance caching layer for expensive operations
##Dependency purpose: Provides efficient caching with TTL (Time-To-Live) for expensive operations like centrality scores and semantic searches
"""Caching System for JENOVA.

This module provides efficient caching for expensive operations like centrality
scores and semantic searches. Implements LRU cache with TTL (Time-To-Live)
expiration.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import structlog

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for cache statistics
class CacheStatsProtocol(Protocol):
    """Protocol for cache statistics."""
    
    ##Method purpose: Get cache statistics
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        ...


##Class purpose: LRU cache with TTL (Time-To-Live) for expensive operations
class TTLCache:
    """LRU cache with TTL expiration.
    
    Provides efficient caching with automatic expiration based on time-to-live.
    Uses LRU eviction when cache reaches maximum size.
    
    Attributes:
        max_size: Maximum number of items to cache.
        ttl_seconds: Time-to-live in seconds.
        cache: OrderedDict storing (value, timestamp) tuples.
        hits: Number of cache hits.
        misses: Number of cache misses.
    """
    
    ##Method purpose: Initialize TTL cache with size limit and expiration time
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        """Initialize TTL cache.
        
        Args:
            max_size: Maximum number of items to cache (LRU eviction).
            ttl_seconds: Time-to-live in seconds (default: 5 minutes).
        """
        ##Step purpose: Store configuration
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        ##Step purpose: Initialize cache storage
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        
        ##Step purpose: Initialize statistics
        self.hits = 0
        self.misses = 0
        
        ##Action purpose: Log initialization
        logger.debug(
            "ttl_cache_initialized",
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )
    
    ##Method purpose: Get item from cache if not expired
    def get(self, key: str) -> Any | None:
        """Retrieve item from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value if found and not expired, None otherwise.
        """
        ##Condition purpose: Check if key exists
        if key not in self.cache:
            self.misses += 1
            return None
        
        ##Step purpose: Get value and timestamp
        value, timestamp = self.cache[key]
        
        ##Condition purpose: Check if item has expired
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None
        
        ##Step purpose: Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return value
    
    ##Method purpose: Store item in cache with current timestamp
    def set(self, key: str, value: Any) -> None:
        """Store item in cache with current timestamp.
        
        Args:
            key: Cache key.
            value: Value to cache.
        """
        ##Condition purpose: Remove oldest item if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)  # Remove oldest (first) item
        
        ##Step purpose: Store with current timestamp
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)  # Mark as recently used
    
    ##Method purpose: Clear all cached items
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("ttl_cache_cleared")
    
    ##Method purpose: Get cache statistics
    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics.
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }
    
    ##Method purpose: Remove expired items from cache
    def cleanup_expired(self) -> int:
        """Remove all expired items from cache.
        
        Returns:
            Number of items removed.
        """
        current_time = time.time()
        expired_keys = [
            key for key, (_value, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        ##Loop purpose: Remove expired items
        for key in expired_keys:
            del self.cache[key]
        
        removed_count = len(expired_keys)
        if removed_count > 0:
            logger.debug("ttl_cache_cleanup", removed=removed_count)
        
        return removed_count


##Class purpose: Configuration for cache manager
@dataclass
class CacheManagerConfig:
    """Configuration for cache manager.
    
    Attributes:
        centrality_cache_size: Size for centrality cache (default: 500).
        centrality_cache_ttl: TTL for centrality cache in seconds (default: 300.0).
        node_search_cache_size: Size for node search cache (default: 1000).
        node_search_cache_ttl: TTL for node search cache in seconds (default: 600.0).
        recency_cache_size: Size for recency cache (default: 500).
        recency_cache_ttl: TTL for recency cache in seconds (default: 180.0).
        similarity_cache_size: Size for similarity cache (default: 2000).
        similarity_cache_ttl: TTL for similarity cache in seconds (default: 600.0).
        cleanup_interval: Cleanup interval in seconds (default: 300.0).
    """
    
    centrality_cache_size: int = 500
    centrality_cache_ttl: float = 300.0
    node_search_cache_size: int = 1000
    node_search_cache_ttl: float = 600.0
    recency_cache_size: int = 500
    recency_cache_ttl: float = 180.0
    similarity_cache_size: int = 2000
    similarity_cache_ttl: float = 600.0
    cleanup_interval: float = 300.0


##Class purpose: Cache manager for coordinating multiple caches
class CacheManager:
    """Cache manager for coordinating multiple caches.
    
    Manages multiple TTL caches for different operation types:
    - Centrality cache for graph centrality scores
    - Node search cache for graph search results
    - Recency cache for recency-based scoring
    - Similarity cache for similarity calculations
    
    Attributes:
        config: Cache manager configuration.
        centrality_cache: Cache for centrality scores.
        node_search_cache: Cache for node search results.
        recency_cache: Cache for recency scores.
        similarity_cache: Cache for similarity scores.
        last_cleanup: Timestamp of last cleanup.
    """
    
    ##Method purpose: Initialize cache manager with configuration
    def __init__(self, config: CacheManagerConfig | None = None) -> None:
        """Initialize cache manager.
        
        Args:
            config: Optional cache manager configuration.
        """
        ##Step purpose: Store configuration
        self.config = config or CacheManagerConfig()
        
        ##Step purpose: Initialize caches
        self.centrality_cache = TTLCache(
            max_size=self.config.centrality_cache_size,
            ttl_seconds=self.config.centrality_cache_ttl,
        )
        
        self.node_search_cache = TTLCache(
            max_size=self.config.node_search_cache_size,
            ttl_seconds=self.config.node_search_cache_ttl,
        )
        
        self.recency_cache = TTLCache(
            max_size=self.config.recency_cache_size,
            ttl_seconds=self.config.recency_cache_ttl,
        )
        
        self.similarity_cache = TTLCache(
            max_size=self.config.similarity_cache_size,
            ttl_seconds=self.config.similarity_cache_ttl,
        )
        
        ##Step purpose: Initialize cleanup tracking
        self.last_cleanup = time.time()
        
        ##Action purpose: Log initialization
        logger.info(
            "cache_manager_initialized",
            cleanup_interval=self.config.cleanup_interval,
        )
    
    ##Method purpose: Get or compute value using cache
    def get_or_compute(
        self,
        cache: TTLCache,
        key: str,
        compute_func: Callable[[], Any],
    ) -> Any:
        """Get value from cache or compute it using provided function.
        
        Args:
            cache: TTL cache to use.
            key: Cache key.
            compute_func: Function to compute value if not cached.
            
        Returns:
            Cached or computed value.
        """
        ##Step purpose: Try to get from cache
        cached_value = cache.get(key)
        if cached_value is not None:
            return cached_value
        
        ##Step purpose: Compute value and cache it
        value = compute_func()
        cache.set(key, value)
        return value
    
    ##Method purpose: Periodic cleanup of expired cache entries
    def cleanup_if_needed(self) -> None:
        """Clean up expired cache entries if cleanup interval has passed."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.config.cleanup_interval:
            ##Step purpose: Cleanup all caches
            self.centrality_cache.cleanup_expired()
            self.node_search_cache.cleanup_expired()
            self.recency_cache.cleanup_expired()
            self.similarity_cache.cleanup_expired()
            
            ##Step purpose: Update last cleanup time
            self.last_cleanup = current_time
            
            logger.debug("cache_manager_cleanup_completed")
    
    ##Method purpose: Clear all caches
    def clear_all(self) -> None:
        """Clear all caches."""
        self.centrality_cache.clear()
        self.node_search_cache.clear()
        self.recency_cache.clear()
        self.similarity_cache.clear()
        logger.info("cache_manager_cleared_all")
    
    ##Method purpose: Get performance statistics for all caches
    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get performance statistics for all caches.
        
        Returns:
            Dictionary mapping cache name to statistics.
        """
        return {
            "centrality_cache": self.centrality_cache.get_stats(),
            "node_search_cache": self.node_search_cache.get_stats(),
            "recency_cache": self.recency_cache.get_stats(),
            "similarity_cache": self.similarity_cache.get_stats(),
        }
    
    ##Method purpose: Generate cache key from multiple components
    @staticmethod
    def make_key(*components: Any) -> str:
        """Generate a cache key from multiple components.
        
        Args:
            *components: Components to combine into key.
            
        Returns:
            Cache key string.
        """
        return "|".join(str(c) for c in components)
