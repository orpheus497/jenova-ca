##Script function and purpose: TTL-based cache implementation for performance optimization
"""
Cache Module - TTL and LRU caching implementations.

This module provides caching infrastructure for performance optimization,
including TTL (time-to-live) expiration and LRU (least-recently-used)
eviction policies.

"""

import threading
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, TypeVar

import structlog

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)

##Step purpose: Define generic type for cache values
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


##Class purpose: A cached entry with TTL tracking
@dataclass
class CacheEntry(Generic[V]):
    """A cached entry with TTL and access tracking.

    Attributes:
        value: The cached value
        created_at: When the entry was created
        last_accessed: When the entry was last accessed
        ttl_seconds: Time-to-live in seconds (None = no expiration)
        access_count: Number of times entry was accessed
    """

    value: V
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl_seconds: int | None = None
    access_count: int = 0

    ##Method purpose: Check if entry has expired
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        ##Condition purpose: No TTL means never expires
        if self.ttl_seconds is None:
            return False

        age = datetime.now() - self.created_at
        return age.total_seconds() > self.ttl_seconds

    ##Method purpose: Mark entry as accessed
    def touch(self) -> None:
        """Mark the entry as accessed."""
        self.last_accessed = datetime.now()
        self.access_count += 1


##Class purpose: Cache statistics
@dataclass
class CacheStats:
    """Statistics for cache operations.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted
        expirations: Number of entries expired
        size: Current number of entries
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0

    ##Method purpose: Get hit rate
    @property
    def hit_rate(self) -> float:
        """Get the cache hit rate.

        Returns:
            Hit rate as float (0.0-1.0), or 0.0 if no requests made
        """
        total = self.hits + self.misses
        ##Condition purpose: Avoid division by zero
        if total == 0:
            return 0.0
        return self.hits / total


##Class purpose: TTL cache with LRU eviction
class TTLCache(Generic[K, V]):
    """Thread-safe TTL cache with LRU eviction.

    This cache combines TTL expiration with LRU eviction for comprehensive
    cache management. Expired entries are lazily cleaned on access.

    Example:
        >>> cache = TTLCache[str, int](max_size=100, default_ttl=300)
        >>> cache.set("key", 42)
        >>> value = cache.get("key")  # Returns 42
        >>> cache.get("missing")  # Returns None
    """

    ##Method purpose: Initialize the TTL cache
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int | None = 300,
        cleanup_interval: int = 60,
    ) -> None:
        """Initialize the TTL cache.

        Args:
            max_size: Maximum number of entries (default 1000)
            default_ttl: Default TTL in seconds (None = no expiration)
            cleanup_interval: Seconds between cleanup runs (default 60)
        """
        ##Update: Use OrderedDict for O(1) LRU tracking
        ##Step purpose: Initialize cache storage with OrderedDict for efficient LRU
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval

        ##Step purpose: Initialize thread safety lock
        self._lock = threading.Lock()

        ##Step purpose: Initialize statistics
        self._stats = CacheStats()

        ##Step purpose: Track last cleanup time
        self._last_cleanup = datetime.now()

        logger.debug(
            "cache_initialized",
            max_size=max_size,
            default_ttl=default_ttl,
        )

    ##Method purpose: Get a value from cache
    def get(self, key: K) -> V | None:
        """Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found/expired
        """
        with self._lock:
            ##Step purpose: Trigger cleanup if needed
            self._maybe_cleanup()

            ##Step purpose: Look up entry
            entry = self._cache.get(key)

            ##Condition purpose: Key not found
            if entry is None:
                self._stats.misses += 1
                return None

            ##Condition purpose: Entry expired
            if entry.is_expired():
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            ##Update: Move to end for LRU tracking (O(1) operation)
            ##Step purpose: Mark as accessed and move to end for LRU
            entry.touch()
            ##Action purpose: Move accessed entry to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return entry.value

    ##Method purpose: Set a value in cache
    def set(self, key: K, value: V, ttl: int | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: TTL in seconds (None = use default)
        """
        with self._lock:
            ##Step purpose: Use default TTL if not specified
            effective_ttl = ttl if ttl is not None else self._default_ttl

            ##Step purpose: Create entry
            entry = CacheEntry(value=value, ttl_seconds=effective_ttl)

            ##Condition purpose: Evict if at capacity and key is new
            if key not in self._cache and len(self._cache) >= self._max_size:
                self._evict_lru()

            ##Update: Store entry and move to end for LRU tracking
            ##Step purpose: Store entry and mark as most recently used
            self._cache[key] = entry
            ##Action purpose: Move new/updated entry to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.size = len(self._cache)

    ##Method purpose: Delete a key from cache
    def delete(self, key: K) -> bool:
        """Delete a key from the cache.

        Args:
            key: The cache key

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            ##Condition purpose: Check if key exists
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    ##Method purpose: Clear all entries
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
            logger.debug("cache_cleared")

    ##Method purpose: Check if key exists and is valid
    def has(self, key: K) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: The cache key

        Returns:
            True if key exists and is valid
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.size = len(self._cache)
                return False
            return True

    ##Update: Optimize LRU eviction to O(1) using OrderedDict
    ##Method purpose: Evict least recently used entry
    def _evict_lru(self) -> None:
        """Evict the least recently used entry.

        Uses OrderedDict's FIFO order where first item is least recently used.
        This is O(1) operation compared to O(n) min() search.
        """
        ##Condition purpose: Nothing to evict if empty
        if not self._cache:
            return

        ##Update: OrderedDict maintains insertion order, first item is LRU
        ##Step purpose: Pop first item (least recently used) - O(1) operation
        lru_key, _ = self._cache.popitem(last=False)
        self._stats.evictions += 1
        self._stats.size = len(self._cache)

        logger.debug("cache_eviction", evicted_key=str(lru_key))

    ##Method purpose: Maybe run cleanup if interval elapsed
    def _maybe_cleanup(self) -> None:
        """Run cleanup if cleanup interval has elapsed."""
        elapsed = datetime.now() - self._last_cleanup
        ##Condition purpose: Check if cleanup needed
        if elapsed.total_seconds() < self._cleanup_interval:
            return

        self._cleanup_expired()
        self._last_cleanup = datetime.now()

    ##Method purpose: Remove all expired entries
    def _cleanup_expired(self) -> None:
        """Remove all expired entries from the cache."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        ##Loop purpose: Remove expired entries
        for key in expired_keys:
            del self._cache[key]
            self._stats.expirations += 1

        self._stats.size = len(self._cache)

        ##Condition purpose: Log if entries were cleaned
        if expired_keys:
            logger.debug("cache_cleanup", expired_count=len(expired_keys))

    ##Method purpose: Get cache statistics
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                size=self._stats.size,
            )

    ##Method purpose: Get current size
    def __len__(self) -> int:
        """Get the current number of entries."""
        with self._lock:
            return len(self._cache)


##Class purpose: Manager for multiple named caches
class CacheManager:
    """Manager for multiple named caches.

    Provides centralized management of multiple cache instances with
    shared configuration and monitoring.

    Example:
        >>> manager = CacheManager()
        >>> manager.create_cache("responses", max_size=100, default_ttl=60)
        >>> manager.get_cache("responses").set("key", "value")
    """

    ##Method purpose: Initialize the cache manager
    def __init__(self) -> None:
        """Initialize the cache manager."""
        self._caches: dict[str, TTLCache[str, object]] = {}
        self._lock = threading.Lock()
        logger.info("cache_manager_initialized")

    ##Method purpose: Create a new named cache
    def create_cache(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl: int | None = 300,
    ) -> TTLCache[str, object]:
        """Create a new named cache.

        Args:
            name: Name for the cache
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds

        Returns:
            The created cache
        """
        with self._lock:
            ##Condition purpose: Return existing if already created
            if name in self._caches:
                return self._caches[name]

            ##Step purpose: Create and store new cache
            cache: TTLCache[str, object] = TTLCache(
                max_size=max_size,
                default_ttl=default_ttl,
            )
            self._caches[name] = cache

            logger.info(
                "cache_created",
                name=name,
                max_size=max_size,
                default_ttl=default_ttl,
            )

            return cache

    ##Method purpose: Get a cache by name
    def get_cache(self, name: str) -> TTLCache[str, object] | None:
        """Get a cache by name.

        Args:
            name: The cache name

        Returns:
            The cache or None if not found
        """
        with self._lock:
            return self._caches.get(name)

    ##Method purpose: Delete a cache by name
    def delete_cache(self, name: str) -> bool:
        """Delete a cache by name.

        Args:
            name: The cache name

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            ##Condition purpose: Check if cache exists
            if name in self._caches:
                del self._caches[name]
                logger.info("cache_deleted", name=name)
                return True
            return False

    ##Method purpose: Clear all caches
    def clear_all(self) -> None:
        """Clear all entries from all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("cache_manager_cleared_all")

    ##Method purpose: Get stats for all caches
    def get_all_stats(self) -> dict[str, CacheStats]:
        """Get statistics for all caches.

        Returns:
            Dictionary mapping cache names to their stats
        """
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}

    ##Method purpose: List all cache names
    def list_caches(self) -> list[str]:
        """List all cache names."""
        with self._lock:
            return list(self._caches.keys())
