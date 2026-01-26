##Script function and purpose: Unit tests for TTL cache implementation
"""
Test suite for TTLCache and CacheManager - Thread-safe TTL and LRU caching.

Tests cover:
- TTL expiration
- LRU eviction
- Thread safety
- Statistics tracking
- Cache manager operations
"""

import threading
import time

import pytest

from jenova.utils.cache import CacheEntry, CacheManager, CacheStats, TTLCache


##Class purpose: Fixture providing fresh cache
@pytest.fixture
def cache() -> TTLCache[str, int]:
    """##Test case: Fresh TTL cache instance."""
    return TTLCache[str, int](max_size=10, default_ttl=300)


##Class purpose: Fixture providing short-ttl cache
@pytest.fixture
def short_ttl_cache() -> TTLCache[str, str]:
    """##Test case: Cache with short TTL for expiration tests."""
    return TTLCache[str, str](max_size=100, default_ttl=1)


##Function purpose: Test cache entry expiration check
def test_cache_entry_not_expired_no_ttl() -> None:
    """##Test case: Entry with no TTL never expires."""
    ##Step purpose: Create entry without TTL
    entry = CacheEntry(value="test", ttl_seconds=None)

    ##Assertion purpose: Verify not expired
    assert not entry.is_expired()


##Function purpose: Test cache entry expiration within TTL
def test_cache_entry_not_expired_within_ttl() -> None:
    """##Test case: Entry within TTL is not expired."""
    ##Step purpose: Create fresh entry
    entry = CacheEntry(value="test", ttl_seconds=300)

    ##Assertion purpose: Verify not expired
    assert not entry.is_expired()


##Function purpose: Test cache entry expiration after TTL
def test_cache_entry_expired_after_ttl() -> None:
    """##Test case: Entry after TTL is expired."""
    ##Step purpose: Create entry with past creation time
    entry = CacheEntry(value="test", ttl_seconds=1)

    ##Action purpose: Wait for expiration
    time.sleep(1.1)

    ##Assertion purpose: Verify expired
    assert entry.is_expired()


##Function purpose: Test cache entry touch
def test_cache_entry_touch() -> None:
    """##Test case: Touch updates access time and count."""
    ##Step purpose: Create entry and touch
    entry = CacheEntry(value="test")
    old_accessed = entry.last_accessed
    entry.touch()

    ##Assertion purpose: Verify updated
    assert entry.last_accessed > old_accessed
    assert entry.access_count == 1


##Function purpose: Test cache stats hit rate
def test_cache_stats_hit_rate() -> None:
    """##Test case: CacheStats calculates hit rate correctly."""
    ##Step purpose: Create stats and update
    stats = CacheStats()
    stats.hits = 8
    stats.misses = 2

    ##Assertion purpose: Verify rate (8 / 10 = 0.8)
    assert stats.hit_rate == 0.8


##Function purpose: Test cache stats hit rate zero
def test_cache_stats_hit_rate_zero() -> None:
    """##Test case: Hit rate is 0 when no accesses."""
    ##Step purpose: Create stats
    stats = CacheStats()

    ##Assertion purpose: Verify 0.0
    assert stats.hit_rate == 0.0


##Function purpose: Test cache initialization
def test_cache_initialization(cache: TTLCache[str, int]) -> None:
    """##Test case: Cache initializes with correct parameters."""
    ##Step purpose: Check initialized state

    ##Assertion purpose: Verify structure
    assert len(cache) == 0
    assert cache._max_size == 10
    assert cache._default_ttl == 300


##Function purpose: Test cache set and get
def test_cache_set_and_get(cache: TTLCache[str, int]) -> None:
    """##Test case: Can set and retrieve values."""
    ##Step purpose: Set value
    cache.set("key1", 42)

    ##Assertion purpose: Verify retrieval
    assert cache.get("key1") == 42


##Function purpose: Test cache miss
def test_cache_miss(cache: TTLCache[str, int]) -> None:
    """##Test case: Cache miss returns None."""
    ##Step purpose: Try to get non-existent key
    result = cache.get("nonexistent")

    ##Assertion purpose: Verify None returned
    assert result is None


##Function purpose: Test cache expiration on get
def test_cache_expiration_on_get(short_ttl_cache: TTLCache[str, str]) -> None:
    """##Test case: Expired entry returns None."""
    ##Step purpose: Set value with short TTL
    short_ttl_cache.set("key1", "value")

    ##Action purpose: Wait for expiration
    time.sleep(1.1)

    ##Assertion purpose: Verify None returned
    assert short_ttl_cache.get("key1") is None


##Function purpose: Test cache delete
def test_cache_delete(cache: TTLCache[str, int]) -> None:
    """##Test case: Can delete keys."""
    ##Step purpose: Set and delete
    cache.set("key1", 42)
    result = cache.delete("key1")

    ##Assertion purpose: Verify deleted
    assert result is True
    assert cache.get("key1") is None


##Function purpose: Test cache delete nonexistent
def test_cache_delete_nonexistent(cache: TTLCache[str, int]) -> None:
    """##Test case: Delete nonexistent returns False."""
    ##Step purpose: Try delete
    result = cache.delete("nonexistent")

    ##Assertion purpose: Verify False
    assert result is False


##Function purpose: Test cache clear
def test_cache_clear(cache: TTLCache[str, int]) -> None:
    """##Test case: Clear removes all entries."""
    ##Step purpose: Add entries and clear
    cache.set("key1", 1)
    cache.set("key2", 2)
    cache.clear()

    ##Assertion purpose: Verify empty
    assert len(cache) == 0


##Function purpose: Test cache has existing
def test_cache_has_existing(cache: TTLCache[str, int]) -> None:
    """##Test case: has returns True for existing keys."""
    ##Step purpose: Set and check
    cache.set("key1", 42)
    result = cache.has("key1")

    ##Assertion purpose: Verify True
    assert result is True


##Function purpose: Test cache has nonexistent
def test_cache_has_nonexistent(cache: TTLCache[str, int]) -> None:
    """##Test case: has returns False for missing keys."""
    ##Step purpose: Check missing key
    result = cache.has("nonexistent")

    ##Assertion purpose: Verify False
    assert result is False


##Function purpose: Test cache LRU eviction
def test_cache_lru_eviction() -> None:
    """##Test case: LRU eviction removes least recently used."""
    ##Step purpose: Create cache with size 3
    cache = TTLCache[str, int](max_size=3, default_ttl=300)

    ##Action purpose: Add entries
    cache.set("key1", 1)
    cache.set("key2", 2)
    cache.set("key3", 3)

    ##Step purpose: Access key1 to update last_accessed
    cache.get("key1")

    ##Action purpose: Add 4th entry (should evict key2, least recently used)
    cache.set("key4", 4)

    ##Assertion purpose: Verify eviction
    assert cache.get("key2") is None  # Should be evicted
    assert len(cache) == 3


##Function purpose: Test custom TTL override
def test_cache_custom_ttl(cache: TTLCache[str, str]) -> None:
    """##Test case: Can override default TTL per entry."""
    ##Step purpose: Set with custom TTL
    cache.set("key1", "value", ttl=1)

    ##Assertion purpose: Verify in cache initially
    assert cache.get("key1") == "value"

    ##Action purpose: Wait for custom TTL
    time.sleep(1.1)

    ##Assertion purpose: Verify expired
    assert cache.get("key1") is None


##Function purpose: Test cache statistics
def test_cache_statistics(cache: TTLCache[str, int]) -> None:
    """##Test case: Statistics track hits and misses."""
    ##Step purpose: Generate some activity
    cache.set("key1", 1)
    cache.get("key1")  # Hit
    cache.get("key2")  # Miss
    cache.get("key1")  # Hit

    ##Action purpose: Get stats
    stats = cache.get_stats()

    ##Assertion purpose: Verify counts
    assert stats.hits == 2
    assert stats.misses == 1


##Function purpose: Test cache statistics expiration
def test_cache_statistics_expiration(short_ttl_cache: TTLCache[str, str]) -> None:
    """##Test case: Statistics track expiration."""
    ##Step purpose: Set and expire entry
    short_ttl_cache.set("key1", "value")
    time.sleep(1.1)
    short_ttl_cache.get("key1")  # Should expire

    ##Action purpose: Get stats
    stats = short_ttl_cache.get_stats()

    ##Assertion purpose: Verify expiration counted
    assert stats.expirations == 1


##Function purpose: Test cache length
def test_cache_length(cache: TTLCache[str, int]) -> None:
    """##Test case: __len__ returns current size."""
    ##Step purpose: Add entries
    cache.set("key1", 1)
    cache.set("key2", 2)

    ##Assertion purpose: Verify length
    assert len(cache) == 2


##Function purpose: Test cache thread safety
def test_cache_thread_safety() -> None:
    """##Test case: Cache operations are thread-safe."""
    ##Step purpose: Create cache
    cache = TTLCache[str, int](max_size=1000, default_ttl=300)
    errors: list[Exception] = []

    ##Function purpose: Worker thread function
    def worker(thread_id: int) -> None:
        """Worker that performs cache operations."""
        try:
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                cache.set(key, i)
                cache.get(key)
        except Exception as e:
            errors.append(e)

    ##Action purpose: Run multiple threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ##Assertion purpose: Verify no errors
    assert len(errors) == 0
    assert len(cache) == 500  # 5 threads × 100 keys


##Function purpose: Test cache cleanup on get
def test_cache_cleanup_on_get(short_ttl_cache: TTLCache[str, str]) -> None:
    """##Test case: Cleanup is triggered on get."""
    ##Step purpose: Add entries with short TTL
    short_ttl_cache.set("key1", "value1")
    short_ttl_cache.set("key2", "value2")

    ##Action purpose: Wait for expiration
    time.sleep(1.1)

    ##Step purpose: Trigger cleanup by getting
    short_ttl_cache.get("key1")

    ##Action purpose: Get stats
    stats = short_ttl_cache.get_stats()

    ##Assertion purpose: Verify cleanup occurred (size reduced)
    assert stats.size <= 2


##Function purpose: Test cache manager initialization
def test_cache_manager_init() -> None:
    """##Test case: CacheManager initializes correctly."""
    ##Step purpose: Create manager
    manager = CacheManager()

    ##Assertion purpose: Verify structure
    assert len(manager.list_caches()) == 0


##Function purpose: Test cache manager create cache
def test_cache_manager_create_cache() -> None:
    """##Test case: CacheManager can create caches."""
    ##Step purpose: Create manager and cache
    manager = CacheManager()
    cache = manager.create_cache("test_cache", max_size=50, default_ttl=60)

    ##Assertion purpose: Verify created
    assert cache is not None
    assert "test_cache" in manager.list_caches()


##Function purpose: Test cache manager get cache
def test_cache_manager_get_cache() -> None:
    """##Test case: CacheManager can retrieve caches."""
    ##Step purpose: Create and get
    manager = CacheManager()
    created = manager.create_cache("test_cache")
    retrieved = manager.get_cache("test_cache")

    ##Assertion purpose: Verify same instance
    assert created is retrieved


##Function purpose: Test cache manager get nonexistent
def test_cache_manager_get_nonexistent() -> None:
    """##Test case: Getting nonexistent cache returns None."""
    ##Step purpose: Try to get nonexistent
    manager = CacheManager()
    result = manager.get_cache("nonexistent")

    ##Assertion purpose: Verify None
    assert result is None


##Function purpose: Test cache manager delete cache
def test_cache_manager_delete_cache() -> None:
    """##Test case: CacheManager can delete caches."""
    ##Step purpose: Create and delete
    manager = CacheManager()
    manager.create_cache("test_cache")
    result = manager.delete_cache("test_cache")

    ##Assertion purpose: Verify deleted
    assert result is True
    assert "test_cache" not in manager.list_caches()


##Function purpose: Test cache manager delete nonexistent
def test_cache_manager_delete_nonexistent() -> None:
    """##Test case: Deleting nonexistent cache returns False."""
    ##Step purpose: Try to delete nonexistent
    manager = CacheManager()
    result = manager.delete_cache("nonexistent")

    ##Assertion purpose: Verify False
    assert result is False


##Function purpose: Test cache manager clear all
def test_cache_manager_clear_all() -> None:
    """##Test case: CacheManager can clear all caches."""
    ##Step purpose: Create multiple caches
    manager = CacheManager()
    c1 = manager.create_cache("cache1")
    c2 = manager.create_cache("cache2")
    c1.set("key1", "value1")
    c2.set("key2", "value2")

    ##Action purpose: Clear all
    manager.clear_all()

    ##Assertion purpose: Verify cleared
    assert len(c1) == 0
    assert len(c2) == 0


##Function purpose: Test cache manager stats
def test_cache_manager_get_all_stats() -> None:
    """##Test case: CacheManager can get stats for all caches."""
    ##Step purpose: Create caches with activity
    manager = CacheManager()
    c1 = manager.create_cache("cache1")
    c2 = manager.create_cache("cache2")
    c1.set("key1", "value1")
    c1.get("key1")
    c2.set("key2", "value2")
    c2.get("missing")  # Miss

    ##Action purpose: Get stats
    all_stats = manager.get_all_stats()

    ##Assertion purpose: Verify structure
    assert "cache1" in all_stats
    assert "cache2" in all_stats
    assert all_stats["cache1"].hits == 1
    assert all_stats["cache2"].misses == 1


##Function purpose: Test cache update existing key
def test_cache_update_existing_key(cache: TTLCache[str, int]) -> None:
    """##Test case: Updating existing key doesn't evict."""
    ##Step purpose: Create cache with small size
    small_cache = TTLCache[str, int](max_size=2, default_ttl=300)
    small_cache.set("key1", 1)
    small_cache.set("key2", 2)

    ##Action purpose: Update existing key
    small_cache.set("key1", 10)

    ##Assertion purpose: Verify both exist
    assert small_cache.get("key1") == 10
    assert small_cache.get("key2") == 2
    assert len(small_cache) == 2
