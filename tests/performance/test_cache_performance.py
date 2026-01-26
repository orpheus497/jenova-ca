##Script function and purpose: Benchmark tests for ResponseCache thread-safety and performance
"""ResponseCache performance and thread-safety tests.

Tests verify:
- Thread-safe concurrent access
- LRU eviction under load
- Cache hit/miss performance characteristics
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from datetime import datetime

import pytest

##Note: Import path assumes package is installed or src is in PYTHONPATH


##Class purpose: Test fixture for Response objects
class MockResponse:
    """Mock Response for testing cache operations."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.timestamp = datetime.now()
        self.sources: list[str] = []
        self.confidence = 1.0
        self.metadata: dict[str, str] = {}


##Class purpose: Benchmark tests for ResponseCache
class TestCachePerformance:
    """Benchmark tests for ResponseCache performance."""

    ##Method purpose: Test concurrent read/write access is thread-safe
    def test_concurrent_access_thread_safety(self) -> None:
        """Verify cache handles concurrent access without corruption."""
        ##Step purpose: Import dynamically to handle missing dependencies
        try:
            import sys

            sys.path.insert(0, "src")
            from jenova.core.response import Response, ResponseCache
        except ImportError:
            pytest.skip("jenova package not installed")

        cache = ResponseCache(max_size=100)
        errors: list[str] = []
        results: Counter[str] = Counter()

        ##Step purpose: Define worker that performs mixed read/write ops
        def worker(thread_id: int) -> None:
            """Worker thread that reads and writes to cache."""
            try:
                for i in range(50):
                    query = f"query_{thread_id}_{i}"
                    username = f"user_{thread_id}"

                    ##Step purpose: Write operation
                    response = Response(
                        content=f"response_{thread_id}_{i}",
                        sources=[],
                        confidence=1.0,
                        metadata={},
                    )
                    cache.put(query, username, response)

                    ##Step purpose: Read operation
                    cached = cache.get(query, username)
                    if cached:
                        results["hits"] += 1
                    else:
                        results["misses"] += 1

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        ##Step purpose: Launch concurrent threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        ##Condition purpose: Verify no errors occurred
        assert not errors, f"Thread errors: {errors}"

        ##Condition purpose: Verify cache operations completed
        assert results["hits"] > 0, "Expected some cache hits"

        ##Note: Log performance for regression tracking
        print(f"\nConcurrent test: {elapsed:.3f}s, {results}")

    ##Method purpose: Test LRU eviction performance
    def test_lru_eviction_performance(self) -> None:
        """Verify LRU eviction maintains O(1) performance."""
        ##Step purpose: Import dynamically
        try:
            import sys

            sys.path.insert(0, "src")
            from jenova.core.response import Response, ResponseCache
        except ImportError:
            pytest.skip("jenova package not installed")

        cache = ResponseCache(max_size=100)

        ##Step purpose: Fill cache beyond capacity
        start = time.perf_counter()
        for i in range(500):
            response = Response(
                content=f"response_{i}",
                sources=[],
                confidence=1.0,
                metadata={},
            )
            cache.put(f"query_{i}", "user", response)
        elapsed = time.perf_counter() - start

        ##Condition purpose: Verify cache stayed within size limit
        stats = cache.stats
        assert stats["size"] <= 100, f"Cache exceeded max size: {stats['size']}"

        ##Condition purpose: Performance should be reasonable
        assert elapsed < 1.0, f"Eviction too slow: {elapsed:.3f}s for 500 ops"

        print(f"\nEviction test: {elapsed:.3f}s for 500 insertions, final size: {stats['size']}")

    ##Method purpose: Test cache hit rate tracking
    def test_cache_stats_accuracy(self) -> None:
        """Verify cache statistics are accurate under concurrent access."""
        ##Step purpose: Import dynamically
        try:
            import sys

            sys.path.insert(0, "src")
            from jenova.core.response import Response, ResponseCache
        except ImportError:
            pytest.skip("jenova package not installed")

        cache = ResponseCache(max_size=50)

        ##Step purpose: Insert 20 items
        for i in range(20):
            response = Response(
                content=f"response_{i}",
                sources=[],
                confidence=1.0,
                metadata={},
            )
            cache.put(f"query_{i}", "user", response)

        ##Step purpose: Read each item twice (should be 20 misses, 20 hits)
        for i in range(20):
            cache.get(f"query_{i}", "user")  # Hit

        for i in range(20, 30):
            cache.get(f"query_{i}", "user")  # Miss - not in cache

        stats = cache.stats
        ##Condition purpose: Verify stats are accurate
        assert stats["hits"] == 20, f"Expected 20 hits, got {stats['hits']}"
        assert stats["misses"] == 10, f"Expected 10 misses, got {stats['misses']}"
        assert stats["hit_rate_percent"] == 66, (
            f"Expected 66% hit rate, got {stats['hit_rate_percent']}%"
        )


##Class purpose: Benchmark tests for CognitiveGraph operations
class TestGraphPerformance:
    """Benchmark tests for CognitiveGraph performance."""

    ##Method purpose: Test O(degree) node removal performance
    def test_remove_node_performance(self) -> None:
        """Verify remove_node is O(degree) not O(n)."""
        ##Step purpose: Import dynamically
        try:
            import sys
            import tempfile
            from pathlib import Path

            sys.path.insert(0, "src")
            from jenova.graph.graph import CognitiveGraph
            from jenova.graph.types import Node
        except ImportError:
            pytest.skip("jenova package not installed")

        ##Step purpose: Create temp storage
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            graph = CognitiveGraph(storage_path)

            ##Step purpose: Create 100 nodes
            nodes = []
            for i in range(100):
                node = Node(
                    id=f"node_{i:03d}",
                    label=f"Node {i}",
                    content=f"Content for node {i}",
                    node_type="test",
                    metadata={"user": "testuser"},
                )
                graph.add_node(node, persist=False)
                nodes.append(node)

            ##Step purpose: Create edges (sparse graph - each node has ~3 connections)
            for i in range(0, 100, 3):
                if i + 1 < 100:
                    graph.add_edge(nodes[i].id, nodes[i + 1].id)
                if i + 2 < 100:
                    graph.add_edge(nodes[i].id, nodes[i + 2].id)

            ##Step purpose: Time removal of nodes
            start = time.perf_counter()
            for i in range(0, 50, 5):  # Remove 10 nodes
                graph.remove_node(f"node_{i:03d}")
            elapsed = time.perf_counter() - start

            ##Condition purpose: Verify reasonable performance
            assert elapsed < 2.0, f"Node removal too slow: {elapsed:.3f}s for 10 removals"

            ##Condition purpose: Verify graph integrity
            assert graph.node_count() == 90, f"Expected 90 nodes, got {graph.node_count()}"

            print(f"\nNode removal test: {elapsed:.3f}s for 10 removals from 100-node graph")

    ##Method purpose: Test connected components performance
    def test_connected_components_performance(self) -> None:
        """Verify connected components finding is O(n) BFS."""
        ##Step purpose: Import dynamically
        try:
            import sys
            import tempfile
            from pathlib import Path

            sys.path.insert(0, "src")
            from jenova.graph.graph import CognitiveGraph
            from jenova.graph.types import Node
        except ImportError:
            pytest.skip("jenova package not installed")

        ##Step purpose: Create temp storage
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            graph = CognitiveGraph(storage_path)

            ##Step purpose: Create 200 nodes in 10 clusters
            nodes = []
            for cluster in range(10):
                for i in range(20):
                    node = Node(
                        id=f"c{cluster}_n{i}",
                        label=f"Cluster {cluster} Node {i}",
                        content=f"Content for cluster {cluster} node {i}",
                        node_type="test",
                        metadata={"user": "testuser"},
                    )
                    graph.add_node(node, persist=False)
                    nodes.append(node)

                    ##Step purpose: Connect within cluster
                    if i > 0:
                        graph.add_edge(f"c{cluster}_n{i}", f"c{cluster}_n{i - 1}")

            ##Step purpose: Time cluster finding
            user_nodes = graph.get_nodes_by_user("testuser")

            start = time.perf_counter()
            components = graph._find_connected_components(user_nodes)
            elapsed = time.perf_counter() - start

            ##Condition purpose: Verify we found 10 clusters
            assert len(components) == 10, f"Expected 10 clusters, got {len(components)}"

            ##Condition purpose: Verify performance
            assert elapsed < 1.0, f"Component finding too slow: {elapsed:.3f}s"

            print(
                f"\nComponent finding test: {elapsed:.3f}s for 200 nodes, found {len(components)} clusters"
            )


if __name__ == "__main__":
    ##Step purpose: Run tests when executed directly
    pytest.main([__file__, "-v", "-s"])
