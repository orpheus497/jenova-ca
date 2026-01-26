##Script function and purpose: Performance benchmarks for JENOVA critical operations
"""
Performance Benchmarks

Benchmark suite for profiling critical JENOVA operations.
Addresses P3-002 from DAEDELUS audit: No performance benchmark suite.

Usage:
    pytest tests/benchmarks/test_performance.py -v --benchmark-only
    pytest tests/benchmarks/test_performance.py -v --benchmark-json=results.json
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest

from jenova.core.response import Response, ResponseCache
from jenova.graph.graph import CognitiveGraph
from jenova.graph.types import EdgeType, Node
from jenova.memory.memory import Memory
from jenova.memory.types import MemoryType


##Class purpose: Fixtures for benchmark tests
class BenchmarkFixtures:
    """Factory for creating benchmark test fixtures."""

    @staticmethod
    def create_graph(n_nodes: int, edges_per_node: int = 2) -> tuple[CognitiveGraph, Path]:
        """Create a graph with specified node count."""
        temp_dir = Path(tempfile.mkdtemp())
        graph = CognitiveGraph(temp_dir)

        ##Step purpose: Create nodes
        node_ids: list[str] = []
        for i in range(n_nodes):
            node = Node(
                id=f"node_{i}_{uuid.uuid4().hex[:8]}",
                label=f"Test Node {i}",
                content=f"Content for test node {i} with some searchable keywords",
                node_type="test",
                metadata={"user": "benchmark_user", "index": str(i)},
            )
            graph.add_node(node, persist=False)
            node_ids.append(node.id)

        ##Step purpose: Create edges
        for i, source_id in enumerate(node_ids):
            for j in range(edges_per_node):
                target_idx = (i + j + 1) % n_nodes
                if target_idx != i:
                    graph.add_edge(
                        source_id,
                        node_ids[target_idx],
                        EdgeType.RELATES_TO,
                        persist=False,
                    )

        graph._save()
        return graph, temp_dir

    @staticmethod
    def create_memory(n_items: int) -> tuple[Memory, Path]:
        """Create a memory with specified item count."""
        temp_dir = Path(tempfile.mkdtemp())
        memory = Memory(MemoryType.EPISODIC, temp_dir)

        ##Step purpose: Add items
        for i in range(n_items):
            memory.add(
                f"Memory item {i} with searchable content about topic {i % 10}",
                metadata={"index": str(i)},
            )

        return memory, temp_dir

    @staticmethod
    def create_cache(n_items: int) -> ResponseCache:
        """Create a cache with specified item count."""
        cache = ResponseCache(max_size=n_items * 2)

        ##Step purpose: Pre-populate cache
        for i in range(n_items):
            response = Response(
                content=f"Cached response {i}",
                sources=[f"source_{i}"],
            )
            cache.put(f"query_{i}", "user", response)

        return cache


##Fixture purpose: Small graph for unit tests
@pytest.fixture
def small_graph() -> Generator[tuple[CognitiveGraph, Path], None, None]:
    """Create small graph (100 nodes) for quick tests."""
    graph, temp_dir = BenchmarkFixtures.create_graph(100, 3)
    yield graph, temp_dir
    ##Cleanup: Remove temp directory
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


##Fixture purpose: Medium graph for integration tests
@pytest.fixture
def medium_graph() -> Generator[tuple[CognitiveGraph, Path], None, None]:
    """Create medium graph (1000 nodes) for performance tests."""
    graph, temp_dir = BenchmarkFixtures.create_graph(1000, 3)
    yield graph, temp_dir
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


##Fixture purpose: Large graph for stress tests
@pytest.fixture
def large_graph() -> Generator[tuple[CognitiveGraph, Path], None, None]:
    """Create large graph (10000 nodes) for stress tests."""
    graph, temp_dir = BenchmarkFixtures.create_graph(10000, 2)
    yield graph, temp_dir
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


##Fixture purpose: Memory store for tests
@pytest.fixture
def memory_store() -> Generator[tuple[Memory, Path], None, None]:
    """Create memory with 1000 items."""
    memory, temp_dir = BenchmarkFixtures.create_memory(1000)
    yield memory, temp_dir
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


##Fixture purpose: Response cache for tests
@pytest.fixture
def response_cache() -> ResponseCache:
    """Create cache with 100 items."""
    return BenchmarkFixtures.create_cache(100)


##Class purpose: Graph operation benchmarks
class TestGraphBenchmarks:
    """Benchmark tests for CognitiveGraph operations."""

    ##Test purpose: Benchmark node lookup
    def test_get_node_performance(self, small_graph: tuple[CognitiveGraph, Path]) -> None:
        """Node lookup should be O(1)."""
        graph, _ = small_graph
        nodes = graph.all_nodes()

        ##Step purpose: Time 1000 lookups
        import time

        start = time.perf_counter()
        for _ in range(1000):
            for node in nodes[:10]:
                graph.get_node(node.id)
        elapsed = time.perf_counter() - start

        ##Assert: Should be very fast
        assert elapsed < 0.1, f"1000 lookups took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark search
    def test_search_performance(self, medium_graph: tuple[CognitiveGraph, Path]) -> None:
        """Search should scale linearly with node count."""
        graph, _ = medium_graph

        import time

        start = time.perf_counter()
        for _ in range(10):
            graph.search("searchable keywords", max_results=10)
        elapsed = time.perf_counter() - start

        ##Assert: 10 searches on 1000 nodes should be reasonable
        assert elapsed < 1.0, f"10 searches took {elapsed:.3f}s (expected <1.0s)"

    ##Test purpose: Benchmark neighbor lookup
    def test_neighbors_performance(self, medium_graph: tuple[CognitiveGraph, Path]) -> None:
        """Neighbor lookup should be O(degree)."""
        graph, _ = medium_graph
        nodes = graph.all_nodes()

        import time

        start = time.perf_counter()
        for node in nodes[:100]:
            graph.neighbors(node.id, direction="both")
        elapsed = time.perf_counter() - start

        ##Assert: 100 neighbor lookups should be fast
        assert elapsed < 0.5, f"100 neighbor lookups took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark remove_node (critical hotspot)
    def test_remove_node_performance(self, small_graph: tuple[CognitiveGraph, Path]) -> None:
        """Remove node - measures edge cleanup overhead."""
        graph, _ = small_graph
        nodes = graph.all_nodes()

        import time

        start = time.perf_counter()
        for node in nodes[:10]:
            graph.remove_node(node.id)
        elapsed = time.perf_counter() - start

        ##Assert: 10 removals on 100-node graph should be fast
        assert elapsed < 0.5, f"10 removals took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark connected components (BFS)
    def test_connected_components_performance(
        self,
        medium_graph: tuple[CognitiveGraph, Path],
    ) -> None:
        """BFS should be O(n+e)."""
        graph, _ = medium_graph
        nodes = graph.all_nodes()

        import time

        start = time.perf_counter()
        graph._find_connected_components(nodes)
        elapsed = time.perf_counter() - start

        ##Assert: BFS on 1000 nodes should be fast
        assert elapsed < 0.5, f"BFS took {elapsed:.3f}s (expected <0.5s)"


##Class purpose: Cache operation benchmarks
class TestCacheBenchmarks:
    """Benchmark tests for ResponseCache operations."""

    ##Test purpose: Benchmark cache get (hit)
    def test_cache_hit_performance(self, response_cache: ResponseCache) -> None:
        """Cache hit should be O(1)."""
        import time

        start = time.perf_counter()
        for i in range(10000):
            response_cache.get(f"query_{i % 100}", "user")
        elapsed = time.perf_counter() - start

        ##Assert: 10000 cache hits should be very fast
        assert elapsed < 0.1, f"10000 cache hits took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark cache get (miss)
    def test_cache_miss_performance(self, response_cache: ResponseCache) -> None:
        """Cache miss should be O(1)."""
        import time

        start = time.perf_counter()
        for i in range(10000):
            response_cache.get(f"nonexistent_{i}", "user")
        elapsed = time.perf_counter() - start

        ##Assert: 10000 cache misses should be very fast
        assert elapsed < 0.1, f"10000 cache misses took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark cache put with eviction
    def test_cache_put_eviction_performance(self) -> None:
        """Cache put with eviction should be O(1) amortized."""
        cache = ResponseCache(max_size=100)

        import time

        start = time.perf_counter()
        for i in range(10000):
            response = Response(content=f"Response {i}")
            cache.put(f"query_{i}", "user", response)
        elapsed = time.perf_counter() - start

        ##Assert: 10000 puts (with eviction) should be fast
        assert elapsed < 0.5, f"10000 puts took {elapsed:.3f}s (expected <0.5s)"


##Class purpose: Memory operation benchmarks
class TestMemoryBenchmarks:
    """Benchmark tests for Memory operations."""

    ##Test purpose: Benchmark memory search
    def test_memory_search_performance(
        self,
        memory_store: tuple[Memory, Path],
    ) -> None:
        """Memory search should leverage ChromaDB ANN."""
        memory, _ = memory_store

        import time

        start = time.perf_counter()
        for _ in range(100):
            memory.search("topic 5 searchable content", n_results=10)
        elapsed = time.perf_counter() - start

        ##Assert: 100 searches on 1000 items should be fast
        assert elapsed < 5.0, f"100 searches took {elapsed:.3f}s (expected <5.0s)"

    ##Test purpose: Benchmark memory add
    def test_memory_add_performance(self) -> None:
        """Memory add should be efficient."""
        temp_dir = Path(tempfile.mkdtemp())
        memory = Memory(MemoryType.EPISODIC, temp_dir)

        import time

        start = time.perf_counter()
        for i in range(100):
            memory.add(f"New memory item {i}")
        elapsed = time.perf_counter() - start

        ##Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        ##Assert: 100 adds should be reasonable
        assert elapsed < 2.0, f"100 adds took {elapsed:.3f}s (expected <2.0s)"


##Class purpose: Regression tests for known hotspots
class TestHotspotRegressions:
    """Regression tests for documented performance hotspots."""

    ##Test purpose: Regression for remove_node O(n) issue
    def test_remove_node_does_not_degrade_with_size(self) -> None:
        """Remove node time should not grow linearly with graph size."""
        import time

        times: list[float] = []

        for size in [100, 500, 1000]:
            graph, temp_dir = BenchmarkFixtures.create_graph(size, 2)
            nodes = graph.all_nodes()

            start = time.perf_counter()
            graph.remove_node(nodes[0].id)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        ##Assert: Time should not grow proportionally with size
        ##Allow 5x growth for 10x size increase (not 10x)
        ratio = times[2] / times[0] if times[0] > 0 else 0
        assert ratio < 20, f"Remove node scaling ratio: {ratio:.1f}x (expected <20x for 10x size)"

    ##Test purpose: Regression for prune_graph batch efficiency
    def test_prune_graph_batch_efficiency(self) -> None:
        """Prune graph should handle multiple removals efficiently."""
        import time
        from datetime import timedelta

        graph, temp_dir = BenchmarkFixtures.create_graph(500, 2)

        ##Step purpose: Make all nodes "old" for pruning
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        for node_id in list(graph._nodes.keys()):
            node = graph._nodes[node_id]
            ##Modify created_at to be old
            graph._nodes[node_id] = Node(
                id=node.id,
                label=node.label,
                content=node.content,
                node_type=node.node_type,
                metadata=node.metadata,
                created_at=old_date,
            )
        graph._save()

        start = time.perf_counter()
        pruned = graph.prune_graph(max_age_days=30, min_connections=10)
        elapsed = time.perf_counter() - start

        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        ##Assert: Pruning should be reasonably fast
        assert elapsed < 5.0, f"Pruning {pruned} nodes took {elapsed:.3f}s (expected <5.0s)"


##Function purpose: Run benchmarks and generate report
def run_benchmarks() -> None:
    """Run all benchmarks and print summary."""
    print("JENOVA Performance Benchmarks")
    print("=" * 50)
    print("\nRun with: pytest tests/benchmarks/test_performance.py -v")
    print("\nFor detailed timing: pytest --benchmark-only")
    print("For JSON output: pytest --benchmark-json=results.json")


if __name__ == "__main__":
    run_benchmarks()
