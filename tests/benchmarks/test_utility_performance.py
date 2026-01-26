##Script function and purpose: Performance benchmarks for 6 utility modules
"""
Performance Benchmarks for Utility Modules

Benchmark suite for profiling performance of:
- core/scheduler.py - Cognitive task scheduling
- graph/proactive.py - Proactive suggestion generation
- utils/cache.py - TTL/LRU cache operations
- utils/performance.py - Performance monitoring overhead
- utils/grammar.py - Grammar loading and caching
- tools.py - Shell command execution utilities

Usage:
    pytest tests/benchmarks/test_utility_performance.py -v
    pytest tests/benchmarks/test_utility_performance.py -v --benchmark-only
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import pytest

from jenova.core.scheduler import (
    CognitiveScheduler,
    SchedulerConfig,
    TaskType,
)
from jenova.graph.proactive import (
    ProactiveConfig,
    ProactiveEngine,
    SuggestionCategory,
)
from jenova.tools import (
    command_exists,
    format_datetime_for_display,
    get_current_date,
    get_current_datetime,
    get_current_time,
    get_system_info,
)
from jenova.utils.cache import TTLCache
from jenova.utils.grammar import GrammarLoader
from jenova.utils.performance import (
    PerformanceMonitor,
    timed,
)


##Class purpose: Mock task executor for scheduler benchmarks
class MockTaskExecutor:
    """Mock task executor for scheduler benchmarks."""

    ##Method purpose: Execute a task
    def execute_task(self, task_type: TaskType, username: str) -> bool:
        """Execute a task (mock)."""
        return True


##Class purpose: Mock graph protocol for proactive benchmarks
class MockGraph:
    """Mock graph for proactive engine benchmarks."""

    ##Method purpose: Get nodes by user
    def get_nodes_by_user(self, username: str) -> list[dict[str, object]]:
        """Get nodes for user."""
        return [
            {"id": f"node_{i}", "type": "insight", "content": f"Content {i}"} for i in range(100)
        ]

    ##Method purpose: Search nodes
    def search(self, query: str, username: str, limit: int = 10) -> list[dict[str, object]]:
        """Search nodes."""
        return self.get_nodes_by_user(username)[:limit]

    ##Method purpose: Get node by ID
    def get_node(self, node_id: str) -> dict[str, object] | None:
        """Get node by ID."""
        return {"id": node_id, "type": "insight", "content": "Test content"}


##Class purpose: Mock LLM protocol for proactive benchmarks
class MockLLM:
    """Mock LLM for proactive engine benchmarks."""

    ##Method purpose: Generate text
    def generate(self, prompt: str) -> str:
        """Generate text."""
        return "Generated response"


##Fixture purpose: Scheduler for benchmarks
@pytest.fixture
def scheduler() -> CognitiveScheduler:
    """Create scheduler for benchmarks."""
    config = SchedulerConfig()
    executor = MockTaskExecutor()
    return CognitiveScheduler(config, executor)


##Fixture purpose: Proactive engine for benchmarks
@pytest.fixture
def proactive_engine() -> ProactiveEngine:
    """Create proactive engine for benchmarks."""
    config = ProactiveConfig()
    graph = MockGraph()
    llm = MockLLM()
    return ProactiveEngine(config, graph, llm)


##Fixture purpose: TTL cache for benchmarks
@pytest.fixture
def ttl_cache() -> TTLCache[str, str]:
    """Create TTL cache for benchmarks."""
    return TTLCache(max_size=1000, default_ttl=300)


##Fixture purpose: Performance monitor for benchmarks
@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Create performance monitor for benchmarks."""
    # Reset singleton for clean testing
    PerformanceMonitor._instance = None
    return PerformanceMonitor.get_instance()


##Fixture purpose: Grammar loader for benchmarks
@pytest.fixture
def grammar_loader() -> GrammarLoader:
    """Create grammar loader for benchmarks."""
    return GrammarLoader()


##Class purpose: Scheduler performance benchmarks
class TestSchedulerPerformance:
    """Benchmark tests for CognitiveScheduler."""

    ##Test purpose: Benchmark turn completion
    def test_turn_completion_performance(self, scheduler: CognitiveScheduler) -> None:
        """Turn completion should be fast."""
        start = time.perf_counter()
        for _ in range(100):
            scheduler.on_turn_complete("user1")
        elapsed = time.perf_counter() - start

        ##Assert: 100 turns should be very fast
        assert elapsed < 0.1, f"100 turns took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark due task calculation
    def test_due_tasks_calculation_performance(self, scheduler: CognitiveScheduler) -> None:
        """Due task calculation should be fast."""
        start = time.perf_counter()
        for _ in range(1000):
            scheduler._get_due_tasks(0)
        elapsed = time.perf_counter() - start

        ##Assert: 1000 calculations should be fast
        assert elapsed < 0.1, f"1000 calculations took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark priority sorting
    def test_priority_sorting_performance(self, scheduler: CognitiveScheduler) -> None:
        """Priority sorting should be fast with small task count."""
        # Force multiple due tasks
        for task in scheduler._tasks:
            task.turns_since_last = task.schedule.interval

        start = time.perf_counter()
        for _ in range(100):
            scheduler._get_due_tasks(0)
        elapsed = time.perf_counter() - start

        ##Assert: Should be fast even with sorting
        assert elapsed < 0.05, f"100 sorts took {elapsed:.3f}s (expected <0.05s)"

    ##Test purpose: Benchmark acceleration logic
    def test_acceleration_logic_performance(self, scheduler: CognitiveScheduler) -> None:
        """Acceleration logic should be fast."""
        start = time.perf_counter()
        for unverified in range(0, 20):
            scheduler._get_due_tasks(unverified)
        elapsed = time.perf_counter() - start

        ##Assert: Should be very fast
        assert elapsed < 0.01, f"20 acceleration checks took {elapsed:.3f}s (expected <0.01s)"


##Class purpose: Proactive engine performance benchmarks
class TestProactivePerformance:
    """Benchmark tests for ProactiveEngine."""

    ##Test purpose: Benchmark single suggestion generation
    def test_single_suggestion_performance(self, proactive_engine: ProactiveEngine) -> None:
        """Single suggestion generation should be reasonable."""
        start = time.perf_counter()
        for _ in range(10):
            proactive_engine.get_suggestion("user1")
        elapsed = time.perf_counter() - start

        ##Assert: 10 suggestions should be reasonable
        assert elapsed < 1.0, f"10 suggestions took {elapsed:.3f}s (expected <1.0s)"

    ##Test purpose: Benchmark category selection
    def test_category_selection_performance(self, proactive_engine: ProactiveEngine) -> None:
        """Category selection should be fast."""
        start = time.perf_counter()
        for _ in range(1000):
            proactive_engine._get_next_category()
        elapsed = time.perf_counter() - start

        ##Assert: 1000 selections should be fast
        assert elapsed < 0.1, f"1000 selections took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark cooldown checking
    def test_cooldown_checking_performance(self, proactive_engine: ProactiveEngine) -> None:
        """Cooldown checking should be fast."""
        start = time.perf_counter()
        for category in SuggestionCategory:
            for _ in range(100):
                proactive_engine._is_on_cooldown(category)
        elapsed = time.perf_counter() - start

        ##Assert: Should be very fast
        assert elapsed < 0.01, f"500 cooldown checks took {elapsed:.3f}s (expected <0.01s)"

    ##Test purpose: Benchmark engagement tracking
    def test_engagement_tracking_performance(self, proactive_engine: ProactiveEngine) -> None:
        """Engagement tracking should be fast."""
        start = time.perf_counter()
        for category in SuggestionCategory:
            for _ in range(100):
                proactive_engine.record_acceptance(category)
        elapsed = time.perf_counter() - start

        ##Assert: Should be very fast
        assert elapsed < 0.01, f"500 acceptance records took {elapsed:.3f}s (expected <0.01s)"


##Class purpose: Cache performance benchmarks
class TestCachePerformance:
    """Benchmark tests for TTLCache."""

    ##Test purpose: Benchmark cache hit performance
    def test_cache_hit_performance(self, ttl_cache: TTLCache[str, str]) -> None:
        """Cache hits should be O(1)."""
        # Pre-populate cache
        for i in range(100):
            ttl_cache.set(f"key_{i}", f"value_{i}")

        start = time.perf_counter()
        for _ in range(10000):
            ttl_cache.get("key_50")
        elapsed = time.perf_counter() - start

        ##Assert: 10000 hits should be very fast
        assert elapsed < 0.5, f"10000 hits took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark cache miss performance
    def test_cache_miss_performance(self, ttl_cache: TTLCache[str, str]) -> None:
        """Cache misses should be O(1)."""
        start = time.perf_counter()
        for _ in range(10000):
            ttl_cache.get("nonexistent_key")
        elapsed = time.perf_counter() - start

        ##Assert: 10000 misses should be fast
        assert elapsed < 0.5, f"10000 misses took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark cache set performance
    def test_cache_set_performance(self, ttl_cache: TTLCache[str, str]) -> None:
        """Cache sets should be O(1) amortized."""
        start = time.perf_counter()
        for i in range(10000):
            ttl_cache.set(f"key_{i}", f"value_{i}")
        elapsed = time.perf_counter() - start

        ##Assert: 10000 sets should be reasonable
        assert elapsed < 2.0, f"10000 sets took {elapsed:.3f}s (expected <2.0s)"

    ##Test purpose: Benchmark LRU eviction performance
    def test_lru_eviction_performance(self) -> None:
        """LRU eviction should be reasonable."""
        cache = TTLCache(max_size=100, default_ttl=None)

        # Fill cache to capacity
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        start = time.perf_counter()
        # Add more to trigger evictions
        for i in range(100, 1000):
            cache.set(f"key_{i}", f"value_{i}")
        elapsed = time.perf_counter() - start

        ##Assert: 900 evictions should be reasonable
        assert elapsed < 1.0, f"900 evictions took {elapsed:.3f}s (expected <1.0s)"

    ##Test purpose: Benchmark TTL expiration checking
    def test_ttl_expiration_performance(self, ttl_cache: TTLCache[str, str]) -> None:
        """TTL expiration checking should be fast."""
        # Add entries with short TTL
        for i in range(100):
            ttl_cache.set(f"key_{i}", f"value_{i}", ttl=1)

        # Wait for expiration
        time.sleep(1.1)

        start = time.perf_counter()
        for i in range(100):
            ttl_cache.get(f"key_{i}")
        elapsed = time.perf_counter() - start

        ##Assert: Expiration checking should be fast
        assert elapsed < 0.1, f"100 expiration checks took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark concurrent access
    def test_concurrent_cache_access(self, ttl_cache: TTLCache[str, str]) -> None:
        """Concurrent access should be thread-safe."""
        # Pre-populate
        for i in range(100):
            ttl_cache.set(f"key_{i}", f"value_{i}")

        errors: list[str] = []
        results: list[bool] = []

        ##Function purpose: Worker thread
        def worker(thread_id: int) -> None:
            try:
                for i in range(100):
                    key = f"key_{(i + thread_id) % 100}"
                    value = ttl_cache.get(key)
                    results.append(value is not None)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        ##Assert: No errors and reasonable performance
        assert not errors, f"Thread errors: {errors}"
        assert elapsed < 1.0, f"Concurrent access took {elapsed:.3f}s (expected <1.0s)"


##Class purpose: Performance monitor benchmarks
class TestPerformanceMonitorPerformance:
    """Benchmark tests for PerformanceMonitor."""

    ##Test purpose: Benchmark decorator overhead
    def test_decorator_overhead(self, performance_monitor: PerformanceMonitor) -> None:
        """Decorator overhead should be minimal."""

        ##Function purpose: Undecorated function
        def undecorated_func() -> int:
            return 42

        ##Function purpose: Decorated function
        @timed("test_operation")
        def decorated_func() -> int:
            return 42

        # Time undecorated
        start = time.perf_counter()
        for _ in range(10000):
            undecorated_func()
        undecorated_time = time.perf_counter() - start

        # Time decorated
        start = time.perf_counter()
        for _ in range(10000):
            decorated_func()
        decorated_time = time.perf_counter() - start

        overhead = decorated_time - undecorated_time

        ##Assert: Overhead should be minimal (<10% of base time)
        assert overhead < undecorated_time * 0.1, f"Overhead {overhead:.3f}s is too high"

    ##Test purpose: Benchmark statistics recording
    def test_statistics_recording_performance(
        self, performance_monitor: PerformanceMonitor
    ) -> None:
        """Statistics recording should be fast."""
        start = time.perf_counter()
        for i in range(10000):
            performance_monitor.record(f"op_{i % 100}", float(i))
        elapsed = time.perf_counter() - start

        ##Assert: 10000 records should be fast
        assert elapsed < 0.5, f"10000 records took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark context manager performance
    def test_context_manager_performance(self, performance_monitor: PerformanceMonitor) -> None:
        """Context manager should have minimal overhead."""
        start = time.perf_counter()
        for _ in range(10000):
            with performance_monitor.measure("test_op"):
                pass
        elapsed = time.perf_counter() - start

        ##Assert: 10000 context managers should be fast
        assert elapsed < 0.5, f"10000 contexts took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark singleton access
    def test_singleton_access_performance(self) -> None:
        """Singleton access should be fast."""
        start = time.perf_counter()
        for _ in range(10000):
            PerformanceMonitor.get_instance()
        elapsed = time.perf_counter() - start

        ##Assert: 10000 accesses should be very fast
        assert elapsed < 0.1, f"10000 accesses took {elapsed:.3f}s (expected <0.1s)"


##Class purpose: Grammar loader benchmarks
class TestGrammarLoaderPerformance:
    """Benchmark tests for GrammarLoader."""

    ##Test purpose: Benchmark grammar loading (first load)
    def test_grammar_loading_performance(self, grammar_loader: GrammarLoader) -> None:
        """Grammar loading should be reasonable."""
        # Skip if llama-cpp-python not available
        if not grammar_loader.is_available:
            pytest.skip("llama-cpp-python not available")

        grammar_loader.clear_cache()

        start = time.perf_counter()
        grammar = grammar_loader.load_json_grammar()
        elapsed = time.perf_counter() - start

        ##Assert: First load should be reasonable
        assert grammar is not None
        assert elapsed < 1.0, f"Grammar load took {elapsed:.3f}s (expected <1.0s)"

    ##Test purpose: Benchmark cached grammar loading
    def test_cached_grammar_performance(self, grammar_loader: GrammarLoader) -> None:
        """Cached grammar loading should be O(1)."""
        if not grammar_loader.is_available:
            pytest.skip("llama-cpp-python not available")

        # Load once to cache
        grammar_loader.load_json_grammar()

        start = time.perf_counter()
        for _ in range(1000):
            grammar_loader.load_json_grammar()
        elapsed = time.perf_counter() - start

        ##Assert: 1000 cached loads should be very fast
        assert elapsed < 0.1, f"1000 cached loads took {elapsed:.3f}s (expected <0.1s)"

    ##Test purpose: Benchmark availability checking
    def test_availability_check_performance(self, grammar_loader: GrammarLoader) -> None:
        """Availability checking should be fast."""
        start = time.perf_counter()
        for _ in range(10000):
            _ = grammar_loader.is_available
        elapsed = time.perf_counter() - start

        ##Assert: 10000 checks should be very fast
        assert elapsed < 0.01, f"10000 checks took {elapsed:.3f}s (expected <0.01s)"


##Class purpose: Tools performance benchmarks
class TestToolsPerformance:
    """Benchmark tests for tools utilities."""

    ##Test purpose: Benchmark datetime formatting
    def test_datetime_formatting_performance(self) -> None:
        """Datetime formatting should be fast."""
        start = time.perf_counter()
        for _ in range(10000):
            get_current_datetime()
        elapsed = time.perf_counter() - start

        ##Assert: 10000 formats should be fast
        assert elapsed < 0.5, f"10000 formats took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark date formatting
    def test_date_formatting_performance(self) -> None:
        """Date formatting should be fast."""
        start = time.perf_counter()
        for _ in range(10000):
            get_current_date()
        elapsed = time.perf_counter() - start

        ##Assert: 10000 formats should be fast
        assert elapsed < 0.5, f"10000 formats took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark time formatting
    def test_time_formatting_performance(self) -> None:
        """Time formatting should be fast."""
        start = time.perf_counter()
        for _ in range(10000):
            get_current_time()
        elapsed = time.perf_counter() - start

        ##Assert: 10000 formats should be fast
        assert elapsed < 0.5, f"10000 formats took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark display formatting
    def test_display_formatting_performance(self) -> None:
        """Display formatting should be fast."""
        dt = datetime.now()

        start = time.perf_counter()
        for _ in range(10000):
            format_datetime_for_display(dt)
        elapsed = time.perf_counter() - start

        ##Assert: 10000 formats should be fast
        assert elapsed < 0.5, f"10000 formats took {elapsed:.3f}s (expected <0.5s)"

    ##Test purpose: Benchmark system info collection
    def test_system_info_performance(self) -> None:
        """System info collection should be fast."""
        start = time.perf_counter()
        for _ in range(1000):
            get_system_info()
        elapsed = time.perf_counter() - start

        ##Assert: 1000 collections should be reasonable
        assert elapsed < 1.0, f"1000 collections took {elapsed:.3f}s (expected <1.0s)"

    ##Test purpose: Benchmark command existence checking
    def test_command_exists_performance(self) -> None:
        """Command existence checking should be reasonable."""
        start = time.perf_counter()
        for cmd in ["ls", "echo", "cat", "pwd", "date"]:
            for _ in range(10):
                command_exists(cmd)
        elapsed = time.perf_counter() - start

        ##Assert: 50 checks should be reasonable (external process)
        assert elapsed < 5.0, f"50 checks took {elapsed:.3f}s (expected <5.0s)"


##Function purpose: Run benchmarks and generate report
def run_benchmarks() -> None:
    """Run all benchmarks and print summary."""
    print("JENOVA Utility Performance Benchmarks")
    print("=" * 50)
    print("\nRun with: pytest tests/benchmarks/test_utility_performance.py -v")
    print("\nFor detailed timing: pytest --benchmark-only")
    print("For JSON output: pytest --benchmark-json=results.json")


if __name__ == "__main__":
    run_benchmarks()
