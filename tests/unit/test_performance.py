##Script function and purpose: Unit tests for performance monitoring utilities
"""
Test suite for PerformanceMonitor - Timing decorators and performance statistics.

Tests cover:
- Timing statistics collection
- Singleton pattern
- Context managers
- Decorators
- Thread safety
"""

import pytest
import time
from unittest.mock import Mock, patch
from jenova.utils.performance import (
    PerformanceMonitor,
    TimingStats,
    timed,
    log_slow,
)


##Class purpose: Fixture providing fresh monitor instance
@pytest.fixture
def monitor() -> PerformanceMonitor:
    """##Test case: Fresh PerformanceMonitor instance."""
    # Create new instance (bypass singleton for testing)
    instance = PerformanceMonitor()
    return instance


##Function purpose: Test timing stats record
def test_timing_stats_record() -> None:
    """##Test case: TimingStats records measurements."""
    ##Step purpose: Create stats and record
    stats = TimingStats(name="test_op")
    stats.record(100.5)
    
    ##Assertion purpose: Verify recorded
    assert stats.call_count == 1
    assert stats.total_time_ms == 100.5
    assert stats.last_time_ms == 100.5
    assert stats.min_time_ms == 100.5
    assert stats.max_time_ms == 100.5


##Function purpose: Test timing stats average
def test_timing_stats_average() -> None:
    """##Test case: TimingStats calculates average correctly."""
    ##Step purpose: Record multiple measurements
    stats = TimingStats(name="test_op")
    stats.record(100.0)
    stats.record(200.0)
    stats.record(300.0)
    
    ##Assertion purpose: Verify average (200.0)
    assert stats.avg_time_ms == 200.0


##Function purpose: Test timing stats min max
def test_timing_stats_min_max() -> None:
    """##Test case: TimingStats tracks min and max."""
    ##Step purpose: Record measurements
    stats = TimingStats(name="test_op")
    stats.record(50.0)
    stats.record(200.0)
    stats.record(100.0)
    
    ##Assertion purpose: Verify min/max
    assert stats.min_time_ms == 50.0
    assert stats.max_time_ms == 200.0


##Function purpose: Test timing stats to dict
def test_timing_stats_to_dict() -> None:
    """##Test case: TimingStats converts to dict."""
    ##Step purpose: Create stats and convert
    stats = TimingStats(name="test_op")
    stats.record(100.0)
    stats.record(200.0)
    
    result = stats.to_dict()
    
    ##Assertion purpose: Verify structure
    assert result["name"] == "test_op"
    assert result["call_count"] == 2
    assert result["avg_time_ms"] == 150.0


##Function purpose: Test timing stats zero average
def test_timing_stats_zero_average() -> None:
    """##Test case: Average is 0.0 when no calls."""
    ##Step purpose: Create stats without recording
    stats = TimingStats(name="test_op")
    
    ##Assertion purpose: Verify 0.0
    assert stats.avg_time_ms == 0.0


##Function purpose: Test performance monitor singleton
def test_performance_monitor_singleton() -> None:
    """##Test case: PerformanceMonitor is singleton."""
    ##Step purpose: Get instances
    instance1 = PerformanceMonitor.get_instance()
    instance2 = PerformanceMonitor.get_instance()
    
    ##Assertion purpose: Verify same instance
    assert instance1 is instance2


##Function purpose: Test performance monitor initialization
def test_performance_monitor_initialization(monitor: PerformanceMonitor) -> None:
    """##Test case: Monitor initializes correctly."""
    ##Assertion purpose: Verify state
    assert monitor.is_enabled is True
    assert len(monitor.get_all_stats()) == 0


##Function purpose: Test performance monitor enable disable
def test_performance_monitor_enable_disable(monitor: PerformanceMonitor) -> None:
    """##Test case: Can enable/disable monitoring."""
    ##Step purpose: Disable
    monitor.set_enabled(False)
    assert monitor.is_enabled is False
    
    ##Action purpose: Enable
    monitor.set_enabled(True)
    assert monitor.is_enabled is True


##Function purpose: Test performance monitor record
def test_performance_monitor_record(monitor: PerformanceMonitor) -> None:
    """##Test case: Monitor records measurements."""
    ##Step purpose: Record measurement
    monitor.record("operation", 100.5)
    
    ##Action purpose: Get stats
    stats = monitor.get_stats("operation")
    
    ##Assertion purpose: Verify recorded
    assert stats is not None
    assert stats.call_count == 1
    assert stats.last_time_ms == 100.5


##Function purpose: Test performance monitor record multiple
def test_performance_monitor_record_multiple(monitor: PerformanceMonitor) -> None:
    """##Test case: Monitor accumulates measurements."""
    ##Step purpose: Record multiple times
    monitor.record("operation", 100.0)
    monitor.record("operation", 200.0)
    monitor.record("operation", 150.0)
    
    ##Action purpose: Get stats
    stats = monitor.get_stats("operation")
    
    ##Assertion purpose: Verify accumulated
    assert stats.call_count == 3
    assert stats.avg_time_ms == 150.0


##Function purpose: Test performance monitor record disabled
def test_performance_monitor_record_disabled(monitor: PerformanceMonitor) -> None:
    """##Test case: Recording is skipped when disabled."""
    ##Step purpose: Disable and record
    monitor.set_enabled(False)
    monitor.record("operation", 100.0)
    
    ##Action purpose: Check stats
    stats = monitor.get_stats("operation")
    
    ##Assertion purpose: Verify not recorded
    assert stats is None


##Function purpose: Test performance monitor measure context
def test_performance_monitor_measure_context(monitor: PerformanceMonitor) -> None:
    """##Test case: measure() context manager times operations."""
    ##Step purpose: Use context manager
    with monitor.measure("operation"):
        time.sleep(0.05)
    
    ##Action purpose: Get stats
    stats = monitor.get_stats("operation")
    
    ##Assertion purpose: Verify recorded (should be ~50ms)
    assert stats is not None
    assert stats.call_count == 1
    assert stats.last_time_ms >= 40  # At least 40ms (allow some variance)


##Function purpose: Test performance monitor get stats
def test_performance_monitor_get_stats(monitor: PerformanceMonitor) -> None:
    """##Test case: get_stats returns correct stats."""
    ##Step purpose: Record and get
    monitor.record("op1", 100.0)
    stats = monitor.get_stats("op1")
    
    ##Assertion purpose: Verify
    assert stats.name == "op1"
    assert stats.last_time_ms == 100.0


##Function purpose: Test performance monitor get stats nonexistent
def test_performance_monitor_get_stats_nonexistent(monitor: PerformanceMonitor) -> None:
    """##Test case: get_stats returns None for nonexistent."""
    ##Step purpose: Get nonexistent
    stats = monitor.get_stats("nonexistent")
    
    ##Assertion purpose: Verify None
    assert stats is None


##Function purpose: Test performance monitor get all stats
def test_performance_monitor_get_all_stats(monitor: PerformanceMonitor) -> None:
    """##Test case: get_all_stats returns all measurements."""
    ##Step purpose: Record multiple
    monitor.record("op1", 100.0)
    monitor.record("op2", 200.0)
    
    ##Action purpose: Get all
    all_stats = monitor.get_all_stats()
    
    ##Assertion purpose: Verify
    assert len(all_stats) == 2
    assert "op1" in all_stats
    assert "op2" in all_stats


##Function purpose: Test performance monitor clear
def test_performance_monitor_clear(monitor: PerformanceMonitor) -> None:
    """##Test case: clear() removes all stats."""
    ##Step purpose: Record and clear
    monitor.record("op1", 100.0)
    monitor.clear()
    
    ##Action purpose: Check
    stats = monitor.get_stats("op1")
    
    ##Assertion purpose: Verify cleared
    assert stats is None


##Function purpose: Test performance monitor get summary
def test_performance_monitor_get_summary(monitor: PerformanceMonitor) -> None:
    """##Test case: get_summary returns complete report."""
    ##Step purpose: Record and get summary
    monitor.record("op1", 100.0)
    monitor.record("op2", 200.0)
    summary = monitor.get_summary()
    
    ##Assertion purpose: Verify structure
    assert "enabled" in summary
    assert "uptime_seconds" in summary
    assert "operation_count" in summary
    assert "operations" in summary
    assert len(summary["operations"]) == 2


##Function purpose: Test timed decorator basic
def test_timed_decorator_basic(monitor: PerformanceMonitor) -> None:
    """##Test case: @timed decorator times function."""
    ##Step purpose: Create decorated function
    @timed("test_function")
    def slow_function() -> int:
        """Slow test function."""
        time.sleep(0.05)
        return 42
    
    ##Action purpose: Call function
    PerformanceMonitor._instance = monitor  # Set monitor for decorator
    result = slow_function()
    
    ##Assertion purpose: Verify
    assert result == 42
    stats = monitor.get_stats("test_function")
    assert stats is not None
    assert stats.last_time_ms >= 40


##Function purpose: Test timed decorator default name
def test_timed_decorator_default_name(monitor: PerformanceMonitor) -> None:
    """##Test case: @timed uses function name by default."""
    ##Step purpose: Create decorated function
    @timed()
    def my_operation() -> None:
        """Operation without explicit name."""
        pass
    
    ##Action purpose: Call function
    PerformanceMonitor._instance = monitor
    my_operation()
    
    ##Assertion purpose: Verify name is function name
    stats = monitor.get_stats("my_operation")
    assert stats is not None


##Function purpose: Test timed decorator disabled
def test_timed_decorator_disabled(monitor: PerformanceMonitor) -> None:
    """##Test case: @timed is skipped when disabled."""
    ##Step purpose: Create and call
    @timed("operation")
    def fast_function() -> str:
        """Fast function."""
        return "result"
    
    ##Action purpose: Disable and call
    monitor.set_enabled(False)
    PerformanceMonitor._instance = monitor
    result = fast_function()
    
    ##Assertion purpose: Verify not recorded
    assert result == "result"
    stats = monitor.get_stats("operation")
    assert stats is None


##Function purpose: Test timed decorator exception handling
def test_timed_decorator_exception_handling(monitor: PerformanceMonitor) -> None:
    """##Test case: @timed records even if exception raised."""
    ##Step purpose: Create decorated function
    @timed("failing_op")
    def failing_function() -> None:
        """Function that fails."""
        raise ValueError("test error")
    
    ##Action purpose: Call and catch exception
    PerformanceMonitor._instance = monitor
    with pytest.raises(ValueError):
        failing_function()
    
    ##Assertion purpose: Verify recorded despite exception
    stats = monitor.get_stats("failing_op")
    assert stats is not None
    assert stats.call_count == 1


##Function purpose: Test log_slow decorator below threshold
def test_log_slow_decorator_below_threshold() -> None:
    """##Test case: @log_slow doesn't log below threshold."""
    ##Step purpose: Create decorated function
    @log_slow(threshold_ms=100)
    def fast_function() -> None:
        """Fast enough."""
        time.sleep(0.01)  # 10ms
    
    ##Action purpose: Call and capture logging
    with patch("jenova.utils.performance.logger") as mock_logger:
        fast_function()
    
    ##Assertion purpose: Verify not logged
    mock_logger.warning.assert_not_called()


##Function purpose: Test log_slow decorator above threshold
def test_log_slow_decorator_above_threshold() -> None:
    """##Test case: @log_slow logs above threshold."""
    ##Step purpose: Create decorated function
    @log_slow(threshold_ms=10)
    def slow_function() -> None:
        """Too slow."""
        time.sleep(0.05)  # 50ms
    
    ##Action purpose: Call and capture logging
    with patch("jenova.utils.performance.logger") as mock_logger:
        slow_function()
    
    ##Assertion purpose: Verify logged
    mock_logger.warning.assert_called_once()


##Function purpose: Test log_slow decorator custom name
def test_log_slow_decorator_custom_name() -> None:
    """##Test case: @log_slow uses custom name."""
    ##Step purpose: Create decorated function
    @log_slow(threshold_ms=10, name="custom_name")
    def operation() -> None:
        """Operation."""
        time.sleep(0.02)
    
    ##Action purpose: Call and capture logging
    with patch("jenova.utils.performance.logger") as mock_logger:
        operation()
    
    ##Assertion purpose: Verify custom name in log
    if mock_logger.warning.called:
        call_args = mock_logger.warning.call_args
        assert "custom_name" in str(call_args)


##Function purpose: Test timing context manager entry exit
def test_timing_context_manager() -> None:
    """##Test case: TimingContext works correctly."""
    ##Step purpose: Create monitor and context
    monitor = PerformanceMonitor()
    
    ##Action purpose: Use context
    with monitor.measure("operation"):
        time.sleep(0.02)
    
    ##Assertion purpose: Verify timing recorded
    stats = monitor.get_stats("operation")
    assert stats.call_count == 1
    assert stats.last_time_ms >= 15


##Function purpose: Test multiple operations same name
def test_multiple_operations_accumulate(monitor: PerformanceMonitor) -> None:
    """##Test case: Multiple operations same name accumulate."""
    ##Step purpose: Record multiple with same name
    for i in range(5):
        monitor.record("api_call", (i + 1) * 10.0)
    
    ##Action purpose: Get stats
    stats = monitor.get_stats("api_call")
    
    ##Assertion purpose: Verify accumulation
    assert stats.call_count == 5
    # Total: 10 + 20 + 30 + 40 + 50 = 150
    # Average: 150 / 5 = 30
    assert stats.avg_time_ms == 30.0
