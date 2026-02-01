##Script function and purpose: Performance monitoring utilities with timing decorators
"""
Performance Monitor - Timing decorators and performance statistics.

This module provides infrastructure for monitoring performance of
JENOVA operations, including timing decorators, statistics collection,
and performance diagnostics.

Reference: .devdocs/resources/src/jenova/utils/performance_monitor.py
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import ParamSpec, TypeVar

import structlog

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)

##Step purpose: Define generic types for decorators
P = ParamSpec("P")
R = TypeVar("R")


##Class purpose: Timing statistics for an operation
@dataclass
class TimingStats:
    """Statistics for a timed operation.

    Attributes:
        name: Name of the operation
        call_count: Number of times called
        total_time_ms: Total time in milliseconds
        min_time_ms: Minimum time in milliseconds
        max_time_ms: Maximum time in milliseconds
        last_time_ms: Last call time in milliseconds
    """

    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    last_time_ms: float = 0.0

    ##Method purpose: Record a timing measurement
    def record(self, duration_ms: float) -> None:
        """Record a timing measurement.

        Args:
            duration_ms: Duration in milliseconds
        """
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.last_time_ms = duration_ms

        ##Condition purpose: Update min/max
        if duration_ms < self.min_time_ms:
            self.min_time_ms = duration_ms
        if duration_ms > self.max_time_ms:
            self.max_time_ms = duration_ms

    ##Method purpose: Get average time
    @property
    def avg_time_ms(self) -> float:
        """Get average time in milliseconds."""
        ##Condition purpose: Avoid division by zero
        if self.call_count == 0:
            return 0.0
        return self.total_time_ms / self.call_count

    ##Method purpose: Convert to dict for serialization
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2) if self.min_time_ms != float("inf") else 0.0,
            "max_time_ms": round(self.max_time_ms, 2),
            "last_time_ms": round(self.last_time_ms, 2),
        }


##Class purpose: Global performance monitor singleton
class PerformanceMonitor:
    """Global performance monitor for collecting timing statistics.

    Thread-safe singleton that collects timing statistics for operations
    across the application.

    Example:
        >>> monitor = PerformanceMonitor.get_instance()
        >>> with monitor.measure("my_operation"):
        ...     do_something()
        >>> stats = monitor.get_stats("my_operation")
    """

    ##Step purpose: Singleton instance
    _instance: "PerformanceMonitor | None" = None
    _lock = threading.Lock()

    ##Method purpose: Get the singleton instance
    @classmethod
    def get_instance(cls) -> "PerformanceMonitor":
        """Get the singleton PerformanceMonitor instance.

        Returns:
            The singleton PerformanceMonitor instance
        """
        with cls._lock:
            ##Condition purpose: Create instance if needed
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    ##Method purpose: Initialize the performance monitor
    def __init__(self) -> None:
        """Initialize the performance monitor."""
        ##Step purpose: Initialize storage
        self._stats: dict[str, TimingStats] = {}
        self._stats_lock = threading.Lock()
        self._enabled = True
        self._start_time = datetime.now()

        logger.debug("performance_monitor_initialized")

    ##Method purpose: Enable or disable monitoring
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable performance monitoring.

        Args:
            enabled: Whether monitoring is enabled
        """
        self._enabled = enabled
        logger.info("performance_monitor_enabled", enabled=enabled)

    ##Method purpose: Check if monitoring is enabled
    @property
    def is_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self._enabled

    ##Method purpose: Record a timing measurement
    def record(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement.

        Args:
            name: Name of the operation
            duration_ms: Duration in milliseconds
        """
        ##Condition purpose: Skip if disabled
        if not self._enabled:
            return

        with self._stats_lock:
            ##Condition purpose: Create stats if needed
            if name not in self._stats:
                self._stats[name] = TimingStats(name=name)

            self._stats[name].record(duration_ms)

    ##Method purpose: Context manager for measuring operations
    class TimingContext:
        """Context manager for timing operations."""

        ##Method purpose: Initialize context
        def __init__(self, monitor: "PerformanceMonitor", name: str) -> None:
            self._monitor = monitor
            self._name = name
            self._start: float = 0.0

        ##Method purpose: Enter context and start timing
        def __enter__(self) -> "PerformanceMonitor.TimingContext":
            self._start = time.perf_counter()
            return self

        ##Method purpose: Exit context and record timing
        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
            duration_ms = (time.perf_counter() - self._start) * 1000
            self._monitor.record(self._name, duration_ms)

    ##Method purpose: Create a context manager for measuring
    def measure(self, name: str) -> TimingContext:
        """Create a context manager for measuring an operation.

        Args:
            name: Name of the operation

        Returns:
            Context manager for timing
        """
        return self.TimingContext(self, name)

    ##Method purpose: Get stats for an operation
    def get_stats(self, name: str) -> TimingStats | None:
        """Get statistics for an operation.

        Args:
            name: Name of the operation

        Returns:
            Statistics or None if not found
        """
        with self._stats_lock:
            return self._stats.get(name)

    ##Method purpose: Get all statistics
    def get_all_stats(self) -> dict[str, TimingStats]:
        """Get statistics for all operations.

        Returns:
            Dictionary mapping names to statistics
        """
        with self._stats_lock:
            return dict(self._stats)

    ##Method purpose: Clear all statistics
    def clear(self) -> None:
        """Clear all statistics."""
        with self._stats_lock:
            self._stats.clear()
        logger.debug("performance_monitor_cleared")

    ##Method purpose: Get summary report
    def get_summary(self) -> dict[str, object]:
        """Get a summary report of all statistics.

        Returns:
            Summary dictionary
        """
        with self._stats_lock:
            uptime = datetime.now() - self._start_time
            return {
                "enabled": self._enabled,
                "uptime_seconds": uptime.total_seconds(),
                "operation_count": len(self._stats),
                "operations": {name: stats.to_dict() for name, stats in self._stats.items()},
            }


##Function purpose: Decorator for timing function execution
def timed(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to time function execution.

    Args:
        name: Optional custom name (defaults to function name)

    Returns:
        Decorated function

    Example:
        >>> @timed("my_operation")
        ... def my_function():
        ...     pass
    """

    ##Function purpose: Inner decorator
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        operation_name = name or func.__name__

        ##Function purpose: Wrapper that times execution
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            monitor = PerformanceMonitor.get_instance()

            ##Condition purpose: Skip timing if disabled
            if not monitor.is_enabled:
                return func(*args, **kwargs)

            ##Step purpose: Time the execution
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                monitor.record(operation_name, duration_ms)

        return wrapper

    return decorator


##Function purpose: Log slow operations
def log_slow(
    threshold_ms: float = 1000.0, name: str | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log slow function execution.

    Args:
        threshold_ms: Threshold in milliseconds for logging
        name: Optional custom name (defaults to function name)

    Returns:
        Decorated function

    Example:
        >>> @log_slow(threshold_ms=500)
        ... def potentially_slow():
        ...     pass
    """

    ##Function purpose: Inner decorator
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        operation_name = name or func.__name__

        ##Function purpose: Wrapper that logs slow execution
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                ##Condition purpose: Log if above threshold
                if duration_ms > threshold_ms:
                    logger.warning(
                        "slow_operation",
                        operation=operation_name,
                        duration_ms=round(duration_ms, 2),
                        threshold_ms=threshold_ms,
                    )

        return wrapper

    return decorator
