##Script function and purpose: Performance Monitor - Timing decorators and statistics for expensive operations
##Dependency purpose: Provides performance metrics and monitoring for expensive operations with timing decorators and statistics
"""Performance Monitoring for JENOVA.

This module provides performance metrics and monitoring for expensive operations.
Includes timing decorators and statistics collection.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Protocol

import structlog

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for file logger operations
class FileLoggerProtocol(Protocol):
    """Protocol for file logger operations."""
    
    ##Method purpose: Log info message
    def log_info(self, message: str) -> None:
        """Log info message."""
        ...


##Class purpose: Configuration for performance monitor
@dataclass
class PerformanceConfig:
    """Configuration for performance monitor.
    
    Attributes:
        enabled: Whether monitoring is enabled (default: True).
        max_timings_per_operation: Maximum timings to keep per operation (default: 100).
    """
    
    enabled: bool = True
    max_timings_per_operation: int = 100


##Class purpose: Monitors and tracks performance metrics for operations
class PerformanceMonitor:
    """Monitors and tracks performance metrics for operations.
    
    Provides timing decorators and statistics collection for expensive
    operations. Tracks operation counts, total times, and recent averages.
    
    Attributes:
        config: Performance monitor configuration.
        file_logger: Optional file logger for logging statistics.
        operation_times: Dictionary mapping operation names to timing lists.
        operation_counts: Dictionary mapping operation names to call counts.
        total_times: Dictionary mapping operation names to total time.
        cache_stats: Dictionary mapping cache names to statistics.
    """
    
    ##Method purpose: Initialize performance monitor
    def __init__(
        self,
        config: PerformanceConfig,
        file_logger: FileLoggerProtocol | None = None,
    ) -> None:
        """Initialize performance monitor.
        
        Args:
            config: Performance monitor configuration.
            file_logger: Optional file logger for statistics logging.
        """
        ##Step purpose: Store configuration and dependencies
        self.config = config
        self.file_logger = file_logger
        
        ##Step purpose: Initialize tracking dictionaries
        self.operation_times: dict[str, list[float]] = defaultdict(list)
        self.operation_counts: dict[str, int] = defaultdict(int)
        self.total_times: dict[str, float] = defaultdict(float)
        
        ##Step purpose: Initialize cache statistics
        self.cache_stats: dict[str, dict[str, Any]] = {}
        
        ##Action purpose: Log initialization
        logger.info(
            "performance_monitor_initialized",
            enabled=self.config.enabled,
        )
    
    ##Method purpose: Decorator to measure function execution time
    def measure_time(self, operation_name: str) -> Callable[[Callable], Callable]:
        """Decorator to measure execution time of a function.
        
        Args:
            operation_name: Name of the operation for tracking.
            
        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                ##Condition purpose: Skip measurement if disabled
                if not self.config.enabled:
                    return func(*args, **kwargs)
                
                ##Step purpose: Measure execution time
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    self.record_operation(operation_name, elapsed)
            
            return wrapper
        return decorator
    
    ##Method purpose: Record operation timing
    def record_operation(self, operation_name: str, elapsed_time: float) -> None:
        """Record timing for an operation.
        
        Args:
            operation_name: Name of the operation.
            elapsed_time: Elapsed time in seconds.
        """
        ##Condition purpose: Skip if disabled
        if not self.config.enabled:
            return
        
        ##Step purpose: Record timing
        self.operation_times[operation_name].append(elapsed_time)
        self.operation_counts[operation_name] += 1
        self.total_times[operation_name] += elapsed_time
        
        ##Condition purpose: Keep only recent timings
        if len(self.operation_times[operation_name]) > self.config.max_timings_per_operation:
            self.operation_times[operation_name] = self.operation_times[operation_name][
                -self.config.max_timings_per_operation:
            ]
    
    ##Method purpose: Get performance statistics for an operation
    def get_stats(self, operation_name: str) -> dict[str, Any] | None:
        """Get performance statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation.
            
        Returns:
            Dictionary with statistics or None if operation not found.
        """
        ##Condition purpose: Return None if operation not tracked
        if operation_name not in self.operation_counts:
            return None
        
        times = self.operation_times[operation_name]
        if not times:
            return None
        
        ##Step purpose: Calculate statistics
        return {
            "count": self.operation_counts[operation_name],
            "total_time": self.total_times[operation_name],
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "recent_avg": (
                sum(times[-10:]) / len(times[-10:])
                if len(times) >= 10
                else sum(times) / len(times)
            ),
        }
    
    ##Method purpose: Get all performance statistics
    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get performance statistics for all tracked operations.
        
        Returns:
            Dictionary mapping operation names to statistics.
        """
        all_stats: dict[str, dict[str, Any]] = {}
        
        ##Loop purpose: Collect statistics for all operations
        for operation_name in self.operation_counts.keys():
            stats = self.get_stats(operation_name)
            if stats:
                all_stats[operation_name] = stats
        
        return all_stats
    
    ##Method purpose: Update cache statistics
    def update_cache_stats(self, cache_name: str, stats: dict[str, Any]) -> None:
        """Update cache performance statistics.
        
        Args:
            cache_name: Name of the cache.
            stats: Cache statistics dictionary.
        """
        self.cache_stats[cache_name] = stats
    
    ##Method purpose: Log performance summary
    def log_summary(self) -> None:
        """Log performance summary to file logger."""
        ##Condition purpose: Skip if disabled or no logger
        if not self.config.enabled or not self.file_logger:
            return
        
        all_stats = self.get_all_stats()
        if not all_stats:
            return
        
        ##Step purpose: Log operation statistics
        self.file_logger.log_info("=== Performance Summary ===")
        for operation_name, stats in all_stats.items():
            self.file_logger.log_info(
                f"{operation_name}: "
                f"count={stats['count']}, "
                f"avg={stats['avg_time']:.4f}s, "
                f"total={stats['total_time']:.2f}s"
            )
        
        ##Condition purpose: Log cache statistics if available
        if self.cache_stats:
            self.file_logger.log_info("=== Cache Performance ===")
            for cache_name, stats in self.cache_stats.items():
                hit_rate = stats.get("hit_rate", 0)
                self.file_logger.log_info(
                    f"{cache_name}: "
                    f"hit_rate={hit_rate:.1f}%, "
                    f"hits={stats.get('hits', 0)}, "
                    f"misses={stats.get('misses', 0)}"
                )
    
    ##Method purpose: Reset all statistics
    def reset(self) -> None:
        """Reset all performance statistics."""
        self.operation_times.clear()
        self.operation_counts.clear()
        self.total_times.clear()
        self.cache_stats.clear()
        logger.debug("performance_monitor_reset")
