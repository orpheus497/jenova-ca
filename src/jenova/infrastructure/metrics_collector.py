# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Performance metrics collection and reporting.

Tracks operation timings, success rates, and system performance.
"""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jenova.infrastructure.data_validator import PerformanceMetric


@dataclass
class MetricStats:
    """Statistics for a metric."""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_time(self) -> float:
        """Average execution time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.success_count / self.count * 100) if self.count > 0 else 0.0

    @property
    def recent_avg_time(self) -> float:
        """Average of recent execution times."""
        return (
            sum(self.recent_times) / len(self.recent_times)
            if self.recent_times
            else 0.0
        )


class MetricsCollector:
    """Collect and analyze performance metrics."""

    def __init__(self, ui_logger=None, file_logger=None):
        """
        Initialize metrics collector.

        Args:
            ui_logger: Optional UI logger
            file_logger: Optional file logger
        """
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.metrics: Dict[str, MetricStats] = defaultdict(MetricStats)
        self.metric_history: List[PerformanceMetric] = []
        self.max_history_size = 1000
        self.start_time = time.time()

    @contextmanager
    def measure(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager to measure operation performance.

        Args:
            operation: Name of the operation
            metadata: Optional additional metadata

        Yields:
            None

        Example:
            with metrics.measure('model_inference'):
                result = model.generate(prompt)
        """
        start_time = time.time()
        success = False
        error = None

        try:
            yield
            success = True
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time
            self.record_metric(
                operation=operation,
                duration=duration,
                success=success,
                metadata=metadata or {},
            )

    def record_metric(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a performance metric.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether operation succeeded
            metadata: Optional additional metadata
        """
        # Update statistics
        stats = self.metrics[operation]
        stats.count += 1
        stats.total_time += duration
        stats.min_time = min(stats.min_time, duration)
        stats.max_time = max(stats.max_time, duration)
        stats.recent_times.append(duration)

        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

        # Create metric entry
        metric = PerformanceMetric(
            operation=operation,
            duration_seconds=duration,
            success=success,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Add to history (with size limit)
        self.metric_history.append(metric)
        if len(self.metric_history) > self.max_history_size:
            self.metric_history = self.metric_history[-self.max_history_size :]

        # Log slow operations
        if duration > 10.0 and self.file_logger:
            self.file_logger.log_warning(
                f"Slow operation: {operation} took {duration:.2f}s"
            )

    def get_stats(self, operation: str) -> Optional[MetricStats]:
        """
        Get statistics for an operation.

        Args:
            operation: Operation name

        Returns:
            MetricStats object or None if operation not found
        """
        return self.metrics.get(operation)

    def get_all_stats(self) -> Dict[str, MetricStats]:
        """
        Get statistics for all operations.

        Returns:
            Dictionary of operation name to MetricStats
        """
        return dict(self.metrics)

    def get_recent_metrics(self, limit: int = 100) -> List[PerformanceMetric]:
        """
        Get recent metrics.

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of recent PerformanceMetric objects
        """
        return self.metric_history[-limit:]

    def get_metrics_by_operation(self, operation: str) -> List[PerformanceMetric]:
        """
        Get all metrics for a specific operation.

        Args:
            operation: Operation name

        Returns:
            List of PerformanceMetric objects
        """
        return [m for m in self.metric_history if m.operation == operation]

    def format_summary(self, top_n: int = 10) -> str:
        """
        Format a summary of metrics.

        Args:
            top_n: Number of top operations to include

        Returns:
            Formatted summary string
        """
        if not self.metrics:
            return "No metrics collected yet."

        lines = [
            "=== Performance Metrics Summary ===",
            f"Uptime: {time.time() - self.start_time:.1f}s",
            f"Total operations: {sum(s.count for s in self.metrics.values())}",
            "",
        ]

        # Sort by total time (most expensive operations first)
        sorted_ops = sorted(
            self.metrics.items(), key=lambda x: x[1].total_time, reverse=True
        )[:top_n]

        lines.append(f"Top {len(sorted_ops)} operations by total time:")
        lines.append("-" * 80)
        lines.append(
            f"{'Operation':<30} {'Count':>8} {'Avg (s)':>10} {'Total (s)':>12} {'Success %':>12}"
        )
        lines.append("-" * 80)

        for op_name, stats in sorted_ops:
            lines.append(
                f"{op_name:<30} {stats.count:>8} "
                f"{stats.avg_time:>10.3f} {stats.total_time:>12.2f} "
                f"{stats.success_rate:>11.1f}%"
            )

        return "\n".join(lines)

    def format_operation_details(self, operation: str) -> str:
        """
        Format detailed statistics for an operation.

        Args:
            operation: Operation name

        Returns:
            Formatted details string
        """
        stats = self.metrics.get(operation)
        if not stats:
            return f"No metrics found for operation: {operation}"

        lines = [
            f"=== Metrics for: {operation} ===",
            f"Total calls: {stats.count}",
            f"Success rate: {stats.success_rate:.1f}%",
            f"Successful: {stats.success_count}",
            f"Failed: {stats.failure_count}",
            "",
            "Timing:",
            f"  Average: {stats.avg_time:.3f}s",
            f"  Recent avg: {stats.recent_avg_time:.3f}s",
            f"  Min: {stats.min_time:.3f}s",
            f"  Max: {stats.max_time:.3f}s",
            f"  Total: {stats.total_time:.2f}s",
        ]

        return "\n".join(lines)

    def log_summary(self, top_n: int = 10):
        """
        Log metrics summary to file logger.

        Args:
            top_n: Number of top operations to include
        """
        if self.file_logger:
            summary = self.format_summary(top_n)
            self.file_logger.log_info(summary)

    def reset_metrics(self):
        """Reset all collected metrics."""
        self.metrics.clear()
        self.metric_history.clear()
        self.start_time = time.time()

        if self.file_logger:
            self.file_logger.log_info("Metrics reset")

    def get_slow_operations(
        self, threshold_seconds: float = 5.0
    ) -> List[PerformanceMetric]:
        """
        Get operations that took longer than threshold.

        Args:
            threshold_seconds: Minimum duration to consider slow

        Returns:
            List of slow PerformanceMetric objects
        """
        return [
            m for m in self.metric_history if m.duration_seconds >= threshold_seconds
        ]

    def get_failed_operations(self) -> List[PerformanceMetric]:
        """
        Get all failed operations.

        Returns:
            List of failed PerformanceMetric objects
        """
        return [m for m in self.metric_history if not m.success]

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics as a dictionary.

        Returns:
            Dictionary with all metrics data
        """
        return {
            "uptime_seconds": time.time() - self.start_time,
            "operations": {
                op_name: {
                    "count": stats.count,
                    "avg_time": stats.avg_time,
                    "total_time": stats.total_time,
                    "min_time": stats.min_time if stats.min_time != float("inf") else 0,
                    "max_time": stats.max_time,
                    "success_rate": stats.success_rate,
                    "success_count": stats.success_count,
                    "failure_count": stats.failure_count,
                }
                for op_name, stats in self.metrics.items()
            },
            "recent_metrics": [
                {
                    "operation": m.operation,
                    "duration": m.duration_seconds,
                    "success": m.success,
                    "timestamp": m.timestamp,
                    "metadata": m.metadata,
                }
                for m in self.metric_history[-100:]  # Last 100 metrics
            ],
        }

    def get_operation_trend(self, operation: str, window_size: int = 10) -> List[float]:
        """
        Get trend of recent execution times for an operation.

        Args:
            operation: Operation name
            window_size: Number of recent executions to include

        Returns:
            List of recent execution times
        """
        metrics = self.get_metrics_by_operation(operation)
        return [m.duration_seconds for m in metrics[-window_size:]]

    def is_degrading(self, operation: str, threshold_factor: float = 2.0) -> bool:
        """
        Check if operation performance is degrading.

        Compares recent average to overall average.

        Args:
            operation: Operation name
            threshold_factor: Factor by which recent avg must exceed overall avg

        Returns:
            True if performance is degrading
        """
        stats = self.metrics.get(operation)
        if not stats or stats.count < 10:
            return False  # Not enough data

        return stats.recent_avg_time > (stats.avg_time * threshold_factor)

    def get_performance_alerts(self) -> List[str]:
        """
        Get list of performance alerts.

        Returns:
            List of alert messages
        """
        alerts = []

        for op_name, stats in self.metrics.items():
            # Check for low success rate
            if stats.count >= 5 and stats.success_rate < 80:
                alerts.append(
                    f"Low success rate for {op_name}: {stats.success_rate:.1f}%"
                )

            # Check for degrading performance
            if self.is_degrading(op_name):
                alerts.append(
                    f"Performance degrading for {op_name}: "
                    f"recent {stats.recent_avg_time:.2f}s vs avg {stats.avg_time:.2f}s"
                )

            # Check for very slow operations
            if stats.avg_time > 30.0:
                alerts.append(
                    f"Very slow operation {op_name}: avg {stats.avg_time:.2f}s"
                )

        return alerts
