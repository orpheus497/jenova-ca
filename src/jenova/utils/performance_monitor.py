##Script function and purpose: Performance Monitoring for The JENOVA Cognitive Architecture
##This module provides performance metrics and monitoring for expensive operations

import time
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
from datetime import datetime

##Class purpose: Monitors and tracks performance metrics for operations
class PerformanceMonitor:
    ##Function purpose: Initialize performance monitor
    def __init__(self, config: Dict[str, Any], file_logger: Any) -> None:
        self.config = config
        self.file_logger = file_logger
        self.enabled = config.get('performance', {}).get('monitoring', {}).get('enabled', True)
        
        ##Block purpose: Track operation timings
        self.operation_times: Dict[str, list] = defaultdict(list)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.total_times: Dict[str, float] = defaultdict(float)
        
        ##Block purpose: Track cache performance
        self.cache_stats: Dict[str, Dict[str, Any]] = {}
    
    ##Function purpose: Decorator to measure function execution time
    def measure_time(self, operation_name: str):
        """Decorator to measure execution time of a function."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    self.record_operation(operation_name, elapsed)
            
            return wrapper
        return decorator
    
    ##Function purpose: Record operation timing
    def record_operation(self, operation_name: str, elapsed_time: float) -> None:
        """Record timing for an operation."""
        if not self.enabled:
            return
        
        self.operation_times[operation_name].append(elapsed_time)
        self.operation_counts[operation_name] += 1
        self.total_times[operation_name] += elapsed_time
        
        ##Block purpose: Keep only recent timings (last 100)
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
    
    ##Function purpose: Get performance statistics for an operation
    def get_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a specific operation."""
        if operation_name not in self.operation_counts:
            return None
        
        times = self.operation_times[operation_name]
        if not times:
            return None
        
        return {
            'count': self.operation_counts[operation_name],
            'total_time': self.total_times[operation_name],
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'recent_avg': sum(times[-10:]) / len(times[-10:]) if len(times) >= 10 else sum(times) / len(times)
        }
    
    ##Function purpose: Get all performance statistics
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all tracked operations."""
        all_stats = {}
        for operation_name in self.operation_counts.keys():
            stats = self.get_stats(operation_name)
            if stats:
                all_stats[operation_name] = stats
        return all_stats
    
    ##Function purpose: Update cache statistics
    def update_cache_stats(self, cache_name: str, stats: Dict[str, Any]) -> None:
        """Update cache performance statistics."""
        self.cache_stats[cache_name] = stats
    
    ##Function purpose: Log performance summary
    def log_summary(self) -> None:
        """Log performance summary to file logger."""
        if not self.enabled:
            return
        
        all_stats = self.get_all_stats()
        if not all_stats:
            return
        
        self.file_logger.log_info("=== Performance Summary ===")
        for operation_name, stats in all_stats.items():
            self.file_logger.log_info(
                f"{operation_name}: "
                f"count={stats['count']}, "
                f"avg={stats['avg_time']:.4f}s, "
                f"total={stats['total_time']:.2f}s"
            )
        
        if self.cache_stats:
            self.file_logger.log_info("=== Cache Performance ===")
            for cache_name, stats in self.cache_stats.items():
                hit_rate = stats.get('hit_rate', 0)
                self.file_logger.log_info(
                    f"{cache_name}: "
                    f"hit_rate={hit_rate:.1f}%, "
                    f"hits={stats.get('hits', 0)}, "
                    f"misses={stats.get('misses', 0)}"
                )
    
    ##Function purpose: Reset all statistics
    def reset(self) -> None:
        """Reset all performance statistics."""
        self.operation_times.clear()
        self.operation_counts.clear()
        self.total_times.clear()
        self.cache_stats.clear()
