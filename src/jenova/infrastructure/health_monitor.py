# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Real-time system health monitoring.

Tracks CPU, memory, GPU resources and provides health status checks.
"""

import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import psutil


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemHealth:
    """System health snapshot."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_percent: Optional[float]
    gpu_memory_used_mb: Optional[int]
    gpu_memory_total_mb: Optional[int]
    gpu_memory_free_mb: Optional[int]
    status: HealthStatus
    warnings: list[str]
    timestamp: float


class HealthMonitor:
    """Monitor system resources in real-time."""

    def __init__(self, ui_logger=None, file_logger=None):
        """
        Initialize health monitor.

        Args:
            ui_logger: Optional UI logger
            file_logger: Optional file logger
        """
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.has_gpu = self._check_gpu_available()
        
        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 80.0
        self.memory_critical_threshold = 90.0
        self.gpu_memory_warning_threshold = 85.0
        self.gpu_memory_critical_threshold = 95.0

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available via nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0 and result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_gpu_stats(self) -> tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
        """
        Get GPU statistics.

        Returns:
            Tuple of (utilization%, used_mb, total_mb, free_mb) or (None, None, None, None)
        """
        if not self.has_gpu:
            return None, None, None, None

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 4:
                    gpu_util = float(parts[0].strip())
                    mem_used = int(parts[1].strip())
                    mem_total = int(parts[2].strip())
                    mem_free = int(parts[3].strip())
                    return gpu_util, mem_used, mem_total, mem_free
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return None, None, None, None

    def get_health_snapshot(self) -> SystemHealth:
        """
        Get current system health snapshot.

        Returns:
            SystemHealth object with current status
        """
        warnings = []
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024 ** 3)
        
        # GPU
        gpu_percent, gpu_mem_used, gpu_mem_total, gpu_mem_free = self._get_gpu_stats()
        
        # Determine status based on thresholds
        status = HealthStatus.HEALTHY
        
        # Check CPU
        if cpu_percent >= self.cpu_critical_threshold:
            status = HealthStatus.CRITICAL
            warnings.append(f"CPU at critical level: {cpu_percent:.1f}%")
        elif cpu_percent >= self.cpu_warning_threshold:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            warnings.append(f"CPU high: {cpu_percent:.1f}%")
        
        # Check Memory
        if memory_percent >= self.memory_critical_threshold:
            status = HealthStatus.CRITICAL
            warnings.append(f"Memory at critical level: {memory_percent:.1f}%")
        elif memory_percent >= self.memory_warning_threshold:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            warnings.append(f"Memory high: {memory_percent:.1f}%")
        
        # Check GPU Memory
        if gpu_mem_total and gpu_mem_used:
            gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
            if gpu_mem_percent >= self.gpu_memory_critical_threshold:
                status = HealthStatus.CRITICAL
                warnings.append(f"GPU memory critical: {gpu_mem_percent:.1f}%")
            elif gpu_mem_percent >= self.gpu_memory_warning_threshold:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                warnings.append(f"GPU memory high: {gpu_mem_percent:.1f}%")
        
        return SystemHealth(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            gpu_percent=gpu_percent,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_total_mb=gpu_mem_total,
            gpu_memory_free_mb=gpu_mem_free,
            status=status,
            warnings=warnings,
            timestamp=time.time()
        )

    def check_health(self) -> tuple[HealthStatus, list[str]]:
        """
        Quick health check.

        Returns:
            Tuple of (status, warnings)
        """
        snapshot = self.get_health_snapshot()
        return snapshot.status, snapshot.warnings

    def is_healthy(self) -> bool:
        """
        Check if system is healthy.

        Returns:
            True if system is healthy, False otherwise
        """
        status, _ = self.check_health()
        return status == HealthStatus.HEALTHY

    def log_health_snapshot(self):
        """Log current health snapshot to loggers."""
        snapshot = self.get_health_snapshot()
        
        msg = (
            f"System Health: {snapshot.status.value.upper()} | "
            f"CPU: {snapshot.cpu_percent:.1f}% | "
            f"Memory: {snapshot.memory_percent:.1f}% ({snapshot.memory_available_gb:.1f}GB free)"
        )
        
        if snapshot.gpu_memory_total:
            gpu_mem_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100
            msg += f" | GPU Memory: {gpu_mem_percent:.1f}% ({snapshot.gpu_memory_free_mb}MB free)"
        
        if self.file_logger:
            self.file_logger.log_info(msg)
        
        # Log warnings
        if snapshot.warnings:
            for warning in snapshot.warnings:
                if self.ui_logger:
                    self.ui_logger.warning(warning)
                if self.file_logger:
                    self.file_logger.log_warning(warning)

    def wait_for_resources(self, 
                          min_memory_gb: float = 1.0,
                          max_wait_seconds: float = 30.0,
                          check_interval: float = 1.0) -> bool:
        """
        Wait for sufficient resources to become available.

        Args:
            min_memory_gb: Minimum free memory required in GB
            max_wait_seconds: Maximum time to wait
            check_interval: Time between checks

        Returns:
            True if resources became available, False if timeout
        """
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_seconds:
            snapshot = self.get_health_snapshot()
            
            if snapshot.memory_available_gb >= min_memory_gb:
                if self.file_logger:
                    self.file_logger.log_info(
                        f"Resources available: {snapshot.memory_available_gb:.1f}GB free"
                    )
                return True
            
            time.sleep(check_interval)
        
        if self.file_logger:
            self.file_logger.log_warning(
                f"Timeout waiting for {min_memory_gb}GB free memory after {max_wait_seconds}s"
            )
        return False

    def get_gpu_info(self) -> Optional[dict]:
        """
        Get detailed GPU information.

        Returns:
            Dictionary with GPU info or None if no GPU
        """
        if not self.has_gpu:
            return None

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 5:
                    return {
                        'name': parts[0].strip(),
                        'driver_version': parts[1].strip(),
                        'memory_total_mb': int(parts[2].strip()),
                        'memory_free_mb': int(parts[3].strip()),
                        'temperature_c': int(parts[4].strip())
                    }
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return None

    def format_health_report(self, snapshot: SystemHealth) -> str:
        """
        Format health snapshot as readable report.

        Args:
            snapshot: SystemHealth object

        Returns:
            Formatted string report
        """
        lines = [
            f"System Health Report - {snapshot.status.value.upper()}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snapshot.timestamp))}",
            "",
            f"CPU Usage: {snapshot.cpu_percent:.1f}%",
            f"Memory Usage: {snapshot.memory_percent:.1f}%",
            f"Memory Available: {snapshot.memory_available_gb:.2f} GB",
        ]
        
        if snapshot.gpu_memory_total:
            gpu_mem_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100
            lines.extend([
                "",
                f"GPU Utilization: {snapshot.gpu_percent:.1f}%" if snapshot.gpu_percent else "GPU Utilization: N/A",
                f"GPU Memory Used: {snapshot.gpu_memory_used_mb} MB / {snapshot.gpu_memory_total_mb} MB ({gpu_mem_percent:.1f}%)",
                f"GPU Memory Free: {snapshot.gpu_memory_free_mb} MB"
            ])
        else:
            lines.append("\nGPU: Not detected or unavailable")
        
        if snapshot.warnings:
            lines.append("\nWarnings:")
            for warning in snapshot.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)
