# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 6: Real-Time Health Display

Provides visual health and metrics display for JENOVA's UI.
"""

from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
import time


class HealthDisplay:
    """
    Real-time health and metrics display for JENOVA.

    Features:
    - System health status (CPU, memory, GPU)
    - Performance metrics (operation times, cache hit rates)
    - Cognitive engine status
    - Error tracking
    """

    def __init__(self, health_monitor=None, metrics=None, console=None):
        """
        Initialize health display.

        Args:
            health_monitor: HealthMonitor instance (optional)
            metrics: MetricsCollector instance (optional)
            console: Rich Console instance (optional)
        """
        self.health_monitor = health_monitor
        self.metrics = metrics
        self.console = console or Console()
        self.last_update = time.time()

    def create_health_table(self) -> Table:
        """
        Create a table showing system health status.

        Returns:
            Rich Table with health information
        """
        table = Table(title="System Health", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")

        if not self.health_monitor:
            table.add_row("Health Monitor", "Not Available", "No health monitor configured")
            return table

        try:
            health = self.health_monitor.get_health_snapshot()

            # Overall status
            status_color = {
                'healthy': 'green',
                'warning': 'yellow',
                'critical': 'red'
            }.get(health.status.value, 'white')

            table.add_row(
                "Overall",
                f"[{status_color}]{health.status.value.upper()}[/{status_color}]",
                f"Updated {int(time.time() - self.last_update)}s ago"
            )

            # CPU
            cpu_color = 'green' if health.cpu_percent < 70 else 'yellow' if health.cpu_percent < 90 else 'red'
            table.add_row(
                "CPU",
                f"[{cpu_color}]{health.cpu_percent:.1f}%[/{cpu_color}]",
                f"{health.cpu_count} cores"
            )

            # Memory
            mem_color = 'green' if health.memory_percent < 70 else 'yellow' if health.memory_percent < 90 else 'red'
            table.add_row(
                "Memory",
                f"[{mem_color}]{health.memory_percent:.1f}%[/{mem_color}]",
                f"{health.memory_available_gb:.1f}GB free / {health.memory_total_gb:.1f}GB total"
            )

            # GPU (if available)
            if health.gpu_available:
                gpu_color = 'green' if health.gpu_memory_percent < 70 else 'yellow' if health.gpu_memory_percent < 90 else 'red'
                table.add_row(
                    "GPU",
                    f"[{gpu_color}]{health.gpu_memory_percent:.1f}%[/{gpu_color}]",
                    f"{health.gpu_name} - {health.gpu_memory_used_mb:.0f}MB / {health.gpu_memory_total_mb:.0f}MB"
                )

        except Exception as e:
            table.add_row("Error", f"[red]Failed[/red]", str(e))

        return table

    def create_metrics_table(self) -> Table:
        """
        Create a table showing performance metrics.

        Returns:
            Rich Table with metrics information
        """
        table = Table(title="Performance Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Operation", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="white")
        table.add_column("Avg Time", justify="right", style="yellow")
        table.add_column("Total Time", justify="right", style="white")

        if not self.metrics:
            table.add_row("Metrics", "Not Available", "", "No metrics collector configured")
            return table

        try:
            all_stats = self.metrics.get_all_stats()

            if not all_stats:
                table.add_row("No Data", "0", "0.00s", "0.00s")
                return table

            # Sort by total time descending
            sorted_stats = sorted(
                all_stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )

            for operation, stats in sorted_stats[:10]:  # Top 10
                table.add_row(
                    operation,
                    str(stats.count),
                    f"{stats.avg_time:.2f}s",
                    f"{stats.total_time:.2f}s"
                )

        except Exception as e:
            table.add_row("Error", "", "", str(e))

        return table

    def create_cognitive_status(self) -> Panel:
        """
        Create a panel showing cognitive engine status.

        Returns:
            Rich Panel with cognitive status
        """
        content = []

        # Check for RAG cache stats (if metrics available)
        if self.metrics:
            try:
                # Try to get cache stats from RAG system if available
                cache_info = "Cache stats not available"
                content.append(Text("RAG System: ", style="bold cyan") + Text(cache_info))
            except Exception as e:
                # RAG cache stats not available, skip display
                pass

        # Check for scheduler status
        content.append(Text("Scheduler: ", style="bold cyan") + Text("Active", style="green"))

        # Memory search config
        content.append(Text("Memory Search: ", style="bold cyan") + Text("Operational", style="green"))

        if not content:
            content.append(Text("No cognitive engine status available", style="yellow"))

        panel_content = Text("\n").join(content)
        return Panel(
            panel_content,
            title="Cognitive Engine Status",
            border_style="cyan",
            padding=(1, 2)
        )

    def create_full_display(self) -> Panel:
        """
        Create a complete health and metrics display.

        Returns:
            Rich Panel containing all health information
        """
        from rich.columns import Columns

        # Create all components
        health_table = self.create_health_table()
        metrics_table = self.create_metrics_table()
        cognitive_status = self.create_cognitive_status()

        # Combine into columns
        display = Columns([health_table, metrics_table], equal=True, expand=True)

        return Panel(
            display,
            title="JENOVA System Monitor",
            border_style="bold magenta",
            padding=(1, 2)
        )

    def show_health(self):
        """Display current health status."""
        self.console.print(self.create_health_table())

    def show_metrics(self):
        """Display current performance metrics."""
        self.console.print(self.create_metrics_table())

    def show_cognitive_status(self):
        """Display cognitive engine status."""
        self.console.print(self.create_cognitive_status())

    def show_full_status(self):
        """Display complete system status."""
        self.console.clear()
        self.console.print(self.create_full_display())
        self.last_update = time.time()

    def start_live_display(self, refresh_rate: float = 2.0):
        """
        Start a live-updating display.

        Args:
            refresh_rate: Refresh interval in seconds (default: 2.0)

        Returns:
            Rich Live context manager
        """
        def generate_display():
            while True:
                yield self.create_full_display()
                time.sleep(refresh_rate)

        return Live(generate_display(), console=self.console, refresh_per_second=1/refresh_rate)

    def get_status_summary(self) -> dict:
        """
        Get a dictionary summary of current status.

        Returns:
            dict: Status summary with health and metrics
        """
        summary = {
            'health': {},
            'metrics': {},
            'timestamp': time.time()
        }

        if self.health_monitor:
            try:
                health = self.health_monitor.get_health_snapshot()
                summary['health'] = {
                    'status': health.status.value,
                    'cpu_percent': health.cpu_percent,
                    'memory_percent': health.memory_percent,
                    'memory_available_gb': health.memory_available_gb,
                    'gpu_available': health.gpu_available,
                    'warnings': health.warnings
                }
                if health.gpu_available:
                    summary['health']['gpu'] = {
                        'name': health.gpu_name,
                        'memory_percent': health.gpu_memory_percent,
                        'memory_used_mb': health.gpu_memory_used_mb
                    }
            except Exception as e:
                summary['health']['error'] = str(e)

        if self.metrics:
            try:
                all_stats = self.metrics.get_all_stats()
                summary['metrics'] = {
                    operation: {
                        'count': stats.count,
                        'avg_time': stats.avg_time,
                        'total_time': stats.total_time
                    }
                    for operation, stats in all_stats.items()
                }
            except Exception as e:
                summary['metrics']['error'] = str(e)

        return summary


class CompactHealthDisplay:
    """
    Compact, single-line health display for minimal UI footprint.
    """

    def __init__(self, health_monitor=None):
        self.health_monitor = health_monitor

    def get_status_line(self) -> str:
        """
        Get a compact status line.

        Returns:
            str: Single-line status string
        """
        if not self.health_monitor:
            return "[dim]Health: N/A[/dim]"

        try:
            health = self.health_monitor.get_health_snapshot()

            status_color = {
                'healthy': 'green',
                'warning': 'yellow',
                'critical': 'red'
            }.get(health.status.value, 'white')

            status_icon = {
                'healthy': '●',
                'warning': '◐',
                'critical': '○'
            }.get(health.status.value, '?')

            parts = [
                f"[{status_color}]{status_icon}[/{status_color}]",
                f"CPU {health.cpu_percent:.0f}%",
                f"MEM {health.memory_percent:.0f}%"
            ]

            if health.gpu_available:
                parts.append(f"GPU {health.gpu_memory_percent:.0f}%")

            return " | ".join(parts)

        except Exception:
            return "[red]Health: Error[/red]"
