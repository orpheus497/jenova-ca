# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 6: Enhanced UI Logging

This module is responsible for the UI logging of the JENOVA Cognitive Architecture.

Enhancements:
- Metrics output display
- Health status display
- Performance tracking output
- Warning/error styling improvements
"""

import os
import queue
import threading
from contextlib import contextmanager
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


class UILogger:
    def __init__(self, message_queue=None):
        self.console = Console()
        self._console_lock = threading.RLock()
        self.message_queue = message_queue  # Queue for non-blocking UI updates
        try:
            self.term_width = os.get_terminal_size().columns
        except OSError:
            self.term_width = 80

    def _direct_banner(self, banner_text, attribution_text):
        with self._console_lock:
            try:
                self.term_width = os.get_terminal_size().columns
            except OSError:
                self.term_width = 80
            self.console.clear()
            panel = Panel(
                Text(banner_text, style="bold cyan", justify="center"),
                title="The JENOVA Cognitive Architecture (JCA)",
                title_align="center",
                subtitle=Text(attribution_text, style="cyan",
                              justify="center"),
                border_style="bold magenta",
            )
            self.console.print(panel)
            self.console.print()

    def banner(self, banner_text, attribution_text):
        # Banner is always printed directly (startup only)
        self._direct_banner(banner_text, attribution_text)

    def _queue_or_print(self, msg_type, *args, **kwargs):
        """Helper method to queue messages when in non-blocking mode, or print directly."""
        if self.message_queue is not None:
            self.message_queue.put((msg_type, args, kwargs))
        else:
            # Direct printing with lock (legacy mode)
            getattr(self, f'_direct_{msg_type}')(*args, **kwargs)

    def _direct_error(self, message):
        with self._console_lock:
            self.console.print(f"[bold red]>> ERROR: {message}[/bold red]")

    def error(self, message):
        self._queue_or_print('error', message)

    def _direct_info(self, message):
        with self._console_lock:
            self.console.print(f"[bold green]>> {message}[/bold green]")

    def info(self, message):
        self._queue_or_print('info', message)

    def _direct_system_message(self, message):
        with self._console_lock:
            self.console.print(Text.from_markup(
                f"[bold red]>> {message}[/bold red]"))

    def system_message(self, message):
        self._queue_or_print('system_message', message)

    def _direct_help_message(self, message):
        with self._console_lock:
            self.console.print(message)

    def help_message(self, message):
        self._queue_or_print('help_message', message)

    def _direct_reflection(self, message):
        with self._console_lock:
            self.console.print(f"\n[italic yellow]({message})[/italic yellow]")

    def reflection(self, message):
        self._queue_or_print('reflection', message)

    @contextmanager
    def cognitive_process(self, message: str):
        """Non-blocking cognitive process status indicator."""
        if self.message_queue is not None:
            # Queue-based mode: just queue the status update
            self.message_queue.put(('start_status', (message,), {
                                   'spinner': 'earth', 'style': 'bold green'}))
            try:
                yield None  # No status object in non-blocking mode
            finally:
                self.message_queue.put(('stop_status', (), {}))
        else:
            # Legacy direct mode with lock
            with self._console_lock:
                with self.console.status(f"[bold green]{message}[/bold green]", spinner="earth") as status:
                    yield status

    @contextmanager
    def thinking_process(self, message: str):
        """Non-blocking thinking process status indicator."""
        if self.message_queue is not None:
            # Queue-based mode: just queue the status update
            self.message_queue.put(('start_status', (message,), {
                                   'spinner': 'dots', 'style': 'bold yellow'}))
            try:
                yield None  # No status object in non-blocking mode
            finally:
                self.message_queue.put(('stop_status', (), {}))
        else:
            # Legacy direct mode with lock
            with self._console_lock:
                with self.console.status(f"[bold yellow]{message}[/bold yellow]", spinner="dots") as status:
                    yield status

    def _direct_user_query(self, text, username: str):
        with self._console_lock:
            panel = Panel(Text(text, style="white"),
                          title=f"{username}@JENOVA", border_style="dark_green")
            self.console.print(panel)

    def user_query(self, text, username: str):
        self._queue_or_print('user_query', text, username)

    def _direct_jenova_response(self, text):
        with self._console_lock:
            if not isinstance(text, str):
                text = str(text)
            try:
                # Try to render as Markdown
                panel = Panel(Markdown(text, style="cyan"),
                              title="JENOVA", border_style="magenta")
            except TypeError:
                # If Markdown parsing fails, render as plain text
                panel = Panel(Text(text, style="cyan"),
                              title="JENOVA", border_style="magenta")
            self.console.print(panel)

    def jenova_response(self, text):
        self._queue_or_print('jenova_response', text)

    def process_queued_messages(self):
        """Process all queued messages. Called from main UI thread."""
        if self.message_queue is None:
            return

        # Process all pending messages
        while not self.message_queue.empty():
            try:
                msg_type, args, kwargs = self.message_queue.get_nowait()

                # Handle special message types
                if msg_type == 'start_status':
                    # For status updates in queue mode, we don't actually start a spinner
                    # The TerminalUI will handle its own spinner
                    pass
                elif msg_type == 'stop_status':
                    # Similarly, stop is handled by TerminalUI
                    pass
                else:
                    # Call the direct method
                    method = getattr(self, f'_direct_{msg_type}', None)
                    if method:
                        method(*args, **kwargs)
            except queue.Empty:
                break

    # Phase 6: Metrics and Health Display Methods

    def _direct_warning(self, message):
        """Display a warning message."""
        with self._console_lock:
            self.console.print(f"[bold yellow]⚠ WARNING: {message}[/bold yellow]")

    def warning(self, message):
        """Queue or print a warning message."""
        self._queue_or_print('warning', message)

    def _direct_success(self, message):
        """Display a success message."""
        with self._console_lock:
            self.console.print(f"[bold green]✓ {message}[/bold green]")

    def success(self, message):
        """Queue or print a success message."""
        self._queue_or_print('success', message)

    def _direct_metrics_table(self, metrics_data: dict):
        """Display performance metrics in a table."""
        with self._console_lock:
            table = Table(title="Performance Metrics", show_header=True, header_style="bold magenta")
            table.add_column("Operation", style="cyan", no_wrap=True)
            table.add_column("Count", justify="right", style="white")
            table.add_column("Avg Time", justify="right", style="yellow")
            table.add_column("Total Time", justify="right", style="white")

            for operation, stats in metrics_data.items():
                table.add_row(
                    operation,
                    str(stats.get('count', 0)),
                    f"{stats.get('avg_time', 0):.2f}s",
                    f"{stats.get('total_time', 0):.2f}s"
                )

            self.console.print(table)

    def metrics_table(self, metrics_data: dict):
        """Queue or print metrics table."""
        self._queue_or_print('metrics_table', metrics_data)

    def _direct_health_status(self, health_data: dict):
        """Display system health status."""
        with self._console_lock:
            status = health_data.get('status', 'unknown')
            status_color = {
                'healthy': 'green',
                'warning': 'yellow',
                'critical': 'red'
            }.get(status, 'white')

            status_text = Text()
            status_text.append("System Health: ", style="bold white")
            status_text.append(status.upper(), style=f"bold {status_color}")

            # Add details
            if 'cpu_percent' in health_data:
                status_text.append(f"\nCPU: {health_data['cpu_percent']:.1f}%", style="cyan")
            if 'memory_percent' in health_data:
                status_text.append(f" | Memory: {health_data['memory_percent']:.1f}%", style="cyan")
            if 'gpu_available' in health_data and health_data['gpu_available']:
                gpu_data = health_data.get('gpu', {})
                status_text.append(f" | GPU: {gpu_data.get('memory_percent', 0):.1f}%", style="cyan")

            panel = Panel(status_text, border_style=status_color, padding=(0, 2))
            self.console.print(panel)

    def health_status(self, health_data: dict):
        """Queue or print health status."""
        self._queue_or_print('health_status', health_data)

    def _direct_progress_message(self, message, percentage: Optional[int] = None):
        """Display a progress message."""
        with self._console_lock:
            if percentage is not None:
                bar_length = 30
                filled = int(bar_length * percentage / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                self.console.print(f"[cyan]{message}[/cyan] [{bar}] {percentage}%")
            else:
                self.console.print(f"[cyan]⋯ {message}[/cyan]")

    def progress_message(self, message, percentage: Optional[int] = None):
        """Queue or print progress message."""
        self._queue_or_print('progress_message', message, percentage)

    def _direct_startup_info(self, component: str, duration: float, details: str = ""):
        """Display startup component information."""
        with self._console_lock:
            status_text = Text()
            status_text.append("✓ ", style="bold green")
            status_text.append(f"{component} ", style="bold white")
            status_text.append(f"({duration:.2f}s)", style="dim")
            if details:
                status_text.append(f" - {details}", style="cyan")
            self.console.print(status_text)

    def startup_info(self, component: str, duration: float, details: str = ""):
        """Queue or print startup info."""
        self._queue_or_print('startup_info', component, duration, details)

    def _direct_cache_stats(self, stats: dict):
        """Display cache statistics."""
        with self._console_lock:
            hits = stats.get('hits', 0)
            misses = stats.get('misses', 0)
            hit_rate = stats.get('hit_rate', '0%')
            size = stats.get('size', 0)
            capacity = stats.get('capacity', 0)

            stats_text = Text()
            stats_text.append("Cache: ", style="bold white")
            stats_text.append(f"{hit_rate} hit rate ", style="green")
            stats_text.append(f"({hits} hits, {misses} misses) ", style="dim")
            stats_text.append(f"[{size}/{capacity}]", style="cyan")

            self.console.print(stats_text)

    def cache_stats(self, stats: dict):
        """Queue or print cache statistics."""
        self._queue_or_print('cache_stats', stats)
