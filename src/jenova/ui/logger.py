##Script function and purpose: UI Logger for The JENOVA Cognitive Architecture
##This module provides rich text formatting and console output with thread-safe message queuing

import os
import threading
import queue
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

##Class purpose: Handles all console output with rich formatting and thread-safe message queuing
class UILogger:
    ##Function purpose: Initialize logger with console and optional message queue
    def __init__(self, message_queue=None):
        self.console = Console()
        self._console_lock = threading.RLock()
        self.message_queue = message_queue  # Queue for non-blocking UI updates
        try:
            self.term_width = os.get_terminal_size().columns
        except OSError:
            self.term_width = 80

    ##Function purpose: Display banner directly to console (startup only)
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
                subtitle=Text(attribution_text, style="cyan", justify="center"),
                border_style="bold magenta",
            )
            self.console.print(panel)
            self.console.print()

    ##Function purpose: Public interface for displaying startup banner
    def banner(self, banner_text, attribution_text):
        # Banner is always printed directly (startup only)
        self._direct_banner(banner_text, attribution_text)

    ##Function purpose: Route messages to queue or print directly based on mode
    def _queue_or_print(self, msg_type, *args, **kwargs):
        """Helper method to queue messages when in non-blocking mode, or print directly."""
        if self.message_queue is not None:
            self.message_queue.put((msg_type, args, kwargs))
        else:
            # Direct printing with lock (legacy mode)
            getattr(self, f'_direct_{msg_type}')(*args, **kwargs)

    ##Function purpose: Display info message directly with green formatting
    def _direct_info(self, message):
        with self._console_lock:
            self.console.print(f"[bold green]>> {message}[/bold green]")

    ##Function purpose: Public interface for info messages
    def info(self, message):
        self._queue_or_print('info', message)

    ##Function purpose: Display system message directly with red formatting
    def _direct_system_message(self, message):
        with self._console_lock:
            self.console.print(Text.from_markup(f"[bold red]>> {message}[/bold red]"))

    ##Function purpose: Public interface for system messages
    def system_message(self, message):
        self._queue_or_print('system_message', message)

    ##Function purpose: Display help message directly to console
    def _direct_help_message(self, message):
        with self._console_lock:
            self.console.print(message)

    ##Function purpose: Public interface for help messages
    def help_message(self, message):
        self._queue_or_print('help_message', message)

    ##Function purpose: Display reflection message directly with yellow italic formatting
    def _direct_reflection(self, message):
        with self._console_lock:
            self.console.print(f"\n[italic yellow]({message})[/italic yellow]")

    ##Function purpose: Public interface for reflection messages
    def reflection(self, message):
        self._queue_or_print('reflection', message)

    ##Function purpose: Context manager for non-blocking cognitive process status indicator
    @contextmanager
    def cognitive_process(self, message: str):
        """Non-blocking cognitive process status indicator."""
        if self.message_queue is not None:
            # Queue-based mode: just queue the status update
            self.message_queue.put(('start_status', (message,), {'spinner': 'earth', 'style': 'bold green'}))
            try:
                yield None  # No status object in non-blocking mode
            finally:
                self.message_queue.put(('stop_status', (), {}))
        else:
            # Legacy direct mode with lock
            with self._console_lock:
                with self.console.status(f"[bold green]{message}[/bold green]", spinner="earth") as status:
                    yield status

    ##Function purpose: Context manager for non-blocking thinking process status indicator
    @contextmanager
    def thinking_process(self, message: str):
        """Non-blocking thinking process status indicator."""
        if self.message_queue is not None:
            # Queue-based mode: just queue the status update
            self.message_queue.put(('start_status', (message,), {'spinner': 'dots', 'style': 'bold yellow'}))
            try:
                yield None  # No status object in non-blocking mode
            finally:
                self.message_queue.put(('stop_status', (), {}))
        else:
            # Legacy direct mode with lock
            with self._console_lock:
                with self.console.status(f"[bold yellow]{message}[/bold yellow]", spinner="dots") as status:
                    yield status

    ##Function purpose: Display user query directly in a styled panel
    def _direct_user_query(self, text, username: str):
        with self._console_lock:
            panel = Panel(Text(text, style="white"), title=f"{username}@JENOVA", border_style="dark_green")
            self.console.print(panel)

    ##Function purpose: Public interface for user query display
    def user_query(self, text, username: str):
        self._queue_or_print('user_query', text, username)

    ##Function purpose: Display JENOVA response directly in a styled panel with markdown
    def _direct_jenova_response(self, text):
        with self._console_lock:
            if not isinstance(text, str):
                text = str(text)
            try:
                # Try to render as Markdown
                panel = Panel(Markdown(text, style="cyan"), title="JENOVA", border_style="magenta")
            except TypeError:
                # If Markdown parsing fails, render as plain text
                panel = Panel(Text(text, style="cyan"), title="JENOVA", border_style="magenta")
            self.console.print(panel)

    ##Function purpose: Public interface for JENOVA response display
    def jenova_response(self, text):
        self._queue_or_print('jenova_response', text)

    ##Function purpose: Process all queued messages from main UI thread
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