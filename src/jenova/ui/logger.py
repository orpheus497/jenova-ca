import os
import threading
import queue
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

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
                subtitle=Text(attribution_text, style="cyan", justify="center"),
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
            self.console.print(Text.from_markup(f"[bold red]>> {message}[/bold red]"))

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

    def _direct_user_query(self, text, username: str):
        with self._console_lock:
            panel = Panel(Text(text, style="white"), title=f"{username}@JENOVA", border_style="dark_green")
            self.console.print(panel)

    def user_query(self, text, username: str):
        self._queue_or_print('user_query', text, username)

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