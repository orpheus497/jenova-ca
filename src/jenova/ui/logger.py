import os
import threading
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

class UILogger:
    def __init__(self):
        self.console = Console()
        self._console_lock = threading.RLock()
        try:
            self.term_width = os.get_terminal_size().columns
        except OSError:
            self.term_width = 80

    def banner(self, banner_text, attribution_text):
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

    def info(self, message):
        with self._console_lock:
            self.console.print(f"[bold green]>> {message}[/bold green]")

    def system_message(self, message):
        with self._console_lock:
            self.console.print(Text.from_markup(f"[bold red]>> {message}[/bold red]"))

    def help_message(self, message):
        with self._console_lock:
            self.console.print(message)

    def reflection(self, message):
        with self._console_lock:
            self.console.print(f"\n[italic yellow]({message})[/italic yellow]")

    @contextmanager
    def cognitive_process(self, message: str):
        with self._console_lock:
            with self.console.status(f"[bold green]{message}[/bold green]", spinner="earth") as status:
                yield status

    @contextmanager
    def thinking_process(self, message: str):
        with self._console_lock:
            with self.console.status(f"[bold yellow]{message}[/bold yellow]", spinner="dots") as status:
                yield status

    def user_query(self, text, username: str):
        with self._console_lock:
            panel = Panel(Text(text, style="white"), title=f"{username}@JENOVA", border_style="dark_green")
            self.console.print(panel)

    def jenova_response(self, text):
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