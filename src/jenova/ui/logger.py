import os
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

class UILogger:
    def __init__(self):
        self.console = Console()
        try:
            self.term_width = os.get_terminal_size().columns
        except OSError:
            self.term_width = 80

    def banner(self, banner_text, attribution_text):
        self.console.clear()
        panel = Panel(
            Text(banner_text, style="bold cyan", justify="center"),
            title="Jenova Cognitive Architecture (JCA)",
            title_align="center",
            subtitle=Text(attribution_text, style="cyan", justify="center"),
            border_style="bold magenta",
            width=self.term_width
        )
        self.console.print(panel)
        self.console.print()

    def info(self, message):
        self.console.print(f"[bold green]>> {message}[/bold green]")

    def system_message(self, message):
        self.console.print(f"[bold red]>> {message}[/bold red]")

    def reflection(self, message):
        self.console.print(f"\n[italic yellow]({message})[/italic yellow]")

    @contextmanager
    def cognitive_process(self, message: str):
        with self.console.status(f"[bold green]{message}[/bold green]", spinner="earth") as status:
            yield status

    @contextmanager
    def thinking_process(self, message: str):
        with self.console.status(f"[bold yellow]{message}[/bold yellow]", spinner="dots") as status:
            yield status

    def user_query(self, text):
        panel = Panel(Text(text, style="white"), title=f"orpheus497@Jenova", border_style="dark_green", width=self.term_width)
        self.console.print(panel)

    def jenova_response(self, text):
        panel = Panel(Markdown(text, style="cyan"), title="Jenova", border_style="magenta", width=self.term_width)
        self.console.print(panel)