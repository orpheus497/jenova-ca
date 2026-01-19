##Script function and purpose: Main Textual TUI application for JENOVA
"""
JENOVA TUI Application

Textual-based terminal UI for JENOVA.
Replaces the legacy Go BubbleTea implementation.
Provides a modern, accessible chat interface with
visual hierarchy and responsive design.

Design System Tokens (Textual CSS Variables):
============================================

Colors:
  - $primary:           Cyan base - AI identity, interactive elements
  - $primary-lighten-1: Lighter cyan - AI message text
  - $primary-darken-1:  Darker cyan - Borders, dividers
  - $primary-darken-2:  Deep cyan - Header background
  - $primary-darken-3:  Deepest cyan - Footer background
  - $success:           Green - User messages, confirmations
  - $warning:           Yellow - System messages, notices
  - $error:             Red - Error messages, alerts
  - $text:              Default text color
  - $text-muted:        Subdued text for hints, labels
  - $surface:           Base background color
  - $surface-darken-1:  Darker surface for sections

Spacing (consistent units):
  - Padding: 0, 1 (standard), 2 (spacious)
  - Margin: 0, 1 (standard)
  - All values use Textual's unit system

Animation Timing:
  - Fast: 80ms (spinner frames)
  - Normal: 100ms (default animations)
  - Slow: 150ms (emphasis animations)
  - Transitions: 150-200ms ease-in-out
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual.binding import Binding

from jenova.ui.components import (
    Banner,
    WelcomePanel,
    StatusBar,
    HelpPanel,
    MessageType,
)

if TYPE_CHECKING:
    from jenova.config.models import JenovaConfig
    from jenova.core.engine import CognitiveEngine


##Class purpose: Main JENOVA TUI application
class JenovaApp(App):
    """
    JENOVA Terminal User Interface.
    
    Provides an interactive chat interface for JENOVA
    with styled messages, loading indicators, and
    command support.
    """
    
    ##Step purpose: Define application title and subtitle
    TITLE = "JENOVA"
    SUB_TITLE = "Self-Aware AI Cognitive Architecture"
    
    ##Step purpose: Define comprehensive app-wide CSS with responsive design
    CSS = """
    /* Layout purpose: Main screen layout with flex distribution */
    Screen {
        layout: vertical;
        background: $surface;
        /* Layout purpose: Minimum width for readable content */
        min-width: 40;
    }
    
    /* Style purpose: Custom header styling */
    Header {
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
    }
    
    /* Layout purpose: Main content container */
    #main-container {
        height: 1fr;
        width: 100%;
        padding: 0;
        /* Layout purpose: Minimum height for usable interface */
        min-height: 10;
    }
    
    /* Style purpose: Banner section at top - responsive */
    #banner-section {
        height: auto;
        width: 100%;
        content-align: center middle;
        padding: 1;
        border-bottom: solid $primary-darken-1;
        /* Layout purpose: Allow shrinking in small terminals */
        max-height: 12;
        overflow: hidden;
    }
    
    /* Layout purpose: Scrollable chat output area */
    #output-container {
        height: 1fr;
        width: 100%;
        border: round $primary-darken-1;
        border-title-align: left;
        border-title-color: $primary;
        border-title-style: bold;
        padding: 0 1;
        margin: 0 1;
        /* Layout purpose: Minimum height for message visibility */
        min-height: 5;
    }
    
    /* Style purpose: Chat output with proper scrolling */
    #output {
        height: auto;
        min-height: 100%;
        padding: 1;
        scrollbar-gutter: stable;
    }
    
    /* Layout purpose: Status bar positioning */
    #status-bar {
        height: 1;
        width: 100%;
        dock: bottom;
        background: $surface-darken-1;
        padding: 0 1;
    }
    
    /* Layout purpose: Input section at bottom - always visible */
    #input-section {
        height: auto;
        width: 100%;
        padding: 1;
        background: $surface-darken-1;
        border-top: solid $primary-darken-2;
        /* Layout purpose: Prevent input from being hidden */
        min-height: 3;
    }
    
    /* Style purpose: Input field with focus styling */
    #input {
        width: 100%;
        border: round $primary-darken-1;
        background: $surface;
        padding: 0 1;
    }
    
    #input:focus {
        border: round $primary;
    }
    
    /* Style purpose: Input label/prompt - hide on very small screens */
    #input-label {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0 1;
        margin-bottom: 0;
    }
    
    /* Style purpose: Footer with key binding hints */
    Footer {
        background: $primary-darken-3;
        color: $text-muted;
    }
    
    /* Style purpose: Help panel overlay - responsive sizing */
    #help-overlay {
        display: none;
        layer: overlay;
        width: 80%;
        height: 80%;
        margin: 2 auto;
        border: double $primary;
        background: $surface;
        padding: 1;
        overflow-y: auto;
        /* Layout purpose: Constrain overlay to reasonable bounds */
        min-width: 50;
        max-width: 100;
    }
    
    #help-overlay.visible {
        display: block;
    }
    
    /* Animation purpose: Message entry animation */
    .message-new {
        opacity: 0;
    }
    
    .message-visible {
        opacity: 1;
    }
    
    /* Layout purpose: Compact mode for very small terminals (<60 cols) */
    Screen.-compact #banner-section {
        display: none;
    }
    
    Screen.-compact #input-label {
        display: none;
    }
    """
    
    ##Step purpose: Define key bindings for accessibility
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("f1", "toggle_help", "Help"),
        Binding("escape", "close_overlay", "Close", show=False),
    ]
    
    ##Method purpose: Initialize app with optional config
    def __init__(self, config: "JenovaConfig | None" = None) -> None:
        """
        Initialize JENOVA app.
        
        Args:
            config: Optional configuration.
        """
        super().__init__()
        self._config = config
        self._engine: "CognitiveEngine | None" = None
        self._help_visible = False
    
    ##Method purpose: Compose the UI layout
    def compose(self) -> ComposeResult:
        """Create child widgets for the application."""
        ##Step purpose: Yield header
        yield Header()
        
        ##Step purpose: Main content container
        with Container(id="main-container"):
            ##Step purpose: Banner section with logo
            with Vertical(id="banner-section"):
                yield Banner(show_attribution=True)
                yield WelcomePanel()
            
            ##Step purpose: Scrollable output area with border title
            with ScrollableContainer(id="output-container"):
                yield RichLog(
                    id="output",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    auto_scroll=True,
                )
            
            ##Step purpose: Status bar for loading state
            yield StatusBar(id="status-bar")
            
            ##Step purpose: Input section
            with Vertical(id="input-section"):
                yield Static(
                    "Type your message or /help for commands:",
                    id="input-label",
                )
                yield Input(
                    placeholder="Enter your message...",
                    id="input",
                )
        
        ##Step purpose: Help overlay (hidden by default)
        with ScrollableContainer(id="help-overlay"):
            yield HelpPanel()
        
        ##Step purpose: Footer with key hints
        yield Footer()
    
    ##Method purpose: Handle app startup
    def on_mount(self) -> None:
        """Called when app is mounted."""
        ##Action purpose: Set border title for output container
        output_container = self.query_one("#output-container", ScrollableContainer)
        output_container.border_title = "Chat"
        
        ##Action purpose: Check terminal size and enable compact mode if needed
        self._check_terminal_size()
        
        ##Action purpose: Focus input field
        self.query_one("#input", Input).focus()
    
    ##Method purpose: Check terminal size and toggle compact mode
    def _check_terminal_size(self) -> None:
        """Enable compact mode for small terminals."""
        ##Step purpose: Get terminal dimensions
        width = self.console.size.width
        height = self.console.size.height
        
        ##Condition purpose: Enable compact mode for narrow terminals
        if width < 60 or height < 20:
            self.screen.add_class("-compact")
        else:
            self.screen.remove_class("-compact")
    
    ##Method purpose: Handle terminal resize events
    def on_resize(self) -> None:
        """Handle terminal resize."""
        ##Action purpose: Re-check terminal size for compact mode
        self._check_terminal_size()
    
    ##Method purpose: Handle input submission
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input."""
        ##Step purpose: Get and clear input
        input_widget = self.query_one("#input", Input)
        message = event.value.strip()
        input_widget.value = ""
        
        ##Condition purpose: Ignore empty messages
        if not message:
            return
        
        ##Step purpose: Handle special commands
        if message.lower() in ("exit", "quit"):
            self.exit()
            return
        
        if message.lower() == "/help":
            self.action_toggle_help()
            return
        
        ##Step purpose: Get widgets for output
        output = self.query_one("#output", RichLog)
        status_bar = self.query_one("#status-bar", StatusBar)
        
        ##Action purpose: Display user message with styling
        output.write(
            f"[bold green]You:[/bold green] {message}"
        )
        
        ##Action purpose: Set loading state
        status_bar.set_loading("Thinking")
        
        ##Step purpose: Generate response
        try:
            ##Condition purpose: Check if engine is available
            if self._engine is not None:
                response = await self._generate_response(message)
            else:
                ##Step purpose: Placeholder response for unconnected engine
                response = f"Echo: {message}"
            
            ##Action purpose: Display AI response with styling
            output.write(
                f"[bold cyan]JENOVA:[/bold cyan] {response}"
            )
            output.write("")  ##Step purpose: Add spacing between exchanges
            
            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")
            
        except Exception as e:
            ##Error purpose: Display error message
            output.write(
                f"[bold red]ERROR:[/bold red] {e}"
            )
            status_bar.set_error(f"Error: {type(e).__name__}")
    
    ##Method purpose: Generate response using engine
    async def _generate_response(self, message: str) -> str:
        """Generate response using cognitive engine.
        
        Args:
            message: User message to process.
            
        Returns:
            Generated response content.
        """
        ##Condition purpose: Verify engine is connected
        if self._engine is None:
            return "[No engine connected] Echo: " + message
        
        ##Step purpose: Import for async execution
        import asyncio
        
        ##Action purpose: Run cognitive cycle in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._engine.think,
            message,
        )
        
        ##Condition purpose: Handle error results
        if result.is_error:
            return f"[Error] {result.error_message}"
        
        return result.content
    
    ##Method purpose: Handle clear action
    def action_clear(self) -> None:
        """Clear the output log."""
        output = self.query_one("#output", RichLog)
        output.clear()
        
        ##Action purpose: Show cleared message
        output.write("[dim]>> Chat cleared[/dim]")
    
    ##Method purpose: Toggle help overlay visibility
    def action_toggle_help(self) -> None:
        """Toggle the help panel overlay."""
        help_overlay = self.query_one("#help-overlay", ScrollableContainer)
        
        ##Condition purpose: Toggle visibility class
        if self._help_visible:
            help_overlay.remove_class("visible")
            self._help_visible = False
            self.query_one("#input", Input).focus()
        else:
            help_overlay.add_class("visible")
            self._help_visible = True
    
    ##Method purpose: Close any open overlay
    def action_close_overlay(self) -> None:
        """Close any open overlay."""
        ##Condition purpose: Close help if visible
        if self._help_visible:
            self.action_toggle_help()
    
    ##Method purpose: Add a system message to output
    def add_system_message(self, message: str) -> None:
        """
        Add a system message to the output.
        
        Args:
            message: The system message to display.
        """
        output = self.query_one("#output", RichLog)
        output.write(f"[bold yellow]>>[/bold yellow] {message}")
    
    ##Method purpose: Add an info message to output
    def add_info_message(self, message: str) -> None:
        """
        Add an info message to the output.
        
        Args:
            message: The info message to display.
        """
        output = self.query_one("#output", RichLog)
        output.write(f"[dim]>>[/dim] {message}")
    
    ##Method purpose: Connect cognitive engine
    def set_engine(self, engine: "CognitiveEngine") -> None:
        """
        Connect the cognitive engine.
        
        Args:
            engine: The CognitiveEngine instance.
        """
        self._engine = engine


##Function purpose: Run the TUI application
def run_tui(config: "JenovaConfig | None" = None) -> None:
    """
    Run the JENOVA TUI.
    
    Args:
        config: Optional configuration.
    """
    app = JenovaApp(config)
    app.run()
