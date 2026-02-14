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
  - Transitions: 150-200ms in_out_cubic
"""

from __future__ import annotations

import getpass
import logging
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from jenova.assumptions.types import Assumption
from jenova.exceptions import AssumptionDuplicateError
from jenova.llm.types import GenerationParams
from jenova.memory.types import MemoryType
from jenova.ui.components import (
    Banner,
    HelpPanel,
    StatusBar,
    WelcomePanel,
)
from jenova.utils.logging import get_logger
from jenova.utils.validation import validate_username

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
        align: center middle;
        width: auto;
        height: 80%;
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
    def __init__(self, config: JenovaConfig | None = None) -> None:
        """
        Initialize JENOVA app.

        Args:
            config: Optional configuration.
        """
        super().__init__()
        self._config = config
        self._engine: CognitiveEngine | None = None
        self._help_visible = False
        self._logger = get_logger(__name__)

        ##Step purpose: Initialize interactive state management
        self._interactive_mode: str = "normal"
        self._pending_assumption: Assumption | None = None
        self._procedure_name: str | None = None
        self._procedure_steps: list[str] = []
        self._procedure_outcome: str | None = None

        ##Step purpose: Get current username for user-specific operations
        try:
            self._username = validate_username(getpass.getuser())
        except Exception:
            ##Step purpose: Fallback to default if username validation fails
            self._username = "default"

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

        ##Step purpose: Get widgets for output and status
        output = self.query_one("#output", RichLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        ##Step purpose: Handle interactive mode first (before commands)
        if self._interactive_mode != "normal":
            await self._handle_interactive_input(message, output, status_bar)
            return

        ##Step purpose: Handle special commands
        message = message.lstrip()
        msg_lower = message.lower()
        if msg_lower in ("exit", "quit"):
            self.exit()
            return

        if msg_lower == "/help":
            self.action_toggle_help()
            return

        ##Update: Add /reset command handler
        if msg_lower == "/reset":
            self._handle_reset()
            return

        ##Update: Add /debug command handler
        if msg_lower == "/debug":
            self._handle_debug()
            return

        ##Step purpose: Handle Phase 1 commands (simple, no interactive flow)
        if msg_lower == "/insight":
            await self._handle_insight_command(output, status_bar)
            return

        if msg_lower == "/reflect":
            await self._handle_reflect_command(output, status_bar)
            return

        if msg_lower == "/memory-insight":
            await self._handle_memory_insight_command(output, status_bar)
            return

        if msg_lower == "/meta":
            await self._handle_meta_command(output, status_bar)
            return

        ##Update: WIRING-008 (2026-02-14) - Add /assume command
        if msg_lower == "/assume" or msg_lower.startswith("/assume "):
            await self._handle_assume_command(message, output, status_bar)
            return

        if msg_lower == "/train":
            self._handle_train_command(output)
            return

        ##Step purpose: Handle Phase 2 commands (interactive flows)
        if msg_lower == "/verify":
            await self._handle_verify_command(output, status_bar)
            return

        if msg_lower == "/develop_insight" or msg_lower.startswith("/develop_insight "):
            await self._handle_develop_insight_command(message, output, status_bar)
            return

        if msg_lower == "/learn_procedure":
            self._handle_learn_procedure_start(output)
            return

        ##Action purpose: Display user message with styling
        output.write(f"[bold green]You:[/bold green] {message}")

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
            output.write(f"[bold cyan]JENOVA:[/bold cyan] {response}")
            output.write("")  ##Step purpose: Add spacing between exchanges

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]ERROR:[/bold red] {e}")
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
        loop = asyncio.get_running_loop()
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

    ##Method purpose: Handle /reset command
    def _handle_reset(self) -> None:
        """Handle /reset command to clear conversation state."""
        ##Step purpose: Get output widget for feedback
        output = self.query_one("#output", RichLog)

        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write("[bold yellow]>>[/bold yellow] Cannot reset: No engine connected.")
            return

        ##Action purpose: Reset engine state
        self._engine.reset()

        ##Action purpose: Clear output display
        output.clear()

        ##Action purpose: Show reset confirmation
        output.write("[bold green]>>[/bold green] Conversation reset. Starting fresh.")
        output.write("")  ##Step purpose: Add spacing

        ##Action purpose: Log reset action
        self._logger.info("conversation_reset_via_tui")

    ##Method purpose: Handle /debug command
    def _handle_debug(self) -> None:
        """Handle /debug command to toggle debug logging."""
        ##Step purpose: Get output widget for feedback
        output = self.query_one("#output", RichLog)

        ##Step purpose: Get root logger to check current level
        root_logger = logging.getLogger()
        current_level = root_logger.level

        ##Condition purpose: Toggle between DEBUG and INFO
        if current_level == logging.DEBUG:
            ##Action purpose: Disable debug logging
            root_logger.setLevel(logging.INFO)
            new_level = "INFO"
            self._logger.info("debug_logging_disabled")
        else:
            ##Action purpose: Enable debug logging
            root_logger.setLevel(logging.DEBUG)
            new_level = "DEBUG"
            self._logger.debug("debug_logging_enabled")

        ##Action purpose: Show status message
        output.write(f"[bold yellow]>>[/bold yellow] Debug mode: {new_level.lower()}")
        output.write("")  ##Step purpose: Add spacing

    ##Method purpose: Connect cognitive engine
    def set_engine(self, engine: CognitiveEngine) -> None:
        """
        Connect the cognitive engine.

        Args:
            engine: The CognitiveEngine instance.
        """
        self._engine = engine

    ##Method purpose: Handle interactive input based on current mode
    async def _handle_interactive_input(
        self,
        message: str,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle user input when in interactive mode.

        Args:
            message: User input message
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Handle verification mode
        if self._interactive_mode == "verify":
            await self._handle_verify_response(message, output, status_bar)
            return

        ##Condition purpose: Handle procedure learning modes
        if self._interactive_mode == "learn_procedure_name":
            self._handle_procedure_name(message, output)
            return

        if self._interactive_mode == "learn_procedure_steps":
            self._handle_procedure_step(message, output)
            return

        if self._interactive_mode == "learn_procedure_outcome":
            await self._handle_procedure_outcome(message, output, status_bar)
            return

        ##Error purpose: Unknown interactive mode - reset to normal
        self._logger.warning("unknown_interactive_mode", mode=self._interactive_mode)
        self._interactive_mode = "normal"
        output.write("[bold yellow]>>[/bold yellow] Unknown interactive mode. Resetting to normal.")

    ##Method purpose: Handle /insight command - Generate insights from conversation
    async def _handle_insight_command(
        self,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle /insight command to generate insights from conversation.

        Args:
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write(
                "[bold yellow]>>[/bold yellow] Cannot generate insights: No engine connected."
            )
            return

        ##Condition purpose: Check if insight manager is available
        if self._engine.insight_manager is None:
            output.write("[bold yellow]>>[/bold yellow] Insight manager not available.")
            return

        ##Action purpose: Set loading state
        status_bar.set_loading("Generating insights...")

        ##Error purpose: Handle errors during insight generation
        try:
            ##Step purpose: Get conversation history from engine (need to access private _history)
            ##Note: Engine doesn't expose history directly, so we'll generate from recent context
            ##For now, we'll use LLM to generate insight from a prompt about the conversation

            ##Step purpose: Use LLM to generate insight content
            prompt_text = """Analyze the recent conversation and extract key insights or takeaways.
Generate a concise insight (1-2 sentences) that captures an important pattern, conclusion, or understanding from the conversation.
Focus on novel observations, not just summaries."""

            ##Step purpose: Generate insight using LLM
            import asyncio

            loop = asyncio.get_running_loop()
            insight_content = await loop.run_in_executor(
                None,
                self._engine.llm.generate_text,
                prompt_text,
                "You are an expert at identifying patterns and extracting insights from conversations.",
                GenerationParams(max_tokens=256, temperature=0.7),
            )

            ##Step purpose: Save insight using InsightManager
            insight = self._engine.insight_manager.save_insight(
                content=insight_content.strip(),
                username=self._username,
                topic="conversation",
            )

            ##Action purpose: Display success message
            output.write("[bold green]>>[/bold green] Insight generated and saved:")
            output.write(f"[white]  {insight.content}[/white]")
            output.write("")  ##Step purpose: Add spacing

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error generating insight: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("insight_generation_failed", error=str(e), exc_info=True)

    ##Method purpose: Handle /reflect command - Deep reflection on cognitive graph
    async def _handle_reflect_command(
        self,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle /reflect command to perform deep reflection.

        Args:
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write("[bold yellow]>>[/bold yellow] Cannot reflect: No engine connected.")
            return

        ##Action purpose: Set loading state
        status_bar.set_loading("Reflecting...")

        ##Error purpose: Handle errors during reflection
        try:
            ##Step purpose: Get graph and LLM from engine
            graph = self._engine.knowledge_store.graph

            ##Step purpose: Import for async execution
            import asyncio

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: graph.reflect(self._username, self._engine.llm),
            )

            ##Action purpose: Display reflection results
            output.write("[bold green]>>[/bold green] Reflection complete:")
            output.write(f"[white]  - Orphans linked: {result.get('orphans_linked', 0)}[/white]")
            output.write(f"[white]  - Clusters found: {result.get('clusters_found', 0)}[/white]")
            insights = result.get("insights_generated", [])
            output.write(f"[white]  - Meta-insights generated: {len(insights)}[/white]")
            if insights:
                for insight in insights[:3]:  ##Step purpose: Show first 3 insights
                    output.write(f"[dim]    • {insight[:100]}...[/dim]")
            output.write("")  ##Step purpose: Add spacing

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error during reflection: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("reflection_failed", error=str(e), exc_info=True)

    ##Method purpose: Handle /memory-insight command - Generate insights from memory
    async def _handle_memory_insight_command(
        self,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle /memory-insight command to generate insights from memory.

        Args:
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write(
                "[bold yellow]>>[/bold yellow] Cannot generate memory insights: No engine connected."
            )
            return

        ##Condition purpose: Check if insight manager is available
        if self._engine.insight_manager is None:
            output.write("[bold yellow]>>[/bold yellow] Insight manager not available.")
            return

        ##Action purpose: Set loading state
        status_bar.set_loading("Searching memory...")

        ##Error purpose: Handle errors during memory insight generation
        try:
            ##Step purpose: Search all memory types
            knowledge_store = self._engine.knowledge_store
            memory_results = knowledge_store.search(
                query="patterns insights conclusions",
                memory_types=list(MemoryType),
                n_results=10,
            )

            ##Condition purpose: Check if any memories found
            if not memory_results.memories:
                output.write("[bold yellow]>>[/bold yellow] No memories found to analyze.")
                status_bar.set_ready("Ready")
                return

            ##Step purpose: Use LLM to generate insight from memory patterns
            memory_content = "\n".join(
                f"- [{m.memory_type.value}] {m.content[:100]}" for m in memory_results.memories[:5]
            )

            prompt_text = f"""Analyze these memories and extract a key insight or pattern.
Look for connections, themes, or conclusions that emerge from this knowledge.

Memories:
{memory_content}

Generate a concise insight (1-2 sentences) that reveals a pattern or conclusion:"""

            ##Step purpose: Generate insight using LLM
            import asyncio

            loop = asyncio.get_running_loop()
            insight_content = await loop.run_in_executor(
                None,
                self._engine.llm.generate_text,
                prompt_text,
                "You are an expert at identifying patterns across diverse information.",
                GenerationParams(max_tokens=256, temperature=0.7),
            )

            ##Step purpose: Save insight
            insight = self._engine.insight_manager.save_insight(
                content=insight_content.strip(),
                username=self._username,
                topic="memory_analysis",
            )

            ##Action purpose: Display success message
            output.write("[bold green]>>[/bold green] Memory insight generated:")
            output.write(f"[white]  {insight.content}[/white]")
            output.write("")  ##Step purpose: Add spacing

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error generating memory insight: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("memory_insight_generation_failed", error=str(e), exc_info=True)

    ##Method purpose: Handle /meta command - Generate meta-insights from clusters
    async def _handle_meta_command(
        self,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle /meta command to generate meta-insights.

        Args:
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write(
                "[bold yellow]>>[/bold yellow] Cannot generate meta-insights: No engine connected."
            )
            return

        ##Action purpose: Set loading state
        status_bar.set_loading("Generating meta-insights...")

        ##Error purpose: Handle errors during meta-insight generation
        try:
            ##Step purpose: Get graph and LLM from engine
            graph = self._engine.knowledge_store.graph

            ##Step purpose: Import for async execution
            import asyncio

            loop = asyncio.get_running_loop()
            meta_insights = await loop.run_in_executor(
                None,
                lambda: graph.generate_meta_insights(self._username, self._engine.llm),
            )

            ##Condition purpose: Check if any meta-insights generated
            if not meta_insights:
                output.write(
                    "[bold yellow]>>[/bold yellow] No meta-insights generated. Need more connected knowledge nodes."
                )
                status_bar.set_ready("Ready")
                return

            ##Action purpose: Display meta-insights
            output.write(
                f"[bold green]>>[/bold green] Generated {len(meta_insights)} meta-insight(s):"
            )
            for idx, insight in enumerate(meta_insights, 1):
                output.write(f"[white]  {idx}. {insight}[/white]")
            output.write("")  ##Step purpose: Add spacing

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error generating meta-insights: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("meta_insight_generation_failed", error=str(e), exc_info=True)

    ##Method purpose: Handle /train command - Show fine-tuning instructions
    def _handle_train_command(self, output: RichLog) -> None:
        """Handle /train command to show fine-tuning instructions.

        Args:
            output: Output widget for feedback
        """
        output.write("[bold yellow]>>[/bold yellow] Fine-tuning Training Data:")
        output.write("[white]  To create training data for fine-tuning, run:[/white]")
        output.write("[dim]    python3 finetune/train.py[/dim]")
        output.write("")  ##Step purpose: Add spacing

    ##Update: WIRING-008 (2026-02-14) - Implementation of /assume command
    ##Method purpose: Handle /assume command - Manually add an assumption
    async def _handle_assume_command(
        self,
        message: str,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle /assume command to manually add an assumption.

        Args:
            message: Full command message
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write(
                "[bold yellow]>>[/bold yellow] Cannot add assumption: No engine connected."
            )
            return

        ##Condition purpose: Check if assumption manager is available
        if self._engine.assumption_manager is None:
            output.write("[bold yellow]>>[/bold yellow] Assumption manager not available.")
            return

        ##Step purpose: Parse content
        parts = message.split(" ", 1)
        if len(parts) < 2:
            output.write("[bold yellow]>>[/bold yellow] Usage: /assume <assumption_text>")
            return

        content = parts[1].strip()
        if not content:
            output.write("[bold yellow]>>[/bold yellow] Usage: /assume <assumption_text>")
            return

        ##Action purpose: Set loading state
        status_bar.set_loading("Adding assumption...")

        ##Error purpose: Handle errors
        try:
            import asyncio

            loop = asyncio.get_running_loop()

            ##Step purpose: Add assumption
            cortex_id = await loop.run_in_executor(
                None,
                self._engine.assumption_manager.add_assumption,
                content,
                self._username,
            )

            ##Action purpose: Display success message
            output.write("[bold green]>>[/bold green] Assumption added:")
            output.write(f"[white]  {content}[/white]")
            output.write(f"[dim]  ID: {cortex_id}[/dim]")
            output.write("")

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        ##Refactor: Catch duplicate assumption explicitly for friendly UX (D3-2026-02-14T10:24:30Z)
        except AssumptionDuplicateError as dup_err:
            output.write("[bold yellow]>>[/bold yellow] This assumption already exists.")
            status_bar.set_ready("Ready")
            self._logger.warning(
                "assume_command_duplicate",
                assumption=content,
                error=str(dup_err),
            )

        except Exception as e:
            output.write(f"[bold red]>>[/bold red] Error adding assumption: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("assume_command_failed", error=str(e), exc_info=True)

    ##Method purpose: Handle /verify command - Start assumption verification
    async def _handle_verify_command(
        self,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle /verify command to start assumption verification.

        Args:
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write(
                "[bold yellow]>>[/bold yellow] Cannot verify assumptions: No engine connected."
            )
            return

        ##Condition purpose: Check if assumption manager is available
        if self._engine.assumption_manager is None:
            output.write("[bold yellow]>>[/bold yellow] Assumption manager not available.")
            return

        ##Action purpose: Set loading state
        status_bar.set_loading("Checking assumptions...")

        ##Error purpose: Handle errors during verification
        try:
            ##Step purpose: Get assumption to verify
            import asyncio

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._engine.assumption_manager.get_assumption_to_verify,
                self._username,
            )

            ##Condition purpose: Check if assumption found
            if not result:
                output.write("[bold yellow]>>[/bold yellow] No unverified assumptions to verify.")
                status_bar.set_ready("Ready")
                return

            assumption, question = result

            ##Action purpose: Display verification question
            output.write("[bold cyan]>>[/bold cyan] JENOVA is asking for clarification:")
            output.write(f"[white]  {question}[/white]")
            output.write("[dim]  Please respond with 'yes' or 'no':[/dim]")
            output.write("")  ##Step purpose: Add spacing

            ##Step purpose: Set interactive mode
            self._interactive_mode = "verify"
            self._pending_assumption = assumption

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error starting verification: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("verification_start_failed", error=str(e), exc_info=True)

    ##Method purpose: Handle verification response
    async def _handle_verify_response(
        self,
        message: str,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle yes/no response for assumption verification.

        Args:
            message: User response
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Step purpose: Normalize response
        response = message.strip().lower()

        ##Condition purpose: Validate response
        valid_yes = ["yes", "y"]
        valid_no = ["no", "n"]

        if response not in valid_yes + valid_no:
            output.write("[bold yellow]>>[/bold yellow] Please respond with 'yes' or 'no':")
            return

        ##Action purpose: Set loading state
        status_bar.set_loading("Recording verification...")

        ##Error purpose: Handle errors during resolution
        try:
            ##Condition purpose: Check if assumption is still pending
            if self._pending_assumption is None:
                output.write(
                    "[bold yellow]>>[/bold yellow] No pending assumption. Verification cancelled."
                )
                self._interactive_mode = "normal"
                status_bar.set_ready("Ready")
                return

            ##Step purpose: Resolve assumption
            import asyncio

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._engine.assumption_manager.resolve_assumption,
                self._pending_assumption,
                response,
                self._username,
            )

            ##Action purpose: Display success message
            is_true = response in valid_yes
            status_text = "confirmed" if is_true else "denied"
            output.write(f"[bold green]>>[/bold green] Assumption {status_text}. Thank you!")
            output.write("")  ##Step purpose: Add spacing

            ##Step purpose: Reset state
            self._interactive_mode = "normal"
            self._pending_assumption = None

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error during verification: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("verification_resolution_failed", error=str(e), exc_info=True)

            ##Step purpose: Reset state on error
            self._interactive_mode = "normal"
            self._pending_assumption = None

    ##Method purpose: Handle /develop_insight command - Dual-mode insight development
    async def _handle_develop_insight_command(
        self,
        message: str,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle /develop_insight command (with or without node_id).

        Args:
            message: Full command message
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        ##Condition purpose: Check if engine is available
        if self._engine is None:
            output.write(
                "[bold yellow]>>[/bold yellow] Cannot develop insight: No engine connected."
            )
            return

        ##Condition purpose: Check if insight manager is available
        if self._engine.insight_manager is None:
            output.write("[bold yellow]>>[/bold yellow] Insight manager not available.")
            return

        ##Step purpose: Parse command arguments
        parts = message.split()
        node_id = parts[1] if len(parts) > 1 else None

        ##Action purpose: Set loading state
        status_bar.set_loading("Developing insight...")

        ##Error purpose: Handle errors during insight development
        try:
            ##Condition purpose: Mode 1: Expand existing insight with node_id
            if node_id:
                ##Step purpose: Get node from graph
                graph = self._engine.knowledge_store.graph
                node = graph.get_node(node_id)

                ##Step purpose: Use LLM to expand insight
                prompt_text = f"""Expand and develop this insight with more context and connections.
Provide additional depth, related concepts, and implications.

Original Insight: "{node.content}"

Expanded insight:"""

                import asyncio

                loop = asyncio.get_running_loop()
                expanded_content = await loop.run_in_executor(
                    None,
                    self._engine.llm.generate_text,
                    prompt_text,
                    "You are an expert at developing and expanding insights.",
                    GenerationParams(max_tokens=512, temperature=0.7),
                )

                ##Step purpose: Update node content
                graph.update_node(node_id, content=expanded_content.strip())

                ##Action purpose: Display success message
                output.write("[bold green]>>[/bold green] Insight expanded:")
                output.write(f"[white]  {expanded_content.strip()[:200]}...[/white]")
                output.write("")  ##Step purpose: Add spacing

            else:
                ##Step purpose: Mode 2: Process documents (placeholder - docs directory not implemented)
                output.write(
                    "[bold yellow]>>[/bold yellow] Document processing not yet implemented."
                )
                output.write(
                    "[dim]  Use /develop_insight [node_id] to expand an existing insight.[/dim]"
                )
                output.write("")  ##Step purpose: Add spacing

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error developing insight: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("insight_development_failed", error=str(e), exc_info=True)

    ##Method purpose: Start procedure learning flow
    def _handle_learn_procedure_start(self, output: RichLog) -> None:
        """Start the procedure learning interactive flow.

        Args:
            output: Output widget for feedback
        """
        ##Step purpose: Initialize procedure data
        self._procedure_name = None
        self._procedure_steps = []
        self._procedure_outcome = None

        ##Step purpose: Set interactive mode
        self._interactive_mode = "learn_procedure_name"

        ##Action purpose: Display start message
        output.write("[bold cyan]>>[/bold cyan] Initiating interactive procedure learning...")
        output.write("[white]  Please enter the procedure name:[/white]")
        output.write("")  ##Step purpose: Add spacing

    ##Method purpose: Handle procedure name input
    def _handle_procedure_name(self, message: str, output: RichLog) -> None:
        """Handle procedure name input.

        Args:
            message: User input
            output: Output widget for feedback
        """
        name = message.strip()

        ##Condition purpose: Validate name is not empty
        if not name:
            output.write(
                "[bold yellow]>>[/bold yellow] Procedure name cannot be empty. Please enter a name:"
            )
            return

        ##Step purpose: Store name and transition to steps
        self._procedure_name = name
        self._interactive_mode = "learn_procedure_steps"

        ##Action purpose: Display confirmation and next prompt
        output.write(f"[bold green]>>[/bold green] Procedure name set to: {name}")
        output.write(
            "[white]  Enter procedure steps one by one. Type 'done' when finished.[/white]"
        )
        output.write(f"[dim]  Step {len(self._procedure_steps) + 1}:[/dim]")
        output.write("")  ##Step purpose: Add spacing

    ##Method purpose: Handle procedure step input
    def _handle_procedure_step(self, message: str, output: RichLog) -> None:
        """Handle procedure step input.

        Args:
            message: User input
            output: Output widget for feedback
        """
        step = message.strip()

        ##Condition purpose: Check if user is done
        if step.lower() == "done":
            ##Condition purpose: Validate at least one step
            if not self._procedure_steps:
                output.write(
                    "[bold yellow]>>[/bold yellow] No steps entered. Please enter at least one step:"
                )
                return

            ##Step purpose: Transition to outcome collection
            self._interactive_mode = "learn_procedure_outcome"
            output.write(
                f"[bold green]>>[/bold green] Recorded {len(self._procedure_steps)} step(s)."
            )
            output.write("[white]  Please enter the expected outcome:[/white]")
            output.write("")  ##Step purpose: Add spacing
            return

        ##Condition purpose: Validate step is not empty
        if not step:
            output.write(
                "[bold yellow]>>[/bold yellow] Empty step entered. Please enter a step or type 'done':"
            )
            return

        ##Step purpose: Add step to list
        self._procedure_steps.append(step)

        ##Action purpose: Display confirmation and prompt for next
        output.write(
            f"[bold green]>>[/bold green] Step {len(self._procedure_steps)} recorded: {step}"
        )
        output.write(f"[dim]  Step {len(self._procedure_steps) + 1} (or type 'done'):[/dim]")
        output.write("")  ##Step purpose: Add spacing

    ##Method purpose: Handle procedure outcome and complete learning
    async def _handle_procedure_outcome(
        self,
        message: str,
        output: RichLog,
        status_bar: StatusBar,
    ) -> None:
        """Handle procedure outcome input and complete learning.

        Args:
            message: User input
            output: Output widget for feedback
            status_bar: Status bar widget
        """
        outcome = message.strip()

        ##Condition purpose: Validate outcome is not empty
        if not outcome:
            output.write(
                "[bold yellow]>>[/bold yellow] Expected outcome cannot be empty. Please enter an outcome:"
            )
            return

        ##Step purpose: Store outcome
        self._procedure_outcome = outcome

        ##Action purpose: Set loading state
        status_bar.set_loading("Saving procedure...")

        ##Error purpose: Handle errors during procedure saving
        try:
            ##Condition purpose: Check if engine is available
            if self._engine is None:
                output.write(
                    "[bold yellow]>>[/bold yellow] Cannot save procedure: No engine connected."
                )
                self._interactive_mode = "normal"
                status_bar.set_ready("Ready")
                return

            ##Step purpose: Format procedure content
            procedure_content = f"""Procedure: {self._procedure_name}

Steps:
{chr(10).join(f"{i + 1}. {step}" for i, step in enumerate(self._procedure_steps))}

Expected Outcome: {self._procedure_outcome}"""

            ##Step purpose: Save to procedural memory
            knowledge_store = self._engine.knowledge_store
            memory_id = knowledge_store.add(
                content=procedure_content,
                memory_type=MemoryType.PROCEDURAL,
                metadata={"username": self._username, "procedure_name": self._procedure_name or ""},
            )

            ##Action purpose: Display success message
            output.write(
                f"[bold green]>>[/bold green] Procedure '{self._procedure_name}' saved to memory."
            )
            output.write(f"[white]  Steps: {len(self._procedure_steps)}[/white]")
            output.write(f"[white]  Memory ID: {memory_id}[/white]")
            output.write("")  ##Step purpose: Add spacing

            ##Step purpose: Reset state
            self._interactive_mode = "normal"
            self._procedure_name = None
            self._procedure_steps = []
            self._procedure_outcome = None

            ##Action purpose: Set ready state
            status_bar.set_ready("Ready")

        except Exception as e:
            ##Error purpose: Display error message
            output.write(f"[bold red]>>[/bold red] Error saving procedure: {e}")
            status_bar.set_error(f"Error: {type(e).__name__}")
            self._logger.error("procedure_save_failed", error=str(e), exc_info=True)

            ##Step purpose: Reset state on error
            self._interactive_mode = "normal"
            self._procedure_name = None
            self._procedure_steps = []
            self._procedure_outcome = None


##Function purpose: Run the TUI application
def run_tui(config: JenovaConfig | None = None) -> None:
    """
    Run the JENOVA TUI.

    Args:
        config: Optional configuration.
    """
    app = JenovaApp(config)
    app.run()
