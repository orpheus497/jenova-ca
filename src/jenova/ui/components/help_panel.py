##Script function and purpose: Help panel component for JENOVA command reference display
"""
Help panel component for JENOVA TUI.

Provides formatted command reference display with
categorized commands and descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    pass


##Class purpose: Data container for command information
@dataclass(frozen=True)
class CommandInfo:
    """Information about a single command."""

    name: str
    description: str
    usage: str | None = None


##Step purpose: Define command categories with detailed descriptions
##Update: All 8 missing commands now implemented - full feature parity restored
##Note: All original commands from legacy codebase are now implemented
IMPLEMENTED_COMMANDS: list[CommandInfo] = [
    CommandInfo(
        name="/help",
        description="Display this comprehensive command reference guide with all available commands and their descriptions.",
    ),
    CommandInfo(
        name="/reset",
        description="Reset the conversation state. Clears the conversation history and starts a fresh session. All session data is preserved.",
    ),
    CommandInfo(
        name="/debug",
        description="Toggle debug logging mode. Switches between INFO and DEBUG log levels for troubleshooting and development.",
    ),
    CommandInfo(
        name="/insight",
        description="Analyze the current conversation and generate new insights. JENOVA extracts key takeaways and stores them as structured insights in long-term memory.",
    ),
    CommandInfo(
        name="/reflect",
        description="Initiate deep reflection within the cognitive architecture. Reorganizes and interlinks cognitive nodes, identifies patterns, and generates higher-level meta-insights.",
    ),
    CommandInfo(
        name="/memory-insight",
        description="Perform comprehensive search across all memory layers. Scans episodic, semantic, and procedural memory to develop new insights from accumulated knowledge.",
    ),
    CommandInfo(
        name="/meta",
        description="Generate higher-level meta-insights from existing knowledge. Analyzes clusters of related insights to form abstract conclusions and identify overarching themes.",
    ),
    CommandInfo(
        name="/verify",
        description="Start the assumption verification process. JENOVA presents an unverified assumption and asks for clarification. Respond with 'yes' or 'no'.",
    ),
    CommandInfo(
        name="/develop_insight",
        description="Dual-purpose command: with node_id expands an existing insight with more context; without node_id scans docs directory for new documents to learn from.",
        usage="/develop_insight [node_id]",
    ),
    CommandInfo(
        name="/learn_procedure",
        description="Interactive guided process to teach JENOVA a new procedure. Prompts for procedure name, individual steps, and expected outcome.",
    ),
    CommandInfo(
        name="/train",
        description="Show instructions for creating fine-tuning training data from your interactions for personalizing the underlying language model.",
    ),
]

##Update: Phase 1 commands moved to IMPLEMENTED_COMMANDS (handled by other agent)
##Note: Phase 2 interactive commands now implemented
PLANNED_COGNITIVE_COMMANDS: list[CommandInfo] = []

##Update: Phase 2 interactive commands moved to IMPLEMENTED_COMMANDS
##Note: Learning commands now implemented
PLANNED_LEARNING_COMMANDS: list[CommandInfo] = []

SYSTEM_COMMANDS: list[CommandInfo] = [
    CommandInfo(
        name="exit / quit",
        description="Exit the application. All session data is automatically saved.",
    ),
]

##Step purpose: Define keyboard shortcuts for help display
KEYBOARD_SHORTCUTS: list[CommandInfo] = [
    CommandInfo(
        name="Enter",
        description="Send your message",
    ),
    CommandInfo(
        name="Ctrl+L",
        description="Clear the chat history",
    ),
    CommandInfo(
        name="F1",
        description="Toggle this help panel",
    ),
    CommandInfo(
        name="Ctrl+C",
        description="Quit the application",
    ),
    CommandInfo(
        name="Escape",
        description="Close overlay/panel",
    ),
]


##Class purpose: Formatted help panel widget
class HelpPanel(Static):
    """
    Help panel displaying command reference.

    Shows all available commands organized by category
    with descriptions and usage examples.
    """

    ##Step purpose: Define help panel CSS with consistent design tokens
    DEFAULT_CSS = """
    HelpPanel {
        width: 100%;
        height: auto;
        /* Layout purpose: Full padding for boxed content */
        padding: 1;
        border: solid $primary;
        /* Layout purpose: Vertical margin for overlay context */
        margin: 1 0;
    }

    HelpPanel .help-title {
        text-align: center;
        text-style: bold;
        /* Style purpose: Primary color for headers */
        color: $primary;
        margin-bottom: 1;
    }

    HelpPanel .help-category {
        text-style: bold;
        /* Style purpose: Warning (yellow) for section headers */
        color: $warning;
        margin-top: 1;
        margin-bottom: 0;
    }

    HelpPanel .help-separator {
        /* Style purpose: Muted for decorative elements */
        color: $text-muted;
    }

    HelpPanel .help-command {
        /* Style purpose: Success (green) for command names */
        color: $success;
    }

    HelpPanel .help-description {
        /* Style purpose: Default text for descriptions */
        color: $text;
    }
    """

    ##Method purpose: Initialize help panel
    def __init__(self, **kwargs: object) -> None:
        """Initialize the help panel."""
        content = self._build_help_content()
        super().__init__(content, **kwargs)

    ##Method purpose: Build the formatted help content
    def _build_help_content(self) -> str:
        """Build the complete help content."""
        ##Style purpose: Use consistent 64-char width for all box elements
        box_width = 64
        inner_width = box_width - 2  ##Step purpose: Account for box borders

        lines: list[str] = []

        ##Step purpose: Add header with box drawing (centered title)
        header_text = "JENOVA COMMAND REFERENCE"
        header_padding = (inner_width - len(header_text)) // 2
        header_line = (
            "║"
            + " " * header_padding
            + header_text
            + " " * (inner_width - header_padding - len(header_text))
            + "║"
        )

        lines.append(f"[bold cyan]╔{'═' * inner_width}╗[/bold cyan]")
        lines.append(f"[bold cyan]{header_line}[/bold cyan]")
        lines.append(f"[bold cyan]╚{'═' * inner_width}╝[/bold cyan]")
        lines.append("")

        ##Step purpose: Add implemented commands section
        lines.append("[bold yellow]IMPLEMENTED COMMANDS[/bold yellow]")
        lines.append(f"[dim]{'─' * box_width}[/dim]")
        lines.extend(self._format_commands(IMPLEMENTED_COMMANDS))
        lines.append("")

        ##Step purpose: Planned sections removed - all commands now implemented
        ##Note: Keeping structure for future expansion if needed

        ##Step purpose: Add system commands section
        lines.append("[bold yellow]SYSTEM COMMANDS[/bold yellow]")
        lines.append(f"[dim]{'─' * box_width}[/dim]")
        lines.extend(self._format_commands(SYSTEM_COMMANDS))
        lines.append("")

        ##Step purpose: Add keyboard shortcuts section
        lines.append("[bold yellow]KEYBOARD SHORTCUTS[/bold yellow]")
        lines.append(f"[dim]{'─' * box_width}[/dim]")
        lines.extend(self._format_shortcuts(KEYBOARD_SHORTCUTS))
        lines.append("")

        ##Step purpose: Add tip footer with consistent width
        tip_text = "Tip: Commands are not stored in conversational memory."
        tip_padding = (inner_width - len(tip_text)) // 2
        tip_line = (
            "║"
            + " " * tip_padding
            + tip_text
            + " " * (inner_width - tip_padding - len(tip_text))
            + "║"
        )

        lines.append(f"[dim]╔{'═' * inner_width}╗[/dim]")
        lines.append(f"[dim]{tip_line}[/dim]")
        lines.append(f"[dim]╚{'═' * inner_width}╝[/dim]")

        return "\n".join(lines)

    ##Method purpose: Format keyboard shortcuts for compact display
    def _format_shortcuts(self, shortcuts: list[CommandInfo]) -> list[str]:
        """Format keyboard shortcuts in a compact two-column layout."""
        lines: list[str] = []

        ##Loop purpose: Format each shortcut entry
        for shortcut in shortcuts:
            ##Step purpose: Build shortcut line with key and description
            key_display = f"[bold cyan]{shortcut.name}[/bold cyan]"
            desc_display = f"[white]{shortcut.description}[/white]"
            lines.append(f"  {key_display}: {desc_display}")

        return lines

    ##Method purpose: Format a list of commands
    def _format_commands(self, commands: list[CommandInfo]) -> list[str]:
        """Format a list of commands for display."""
        lines: list[str] = []

        ##Loop purpose: Format each command entry
        for cmd in commands:
            ##Step purpose: Build command line with name and description
            name_display = f"[bold green]{cmd.name}[/bold green]"
            desc_display = f"[white]{cmd.description}[/white]"
            lines.append(f"  {name_display}")
            lines.append(f"    {desc_display}")

            ##Condition purpose: Add usage if specified
            if cmd.usage:
                lines.append(f"    [dim]Usage: {cmd.usage}[/dim]")

        return lines


##Class purpose: Compact inline help display
class HelpHint(Static):
    """
    Compact help hint for footer display.

    Shows abbreviated command hints.
    """

    ##Step purpose: Define help hint CSS with consistent design tokens
    DEFAULT_CSS = """
    HelpHint {
        width: 100%;
        height: 1;
        text-align: center;
        /* Style purpose: Muted color for subtle hints */
        color: $text-muted;
    }
    """

    ##Method purpose: Initialize help hint
    def __init__(self, **kwargs: object) -> None:
        """Initialize the help hint."""
        hint = "[bold]/help[/bold]: commands • [bold]Enter[/bold]: send • [bold]Ctrl+C[/bold]: quit"
        super().__init__(hint, **kwargs)
