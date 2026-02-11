##Script function and purpose: Animated loading indicator widget for JENOVA TUI
"""
Loading indicator components for JENOVA TUI.

Provides animated spinners and progress indicators
for visual feedback during processing.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

##Step purpose: Define spinner animation frames with various styles
##Animation purpose: Braille spinner provides smooth 10-frame cycle
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
##Animation purpose: Simple dots for minimal visual distraction
DOTS_FRAMES = [".", "..", "...", ""]
##Animation purpose: Rotating circle for processing indication
PULSE_FRAMES = ["◐", "◓", "◑", "◒"]
##Animation purpose: Block-based spinner for high-contrast displays
BLOCK_FRAMES = ["▖", "▘", "▝", "▗"]
##Animation purpose: Arc spinner for elegant processing display
ARC_FRAMES = ["◜", "◠", "◝", "◞", "◡", "◟"]

##Step purpose: Define animation timing constants
SPINNER_FPS_FAST = 0.08  ##Animation purpose: 12.5fps for responsive feel
SPINNER_FPS_NORMAL = 0.1  ##Animation purpose: 10fps default
SPINNER_FPS_SLOW = 0.15  ##Animation purpose: 6.7fps for subtle animation


##Class purpose: Animated spinner widget with customizable frames
class Spinner(Static):
    """
    Animated spinner widget.

    Displays a cycling animation to indicate processing.
    Uses Braille pattern characters for smooth animation.
    """

    ##Step purpose: Define spinner-specific CSS
    DEFAULT_CSS = """
    Spinner {
        width: auto;
        height: 1;
        color: $primary;
    }

    Spinner.-spinning {
        text-style: bold;
    }
    """

    ##Step purpose: Define reactive state for animation frame
    frame_index: reactive[int] = reactive(0)
    is_spinning: reactive[bool] = reactive(False)

    ##Method purpose: Initialize spinner with optional label
    def __init__(
        self,
        label: str = "Thinking",
        frames: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        """
        Initialize the spinner.

        Args:
            label: Text to display alongside spinner.
            frames: Custom animation frames. Defaults to Braille spinner.
            **kwargs: Additional arguments passed to Static.
        """
        self._label = label
        self._frames = frames or SPINNER_FRAMES
        self._timer_handle: object | None = None

        super().__init__(**kwargs)

    ##Method purpose: Start the spinner animation
    def start(self) -> None:
        """Start the spinner animation."""
        ##Condition purpose: Don't start if already spinning
        if self.is_spinning:
            return

        self.is_spinning = True
        self.add_class("-spinning")

        ##Action purpose: Start animation timer (12.5 fps for responsive feel)
        self._timer_handle = self.set_interval(SPINNER_FPS_FAST, self._advance_frame)
        self._update_display()

    ##Method purpose: Stop the spinner animation
    def stop(self) -> None:
        """Stop the spinner animation."""
        self.is_spinning = False
        self.remove_class("-spinning")

        ##Condition purpose: Cancel timer if running
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None

        ##Action purpose: Clear display
        self.update("")

    ##Method purpose: Advance to next animation frame
    def _advance_frame(self) -> None:
        """Advance to the next animation frame."""
        self.frame_index = (self.frame_index + 1) % len(self._frames)
        self._update_display()

    ##Method purpose: Update the displayed content
    def _update_display(self) -> None:
        """Update the spinner display."""
        frame = self._frames[self.frame_index]
        self.update(f"[bold cyan]{frame}[/bold cyan] {self._label}...")

    ##Method purpose: Watch for frame changes
    def watch_frame_index(self, _old: int, _new: int) -> None:
        """React to frame index changes."""
        ##Condition purpose: Only update if spinning
        if self.is_spinning:
            self._update_display()


##Class purpose: Status bar with optional spinner integration
class StatusBar(Static):
    """
    Status bar widget with integrated spinner.

    Shows current status with optional loading animation.
    """

    ##Step purpose: Define status bar CSS with state transitions
    DEFAULT_CSS = """
    StatusBar {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        /* Animation purpose: Smooth color transitions between states */
        /* Fix: Use 'in_out_cubic' instead of 'ease-in-out' for Textual (2026-02-11T06:32:20Z) */
        transition: color 200ms in_out_cubic, background 200ms in_out_cubic;
    }

    StatusBar.-loading {
        color: $primary;
    }

    StatusBar.-error {
        color: $error;
        background: $error-darken-3;
    }

    StatusBar.-success {
        color: $success;
    }
    """

    ##Step purpose: Define reactive status state
    status_text: reactive[str] = reactive("Ready")
    is_loading: reactive[bool] = reactive(False)

    ##Method purpose: Initialize status bar
    def __init__(self, **kwargs: object) -> None:
        """Initialize the status bar."""
        super().__init__(**kwargs)
        self._spinner_frame = 0
        self._timer_handle: object | None = None

    ##Method purpose: Handle component mount
    def on_mount(self) -> None:
        """Initialize display on mount."""
        self._update_display()

    ##Method purpose: Set status to loading state
    def set_loading(self, message: str = "Thinking") -> None:
        """
        Set loading status with animation.

        Args:
            message: Loading message to display.
        """
        self.status_text = message
        self.is_loading = True
        self.add_class("-loading")
        self.remove_class("-error", "-success")

        ##Action purpose: Start spinner animation with fast frame rate
        if self._timer_handle is None:
            self._timer_handle = self.set_interval(SPINNER_FPS_FAST, self._animate_spinner)

    ##Method purpose: Set status to ready state
    def set_ready(self, message: str = "Ready") -> None:
        """
        Set ready status.

        Args:
            message: Status message to display.
        """
        self._stop_animation()
        self.status_text = message
        self.is_loading = False
        self.remove_class("-loading", "-error")
        self.add_class("-success")
        self._update_display()

        ##Action purpose: Remove success class after delay
        self.set_timer(2.0, self._clear_success)

    ##Method purpose: Set status to error state
    def set_error(self, message: str = "Error") -> None:
        """
        Set error status.

        Args:
            message: Error message to display.
        """
        self._stop_animation()
        self.status_text = message
        self.is_loading = False
        self.remove_class("-loading", "-success")
        self.add_class("-error")
        self._update_display()

    ##Method purpose: Animate the spinner
    def _animate_spinner(self) -> None:
        """Advance spinner animation frame."""
        self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
        self._update_display()

    ##Method purpose: Stop the animation timer
    def _stop_animation(self) -> None:
        """Stop the spinner animation."""
        if self._timer_handle is not None:
            self._timer_handle.stop()
            self._timer_handle = None

    ##Method purpose: Clear success styling
    def _clear_success(self) -> None:
        """Clear success class after timeout."""
        self.remove_class("-success")

    ##Method purpose: Update the display content
    def _update_display(self) -> None:
        """Update the status bar display."""
        ##Condition purpose: Show spinner if loading
        if self.is_loading:
            frame = SPINNER_FRAMES[self._spinner_frame]
            self.update(f"{frame} {self.status_text}...")
        else:
            self.update(self.status_text)

    ##Method purpose: Watch for status text changes
    def watch_status_text(self, _old: str, _new: str) -> None:
        """React to status text changes."""
        self._update_display()
