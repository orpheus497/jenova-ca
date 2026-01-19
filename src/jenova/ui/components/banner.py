##Script function and purpose: ASCII banner component for JENOVA TUI branding display
"""
Banner component for JENOVA TUI.

Provides the ASCII art logo banner with attribution
for display at application startup.
"""

from __future__ import annotations

from textual.widgets import Static


##Step purpose: Define the JENOVA ASCII art logo
JENOVA_BANNER = """
     ██╗███████╗███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ 
     ██║██╔════╝████╗  ██║██╔═══██╗██║   ██║██╔══██╗
     ██║█████╗  ██╔██╗ ██║██║   ██║██║   ██║███████║
██   ██║██╔══╝  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
╚█████╔╝███████╗██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
 ╚════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
""".strip()

##Step purpose: Define attribution text
ATTRIBUTION = "Designed by orpheus497 - https://github.com/orpheus497"


##Class purpose: ASCII art banner widget for application branding
class Banner(Static):
    """
    ASCII art banner widget.
    
    Displays the JENOVA logo with gradient coloring
    and attribution text.
    """
    
    ##Step purpose: Define banner-specific CSS styling with responsive design
    DEFAULT_CSS = """
    Banner {
        width: 100%;
        height: auto;
        content-align: center middle;
        text-align: center;
        padding: 1 0;
        margin: 0 0 1 0;
        /* Layout purpose: Constrain maximum height for very tall banners */
        max-height: 10;
        overflow: hidden;
    }
    
    Banner .banner-logo {
        color: $primary;
        text-style: bold;
    }
    
    Banner .banner-attribution {
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }
    
    /* Layout purpose: Hide full banner on very narrow terminals */
    Banner.-compact {
        height: 2;
    }
    """
    
    ##Method purpose: Initialize banner with optional custom text
    def __init__(
        self,
        show_attribution: bool = True,
        **kwargs: object,
    ) -> None:
        """
        Initialize the banner.
        
        Args:
            show_attribution: Whether to show attribution text.
            **kwargs: Additional arguments passed to Static.
        """
        self._show_attribution = show_attribution
        
        ##Step purpose: Build banner content with Rich markup
        content = self._build_banner()
        
        super().__init__(content, **kwargs)
    
    ##Method purpose: Build the banner content with styling
    def _build_banner(self) -> str:
        """Build the styled banner content."""
        ##Step purpose: Apply gradient coloring to logo
        styled_logo = f"[bold cyan]{JENOVA_BANNER}[/bold cyan]"
        
        ##Condition purpose: Add attribution if enabled
        if self._show_attribution:
            styled_attribution = f"\n[dim italic]{ATTRIBUTION}[/dim italic]"
            return styled_logo + styled_attribution
        
        return styled_logo


##Class purpose: Compact single-line title banner
class TitleBanner(Static):
    """
    Compact title banner for space-constrained layouts.
    
    Shows "JENOVA" with styling instead of full ASCII art.
    """
    
    ##Step purpose: Define compact banner CSS
    DEFAULT_CSS = """
    TitleBanner {
        width: 100%;
        height: 1;
        content-align: center middle;
        text-align: center;
        color: $primary;
        text-style: bold;
        padding: 0 1;
    }
    """
    
    ##Method purpose: Initialize compact title banner
    def __init__(self, subtitle: str | None = None, **kwargs: object) -> None:
        """
        Initialize the title banner.
        
        Args:
            subtitle: Optional subtitle text.
            **kwargs: Additional arguments passed to Static.
        """
        ##Step purpose: Build title with optional subtitle
        title = "[bold cyan]JENOVA[/bold cyan]"
        
        if subtitle:
            title = f"{title} [dim]- {subtitle}[/dim]"
        
        super().__init__(title, **kwargs)


##Class purpose: Welcome message panel with initialization info
class WelcomePanel(Static):
    """
    Welcome panel displayed after banner.
    
    Shows initialization status and usage hints.
    """
    
    ##Step purpose: Define welcome panel CSS with consistent spacing
    DEFAULT_CSS = """
    WelcomePanel {
        width: 100%;
        height: auto;
        /* Layout purpose: Consistent padding with other components */
        padding: 0 1;
        margin: 0 0 1 0;
    }
    
    WelcomePanel .welcome-ready {
        color: $success;
    }
    
    WelcomePanel .welcome-hint {
        color: $text-muted;
    }
    """
    
    ##Method purpose: Initialize welcome panel with status messages
    def __init__(self, **kwargs: object) -> None:
        """Initialize the welcome panel."""
        ##Step purpose: Build welcome content
        content = self._build_welcome()
        super().__init__(content, **kwargs)
    
    ##Method purpose: Build welcome message content
    def _build_welcome(self) -> str:
        """Build the welcome message content."""
        ##Step purpose: Build informative welcome lines with consistent styling
        lines = [
            "[bold green]>>[/bold green] Initialized and Ready.",
            "[dim]>>[/dim] Type your message and press [bold]Enter[/bold] to begin.",
            "[dim]>>[/dim] Press [bold]F1[/bold] or type [bold]/help[/bold] for commands.",
            "[dim]>>[/dim] Type [bold]exit[/bold] or press [bold]Ctrl+C[/bold] to quit.",
        ]
        return "\n".join(lines)
