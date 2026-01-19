##Script function and purpose: UI package initialization and exports
"""
UI components for JENOVA.

Provides the terminal user interface built with Textual,
including the main application and reusable components.
"""

from jenova.ui.app import JenovaApp, run_tui

__all__ = [
    "JenovaApp",
    "run_tui",
]
