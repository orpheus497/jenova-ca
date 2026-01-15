##Script function and purpose: UI package for The JENOVA Cognitive Architecture
##This package provides user interface components:
##- Bubble Tea TUI integration (bubbletea.py) - The SOLE user interface
##- UI logging utilities (logger.py)
##
##Note: As of Phase A refactoring, BubbleTea is the only supported UI.
##The Python-based terminal UI has been removed in favor of the Go-based TUI.

from jenova.ui.bubbletea import BubbleTeaUI
from jenova.ui.logger import UILogger

__all__ = ['BubbleTeaUI', 'UILogger']
