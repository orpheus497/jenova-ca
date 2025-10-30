# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 6: Enhanced UI Package

Provides terminal UI, logging, and health display components.

Components:
- UILogger: Enhanced logging with metrics and health display
- TerminalUI: Interactive terminal interface with health monitoring
- HealthDisplay: Real-time system health and metrics display
- CompactHealthDisplay: Compact single-line health status
"""

from jenova.ui.logger import UILogger
from jenova.ui.terminal import TerminalUI
from jenova.ui.health_display import HealthDisplay, CompactHealthDisplay

__version__ = '4.2.0'
__phase__ = 'Phase 6: UI and Main Entry Enhancements'

__all__ = [
    'UILogger',
    'TerminalUI',
    'HealthDisplay',
    'CompactHealthDisplay',
]
