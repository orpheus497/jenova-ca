# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 23: Command Refactoring - Modular Command System

"""
Modular command system for The JENOVA Cognitive Architecture.

This module provides a refactored command architecture replacing the monolithic
1,330-line commands.py with specialized handler classes.

The command system is organized into specialized handlers:
    - SystemCommandHandler: System info, help, profile, learning stats
    - NetworkCommandHandler: Network management, peer connections
    - SettingsCommandHandler: Configuration and preferences
    - MemoryCommandHandler: Backup, export, import operations
    - CodeToolsCommandHandler: Code editing, analysis, refactoring
    - OrchestrationCommandHandler: Git, tasks, workflows, automation

Example:
    >>> from jenova.ui.commands import CommandRegistry
    >>> registry = CommandRegistry(cognitive_engine, ui_logger, file_logger)
    >>> result = registry.execute("/help")
"""

from jenova.ui.commands.base import Command, CommandCategory, BaseCommandHandler
from jenova.ui.commands.registry import CommandRegistry
from jenova.ui.commands.system_handler import SystemCommandHandler
from jenova.ui.commands.network_handler import NetworkCommandHandler
from jenova.ui.commands.settings_handler import SettingsCommandHandler
from jenova.ui.commands.memory_handler import MemoryCommandHandler
from jenova.ui.commands.code_tools_handler import CodeToolsCommandHandler
from jenova.ui.commands.orchestration_handler import OrchestrationCommandHandler

__all__ = [
    "Command",
    "CommandCategory",
    "BaseCommandHandler",
    "CommandRegistry",
    "SystemCommandHandler",
    "NetworkCommandHandler",
    "SettingsCommandHandler",
    "MemoryCommandHandler",
    "CodeToolsCommandHandler",
    "OrchestrationCommandHandler",
]
