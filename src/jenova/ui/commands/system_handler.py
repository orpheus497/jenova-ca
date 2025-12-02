# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 23: Command Refactoring - System Command Handler

"""
System command handler for The JENOVA Cognitive Architecture.

Handles system-related commands: help, status, profile, learning stats.
"""

from typing import List, Any
from jenova.ui.commands.base import Command, CommandCategory, BaseCommandHandler


class SystemCommandHandler(BaseCommandHandler):
    """Handler for system-related commands."""

    def register_commands(self) -> None:
        """Register system commands."""
        self._register(Command(
            name="profile",
            description="Show user profile information",
            category=CommandCategory.SYSTEM,
            handler=self._cmd_profile,
            usage="/profile",
        ))

        self._register(Command(
            name="learn",
            description="Show learning statistics",
            category=CommandCategory.LEARNING,
            handler=self._cmd_learn,
            usage="/learn [stats|insights|gaps|skills]",
        ))

    def _cmd_profile(self, args: List[str]) -> str:
        """Handle /profile command."""
        return "User profile information (placeholder)"

    def _cmd_learn(self, args: List[str]) -> str:
        """Handle /learn command."""
        if not args:
            return "Learning system commands: stats, insights, gaps, skills"

        subcommand = args[0].lower()
        if subcommand == "stats":
            return "Learning statistics (placeholder)"
        elif subcommand == "insights":
            return "Learning insights (placeholder)"
        elif subcommand == "gaps":
            return "Knowledge gaps (placeholder)"
        elif subcommand == "skills":
            return "Acquired skills (placeholder)"
        else:
            return f"Unknown learn subcommand: {subcommand}"
