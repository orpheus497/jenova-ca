# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 23: Command Refactoring - Base Classes

"""
Base classes for command system.

Provides foundational abstractions for all command handlers, ensuring consistent
interfaces, error handling, and command registration.
"""

from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from abc import ABC, abstractmethod


class CommandCategory(Enum):
    """Command categories for organization."""

    SYSTEM = "system"
    NETWORK = "network"
    MEMORY = "memory"
    LEARNING = "learning"
    SETTINGS = "settings"
    HELP = "help"
    CODE = "code"
    GIT = "git"
    ANALYSIS = "analysis"
    ORCHESTRATION = "orchestration"
    AUTOMATION = "automation"


class Command:
    """
    Represents a slash command.

    Attributes:
        name: Command name (without leading slash)
        description: Human-readable description
        category: Command category for organization
        handler: Callable that executes the command
        aliases: Alternative names for this command
        usage: Usage string showing syntax
        examples: Example usages with descriptions
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: CommandCategory,
        handler: Callable,
        aliases: Optional[List[str]] = None,
        usage: Optional[str] = None,
        examples: Optional[List[str]] = None,
    ):
        """
        Initialize command.

        Args:
            name: Command name
            description: Command description
            category: Command category
            handler: Command handler function
            aliases: Alternative command names
            usage: Usage string
            examples: Example usage strings
        """
        self.name = name
        self.description = description
        self.category = category
        self.handler = handler
        self.aliases = aliases or []
        self.usage = usage or f"/{name}"
        self.examples = examples or []


class BaseCommandHandler(ABC):
    """
    Abstract base class for command handlers.

    All command handlers should inherit from this class and implement the
    register_commands() method to add their commands to the registry.

    Attributes:
        cognitive_engine: Reference to cognitive engine
        ui_logger: UI logger instance
        file_logger: File logger instance
        commands: Dictionary of commands provided by this handler
    """

    def __init__(self, cognitive_engine: Any, ui_logger: Any, file_logger: Any, **kwargs):
        """
        Initialize command handler.

        Args:
            cognitive_engine: Cognitive engine instance
            ui_logger: UI logger instance
            file_logger: File logger instance
            **kwargs: Additional handler-specific dependencies
        """
        self.cognitive_engine = cognitive_engine
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.commands: Dict[str, Command] = {}

        # Store kwargs for handler-specific dependencies
        self.kwargs = kwargs

        # Register this handler's commands
        self.register_commands()

    @abstractmethod
    def register_commands(self) -> None:
        """
        Register commands provided by this handler.

        Subclasses must implement this method to create and register
        their commands using the _register() method.
        """
        pass

    def _register(self, command: Command) -> None:
        """
        Register a command with this handler.

        Args:
            command: Command to register

        Example:
            >>> self._register(Command(
            ...     name="example",
            ...     description="Example command",
            ...     category=CommandCategory.SYSTEM,
            ...     handler=self._cmd_example
            ... ))
        """
        self.commands[command.name] = command
        # Also register aliases
        for alias in command.aliases:
            self.commands[alias] = command

    def get_commands(self) -> Dict[str, Command]:
        """
        Get all commands from this handler.

        Returns:
            Dictionary mapping command names to Command objects
        """
        return self.commands

    def _format_error(self, message: str) -> str:
        """
        Format an error message consistently.

        Args:
            message: Error message

        Returns:
            Formatted error message

        Example:
            >>> return self._format_error("Invalid argument")
        """
        return f"❌ Error: {message}"

    def _format_success(self, message: str) -> str:
        """
        Format a success message consistently.

        Args:
            message: Success message

        Returns:
            Formatted success message

        Example:
            >>> return self._format_success("Operation completed")
        """
        return f"✓ {message}"

    def _log_command_execution(self, command_name: str, args: List[str]) -> None:
        """
        Log command execution for audit trail.

        Args:
            command_name: Name of command being executed
            args: Command arguments
        """
        if self.file_logger:
            self.file_logger.log_info(
                f"Command executed: /{command_name} {' '.join(args)}"
            )
