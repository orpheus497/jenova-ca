# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 23: Command Refactoring - Command Registry

"""
Command registry for The JENOVA Cognitive Architecture.

Centralizes command registration and dispatch for all command handlers.
"""

from typing import Dict, List, Optional, Any, Callable
from jenova.ui.commands.base import Command, CommandCategory, BaseCommandHandler


class CommandRegistry:
    """
    Central registry for all commands.

    Manages command registration, lookup, and execution across all handlers.

    Attributes:
        commands: Dictionary mapping command names to Command objects
        handlers: List of registered command handlers
    """

    def __init__(
        self,
        cognitive_engine: Any,
        ui_logger: Any,
        file_logger: Optional[Any] = None,
        **kwargs: Any
    ):
        """
        Initialize command registry.

        Args:
            cognitive_engine: CognitiveEngine instance that provides AI capabilities.
                              Used for AI-assisted command execution.
            ui_logger: UILogger instance for displaying output to the user.
                       Must have methods like info(), warning(), error(), success().
            file_logger: Optional FileLogger instance for logging to files.
                         Must have methods like log_info(), log_warning(), log_error().
            **kwargs: Additional dependencies to pass to command handlers.
                      Common kwargs include: config (dict), memory_search, llm_interface.
        """
        self.cognitive_engine = cognitive_engine
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.kwargs = kwargs

        self.commands: Dict[str, Command] = {}
        self.handlers: List[BaseCommandHandler] = []

        # Register built-in commands
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in commands."""
        # Help command
        help_cmd = Command(
            name="help",
            description="Show available commands or help for a specific command",
            category=CommandCategory.HELP,
            handler=self._cmd_help,
            usage="/help [command]",
            examples=["/help", "/help status"],
        )
        self._register(help_cmd)

        # Status command
        status_cmd = Command(
            name="status",
            description="Show system status",
            category=CommandCategory.SYSTEM,
            handler=self._cmd_status,
            usage="/status",
        )
        self._register(status_cmd)

        # Exit command
        exit_cmd = Command(
            name="exit",
            description="Exit JENOVA",
            category=CommandCategory.SYSTEM,
            handler=self._cmd_exit,
            aliases=["quit", "bye"],
            usage="/exit",
        )
        self._register(exit_cmd)

    def register_handler(self, handler: BaseCommandHandler) -> None:
        """
        Register a command handler and its commands.

        Args:
            handler: Command handler to register
        """
        self.handlers.append(handler)
        for name, command in handler.get_commands().items():
            self._register(command)

    def _register(self, command: Command) -> None:
        """
        Register a single command.

        Args:
            command: Command to register
        """
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command

    def execute(self, input_str: str) -> Optional[str]:
        """
        Execute a command from input string.

        Args:
            input_str: Command input (e.g., "/help status")

        Returns:
            Command result or None if not a command
        """
        if not input_str.startswith("/"):
            return None

        parts = input_str[1:].split()
        if not parts:
            return None

        cmd_name = parts[0].lower()
        args = parts[1:]

        if cmd_name not in self.commands:
            return f"Unknown command: /{cmd_name}. Type /help for available commands."

        command = self.commands[cmd_name]

        # Log command execution
        if self.file_logger:
            self.file_logger.log_info(f"Executing command: /{cmd_name} {' '.join(args)}")

        try:
            return command.handler(args)
        except Exception as e:
            error_msg = f"Error executing /{cmd_name}: {e}"
            if self.file_logger:
                self.file_logger.log_error(error_msg)
            return f"❌ {error_msg}"

    def get_commands_by_category(self) -> Dict[CommandCategory, List[Command]]:
        """
        Get commands organized by category.

        Returns:
            Dictionary mapping categories to list of commands
        """
        by_category: Dict[CommandCategory, List[Command]] = {}

        # Use a set to avoid duplicates from aliases
        seen = set()

        for name, command in self.commands.items():
            if command.name not in seen:
                seen.add(command.name)
                if command.category not in by_category:
                    by_category[command.category] = []
                by_category[command.category].append(command)

        return by_category

    def is_command(self, input_str: str) -> bool:
        """
        Check if input string is a command.

        Args:
            input_str: Input to check

        Returns:
            True if input starts with '/' and matches a command
        """
        if not input_str.startswith("/"):
            return False

        parts = input_str[1:].split()
        if not parts:
            return False

        return parts[0].lower() in self.commands

    # Built-in command handlers

    def _cmd_help(self, args: List[str]) -> str:
        """Handle /help command."""
        if args:
            # Help for specific command
            cmd_name = args[0].lower().lstrip("/")
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                lines = [
                    f"📖 Help: /{cmd.name}",
                    f"   {cmd.description}",
                    f"   Usage: {cmd.usage}",
                ]
                if cmd.aliases:
                    lines.append(f"   Aliases: {', '.join('/' + a for a in cmd.aliases)}")
                if cmd.examples:
                    lines.append("   Examples:")
                    for ex in cmd.examples:
                        lines.append(f"      {ex}")
                return "\n".join(lines)
            else:
                return f"Unknown command: /{cmd_name}"

        # List all commands
        by_category = self.get_commands_by_category()

        lines = ["📖 Available Commands:", ""]

        for category in CommandCategory:
            commands = by_category.get(category, [])
            if commands:
                lines.append(f"  {category.value.upper()}")
                for cmd in sorted(commands, key=lambda c: c.name):
                    aliases = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
                    lines.append(f"    /{cmd.name}{aliases} - {cmd.description}")
                lines.append("")

        lines.append("Type /help <command> for detailed help.")
        return "\n".join(lines)

    def _cmd_status(self, args: List[str]) -> str:
        """Handle /status command."""
        lines = ["📊 System Status", ""]

        # Basic status info
        lines.append(f"  Commands registered: {len(self.commands)}")
        lines.append(f"  Handlers loaded: {len(self.handlers)}")

        if self.cognitive_engine:
            lines.append("  Cognitive Engine: Online")
        else:
            lines.append("  Cognitive Engine: Not available")

        return "\n".join(lines)

    def _cmd_exit(self, args: List[str]) -> str:
        """Handle /exit command."""
        return "EXIT_REQUESTED"
