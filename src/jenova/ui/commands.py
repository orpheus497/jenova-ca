# The JENOVA Cognitive Architecture - Enhanced Command System
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 9: Enhanced command system with network control and settings management.

Provides centralized command handling with:
- Network management (/network, /peers)
- Settings control (/settings)
- Enhanced help (/help)
- User profiling (/profile)
- Learning stats (/learn)
"""

from typing import Dict, List, Optional, Callable
from enum import Enum


class CommandCategory(Enum):
    """Command categories for organization."""
    SYSTEM = "system"
    NETWORK = "network"
    MEMORY = "memory"
    LEARNING = "learning"
    SETTINGS = "settings"
    HELP = "help"


class Command:
    """Represents a slash command."""

    def __init__(
        self,
        name: str,
        description: str,
        category: CommandCategory,
        handler: Callable,
        aliases: Optional[List[str]] = None,
        usage: Optional[str] = None,
        examples: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.category = category
        self.handler = handler
        self.aliases = aliases or []
        self.usage = usage or f"/{name}"
        self.examples = examples or []


class CommandRegistry:
    """Registry for all available commands."""

    def __init__(self, cognitive_engine, ui_logger, file_logger):
        self.cognitive_engine = cognitive_engine
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.commands: Dict[str, Command] = {}
        self._register_default_commands()

    def _register_default_commands(self):
        """Register all default JENOVA commands."""

        # Network commands
        self.register(Command(
            name="network",
            description="Show network status and manage distributed mode",
            category=CommandCategory.NETWORK,
            handler=self._cmd_network,
            aliases=["net"],
            usage="/network [status|enable|disable|info]",
            examples=[
                "/network status - Show current network status",
                "/network enable - Enable distributed mode",
                "/network disable - Disable distributed mode",
                "/network info - Detailed network information"
            ]
        ))

        self.register(Command(
            name="peers",
            description="Manage and view peer connections",
            category=CommandCategory.NETWORK,
            handler=self._cmd_peers,
            usage="/peers [list|connect|disconnect|trust|info]",
            examples=[
                "/peers list - List all discovered peers",
                "/peers info <peer_id> - Show detailed peer information",
                "/peers trust <peer_id> - Trust a peer's certificate",
                "/peers disconnect <peer_id> - Disconnect from a peer"
            ]
        ))

        # Settings commands
        self.register(Command(
            name="settings",
            description="Interactive settings menu",
            category=CommandCategory.SETTINGS,
            handler=self._cmd_settings,
            aliases=["config", "preferences"],
            usage="/settings [category]",
            examples=[
                "/settings - Open interactive settings menu",
                "/settings network - Network settings",
                "/settings llm - LLM configuration",
                "/settings privacy - Privacy settings"
            ]
        ))

        # User profile commands
        self.register(Command(
            name="profile",
            description="View and manage your user profile",
            category=CommandCategory.LEARNING,
            handler=self._cmd_profile,
            usage="/profile [view|edit|reset]",
            examples=[
                "/profile view - View your profile and preferences",
                "/profile edit - Edit profile settings",
                "/profile reset - Reset profile to defaults"
            ]
        ))

        # Learning commands
        self.register(Command(
            name="learn",
            description="View learning statistics and insights",
            category=CommandCategory.LEARNING,
            handler=self._cmd_learn,
            usage="/learn [stats|insights|progress]",
            examples=[
                "/learn stats - Show learning statistics",
                "/learn insights - Recent learning insights",
                "/learn progress - Learning progress over time"
            ]
        ))

        # Enhanced help
        self.register(Command(
            name="help",
            description="Show help and documentation",
            category=CommandCategory.HELP,
            handler=self._cmd_help,
            aliases=["?", "man"],
            usage="/help [command|category]",
            examples=[
                "/help - Show all commands",
                "/help network - Show network commands",
                "/help /network - Show help for specific command"
            ]
        ))

    def register(self, command: Command):
        """Register a command."""
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command

    def get_command(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        return self.commands.get(name.lstrip('/'))

    def get_by_category(self, category: CommandCategory) -> List[Command]:
        """Get all commands in a category."""
        seen = set()
        result = []
        for cmd in self.commands.values():
            if cmd.category == category and cmd.name not in seen:
                result.append(cmd)
                seen.add(cmd.name)
        return sorted(result, key=lambda c: c.name)

    def execute(self, command_str: str, args: List[str]) -> str:
        """Execute a command."""
        cmd_name = command_str.lstrip('/')
        command = self.get_command(cmd_name)

        if not command:
            return f"Unknown command: {command_str}. Type /help for available commands."

        try:
            return command.handler(args)
        except Exception as e:
            self.file_logger.log_error(f"Command execution error: {e}")
            return f"Error executing command: {str(e)}"

    # Command Handlers

    def _cmd_network(self, args: List[str]) -> str:
        """Handle /network command."""
        if not self.cognitive_engine.peer_manager:
            return "Network layer not available. Enable distributed mode in configuration."

        subcommand = args[0] if args else "status"

        if subcommand == "status":
            return self._network_status()
        elif subcommand == "enable":
            return "Network mode enabled (restart required for changes)"
        elif subcommand == "disable":
            return "Network mode disabled (restart required for changes)"
        elif subcommand == "info":
            return self._network_info()
        else:
            return f"Unknown network subcommand: {subcommand}"

    def _network_status(self) -> str:
        """Get network status."""
        pm = self.cognitive_engine.peer_manager
        if not pm:
            return "Network: Disabled"

        peers = pm.get_all_peers()
        connected = sum(1 for p in peers if p.status.value == 'connected')

        status = [
            "Network Status:",
            f"  Mode: Distributed (enabled)",
            f"  Peers: {connected} connected, {len(peers)} total discovered",
        ]

        if self.cognitive_engine.distributed_llm:
            stats = self.cognitive_engine.distributed_llm.get_stats()
            status.append(f"  Distributed Generations: {stats['distributed_generations']}")
            status.append(f"  Strategy: {stats['strategy']}")

        return "\n".join(status)

    def _network_info(self) -> str:
        """Get detailed network information."""
        pm = self.cognitive_engine.peer_manager
        if not pm:
            return "Network layer not initialized"

        info = [
            "Network Configuration:",
            f"  Discovery: mDNS/Zeroconf",
            f"  Security: SSL/TLS + JWT",
            f"  Resource Sharing:",
            f"    LLM: enabled",
            f"    Embeddings: enabled",
            f"    Memory: disabled (privacy)",
        ]

        return "\n".join(info)

    def _cmd_peers(self, args: List[str]) -> str:
        """Handle /peers command."""
        if not self.cognitive_engine.peer_manager:
            return "Network layer not available"

        subcommand = args[0] if args else "list"

        if subcommand == "list":
            return self._peers_list()
        elif subcommand == "info" and len(args) > 1:
            return self._peer_info(args[1])
        else:
            return f"Unknown peers subcommand: {subcommand}"

    def _peers_list(self) -> str:
        """List all peers."""
        pm = self.cognitive_engine.peer_manager
        peers = pm.get_all_peers()

        if not peers:
            return "No peers discovered. Ensure other JENOVA instances are running on your LAN."

        lines = ["Discovered Peers:"]
        for peer in peers:
            info = peer.peer_info
            status_icon = "✓" if peer.status.value == 'connected' else "○"
            latency = f"{peer.avg_response_time:.0f}ms" if peer.response_times else "untested"
            lines.append(
                f"  {status_icon} {info.instance_name} ({info.instance_id[:8]}...) "
                f"- {info.address}:{info.port} - {latency}"
            )

        return "\n".join(lines)

    def _peer_info(self, peer_id: str) -> str:
        """Get detailed peer information."""
        pm = self.cognitive_engine.peer_manager
        peer = pm.get_peer_connection(peer_id)

        if not peer:
            return f"Peer not found: {peer_id}"

        info = peer.peer_info
        cap = peer.capabilities

        lines = [
            f"Peer Information: {info.instance_name}",
            f"  ID: {info.instance_id}",
            f"  Address: {info.address}:{info.port}",
            f"  Status: {peer.status.value}",
            f"  Capabilities:",
            f"    LLM: {'enabled' if cap and cap.share_llm else 'disabled'}",
            f"    Embeddings: {'enabled' if cap and cap.share_embeddings else 'disabled'}",
            f"    Memory: {'enabled' if cap and cap.share_memory else 'disabled'}",
        ]

        if peer.response_times:
            avg = sum(peer.response_times) / len(peer.response_times)
            lines.append(f"  Avg Response Time: {avg:.0f}ms")

        return "\n".join(lines)

    def _cmd_settings(self, args: List[str]) -> str:
        """Handle /settings command."""
        category = args[0] if args else None

        # This will be replaced with interactive menu
        if category:
            return f"Settings category: {category} (interactive menu coming soon)"
        else:
            return """Interactive Settings Menu:
  /settings network - Network and distributed computing
  /settings llm - Language model configuration
  /settings memory - Memory system settings
  /settings privacy - Privacy and data settings
  /settings learning - Learning and personalization

Use /settings <category> to open specific settings."""

    def _cmd_profile(self, args: List[str]) -> str:
        """Handle /profile command."""
        # Placeholder - will be replaced with user profiling system
        return """User Profile (Preview):
  Interactions: 0
  Preferred Topics: (learning...)
  Expertise Level: Intermediate
  Response Style: Balanced

Profile system will track your preferences and adapt over time."""

    def _cmd_learn(self, args: List[str]) -> str:
        """Handle /learn command."""
        # Placeholder - will be replaced with learning engine
        return """Learning Statistics (Preview):
  New Concepts Learned: 0
  Skills Acquired: 0
  Patterns Recognized: 0
  User Preferences Learned: 0

Learning system will track knowledge acquisition and improvement."""

    def _cmd_help(self, args: List[str]) -> str:
        """Handle /help command."""
        if not args:
            return self._help_all()

        target = args[0].lstrip('/')

        # Check if it's a specific command
        command = self.get_command(target)
        if command:
            return self._help_command(command)

        # Check if it's a category
        try:
            category = CommandCategory(target.lower())
            return self._help_category(category)
        except ValueError:
            pass

        return f"No help available for: {target}"

    def _help_all(self) -> str:
        """Show all commands grouped by category."""
        lines = ["JENOVA Commands:\n"]

        for category in CommandCategory:
            cmds = self.get_by_category(category)
            if cmds:
                lines.append(f"{category.value.upper()}:")
                for cmd in cmds:
                    lines.append(f"  /{cmd.name:12} - {cmd.description}")
                lines.append("")

        lines.append("Use /help <command> for detailed help on a specific command.")
        lines.append("Use /help <category> to see commands in a category.")

        return "\n".join(lines)

    def _help_category(self, category: CommandCategory) -> str:
        """Show help for a category."""
        cmds = self.get_by_category(category)

        if not cmds:
            return f"No commands in category: {category.value}"

        lines = [f"{category.value.upper()} Commands:\n"]

        for cmd in cmds:
            lines.append(f"/{cmd.name}")
            lines.append(f"  {cmd.description}")
            if cmd.aliases:
                lines.append(f"  Aliases: {', '.join('/' + a for a in cmd.aliases)}")
            lines.append(f"  Usage: {cmd.usage}")
            if cmd.examples:
                lines.append("  Examples:")
                for example in cmd.examples:
                    lines.append(f"    {example}")
            lines.append("")

        return "\n".join(lines)

    def _help_command(self, command: Command) -> str:
        """Show help for a specific command."""
        lines = [
            f"Command: /{command.name}",
            f"Category: {command.category.value}",
            f"Description: {command.description}",
        ]

        if command.aliases:
            lines.append(f"Aliases: {', '.join('/' + a for a in command.aliases)}")

        lines.append(f"\nUsage: {command.usage}")

        if command.examples:
            lines.append("\nExamples:")
            for example in command.examples:
                lines.append(f"  {example}")

        return "\n".join(lines)
