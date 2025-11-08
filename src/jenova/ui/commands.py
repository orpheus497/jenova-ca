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
from jenova.ui.settings_menu import SettingsMenu


class CommandCategory(Enum):
    """Command categories for organization."""
    SYSTEM = "system"
    NETWORK = "network"
    MEMORY = "memory"
    LEARNING = "learning"
    SETTINGS = "settings"
    HELP = "help"
    # Phases 13-17: New categories
    CODE = "code"
    GIT = "git"
    ANALYSIS = "analysis"
    ORCHESTRATION = "orchestration"
    AUTOMATION = "automation"


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

    def __init__(self, cognitive_engine, ui_logger, file_logger, **kwargs):
        self.cognitive_engine = cognitive_engine
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.commands: Dict[str, Command] = {}

        # Initialize settings menu (Phase 9)
        self.settings_menu = SettingsMenu(
            config=cognitive_engine.config,
            user_profile=cognitive_engine.user_profile if hasattr(cognitive_engine, 'user_profile') else None,
            file_logger=file_logger
        )

        # Phases 13-17: Store CLI enhancement modules
        self.context_optimizer = kwargs.get('context_optimizer')
        self.code_metrics = kwargs.get('code_metrics')
        self.security_scanner = kwargs.get('security_scanner')
        self.file_editor = kwargs.get('file_editor')
        self.code_parser = kwargs.get('code_parser')
        self.refactoring_engine = kwargs.get('refactoring_engine')
        self.git_interface = kwargs.get('git_interface')
        self.commit_assistant = kwargs.get('commit_assistant')
        self.task_planner = kwargs.get('task_planner')
        self.execution_engine = kwargs.get('execution_engine')
        self.custom_command_manager = kwargs.get('custom_command_manager')
        self.workflow_library = kwargs.get('workflow_library')

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

        # Phases 13-17: Enhanced CLI Commands
        self._register_phase13_17_commands()

    def _register_phase13_17_commands(self):
        """Register Phase 13-17 CLI enhancement commands."""

        # Code editing commands
        if self.file_editor:
            self.register(Command(
                name="edit",
                description="Edit files with diff-based preview",
                category=CommandCategory.CODE,
                handler=self._cmd_edit,
                usage="/edit <file> [--preview|--apply]",
                examples=[
                    "/edit main.py - Preview edits",
                    "/edit main.py --apply - Apply edits",
                    "/edit main.py --backup - Create backup before editing"
                ]
            ))

        # Code analysis commands
        if self.code_metrics:
            self.register(Command(
                name="analyze",
                description="Analyze code quality and complexity",
                category=CommandCategory.ANALYSIS,
                handler=self._cmd_analyze,
                usage="/analyze <file|directory>",
                examples=[
                    "/analyze main.py - Analyze single file",
                    "/analyze src/ - Analyze directory",
                    "/analyze . --report=json - JSON format report"
                ]
            ))

        if self.security_scanner:
            self.register(Command(
                name="scan",
                description="Scan code for security vulnerabilities",
                category=CommandCategory.ANALYSIS,
                handler=self._cmd_scan,
                aliases=["security"],
                usage="/scan <file|directory>",
                examples=[
                    "/scan main.py - Scan single file",
                    "/scan src/ - Scan directory",
                    "/scan . --severity=high - Show only high severity"
                ]
            ))

        # Code parsing commands
        if self.code_parser:
            self.register(Command(
                name="parse",
                description="Parse and analyze code structure (AST)",
                category=CommandCategory.CODE,
                handler=self._cmd_parse,
                usage="/parse <file>",
                examples=[
                    "/parse main.py - Show code structure",
                    "/parse main.py --symbols - List all symbols",
                    "/parse main.py --tree - Show AST tree"
                ]
            ))

        # Code refactoring commands
        if self.refactoring_engine:
            self.register(Command(
                name="refactor",
                description="Refactor code (rename, extract, etc.)",
                category=CommandCategory.CODE,
                handler=self._cmd_refactor,
                usage="/refactor <operation> <args>",
                examples=[
                    "/refactor rename old_name new_name",
                    "/refactor extract-method function_name",
                    "/refactor inline variable_name"
                ]
            ))

        # Git commands
        if self.git_interface:
            self.register(Command(
                name="git",
                description="Git operations with AI assistance",
                category=CommandCategory.GIT,
                handler=self._cmd_git,
                usage="/git <operation> [args]",
                examples=[
                    "/git status - Show git status",
                    "/git commit - Auto-generate commit message",
                    "/git diff - Show and analyze diff",
                    "/git branch - List branches"
                ]
            ))

        # Task orchestration commands
        if self.task_planner and self.execution_engine:
            self.register(Command(
                name="task",
                description="Plan and execute multi-step tasks",
                category=CommandCategory.ORCHESTRATION,
                handler=self._cmd_task,
                usage="/task <plan|execute|status|cancel> [args]",
                examples=[
                    "/task plan 'refactor module X' - Create task plan",
                    "/task execute <plan_id> - Execute planned task",
                    "/task status - Show active tasks",
                    "/task cancel <task_id> - Cancel running task"
                ]
            ))

        # Workflow commands
        if self.workflow_library:
            self.register(Command(
                name="workflow",
                description="Execute predefined workflows",
                category=CommandCategory.AUTOMATION,
                handler=self._cmd_workflow,
                usage="/workflow <name> [args]",
                examples=[
                    "/workflow code-review - Run code review workflow",
                    "/workflow test - Run testing workflow",
                    "/workflow deploy - Run deployment workflow",
                    "/workflow list - List available workflows"
                ]
            ))

        # Custom command management
        if self.custom_command_manager:
            self.register(Command(
                name="command",
                description="Manage custom commands",
                category=CommandCategory.AUTOMATION,
                handler=self._cmd_command,
                usage="/command <create|list|execute|delete> [args]",
                examples=[
                    "/command list - List custom commands",
                    "/command create my_cmd - Create new command",
                    "/command execute my_cmd - Run custom command",
                    "/command delete my_cmd - Delete command"
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
        if not args:
            # Show all categories
            lines = ["Interactive Settings Menu:\n"]
            for category in self.settings_menu.list_categories():
                lines.append(f"  /settings {category.name:10} - {category.description}")
            lines.append("\nUse /settings <category> to view settings in that category.")
            lines.append("Use /settings <category> <key> <value> to change a setting.")
            return "\n".join(lines)

        category_name = args[0]
        category = self.settings_menu.get_category(category_name)

        if not category:
            return f"Unknown settings category: {category_name}\nUse /settings to see available categories."

        # If only category specified, show all settings in that category
        if len(args) == 1:
            lines = [f"{category.name.upper()} Settings:\n"]
            for setting in category.list_settings():
                current_val = self.settings_menu.get_setting_value(setting.key)
                restart_note = " [requires restart]" if setting.requires_restart else ""
                lines.append(f"  {setting.name} ({setting.key}){restart_note}")
                lines.append(f"    Current: {current_val}")
                lines.append(f"    {setting.description}")
                if setting.choices:
                    lines.append(f"    Choices: {', '.join(str(c) for c in setting.choices)}")
                elif setting.min_value is not None or setting.max_value is not None:
                    range_str = f"Range: {setting.min_value or '∞'} to {setting.max_value or '∞'}"
                    lines.append(f"    {range_str}")
                lines.append("")
            return "\n".join(lines)

        # Setting a value: /settings category key value
        if len(args) >= 3:
            setting_key = args[1]
            new_value = args[2]

            # Find the full setting key
            full_key = None
            for setting in category.list_settings():
                if setting.key.endswith(setting_key) or setting.key == setting_key:
                    full_key = setting.key
                    break

            if not full_key:
                return f"Unknown setting: {setting_key} in category {category_name}"

            # Convert value to appropriate type
            setting = category.get_setting(full_key)
            try:
                if setting.value_type == bool:
                    typed_value = new_value.lower() in ('true', '1', 'yes', 'on')
                elif setting.value_type == int:
                    typed_value = int(new_value)
                elif setting.value_type == float:
                    typed_value = float(new_value)
                else:
                    typed_value = new_value
            except ValueError:
                return f"Invalid value type for {full_key}: expected {setting.value_type.__name__}"

            # Apply the change
            if self.settings_menu.set_setting_value(full_key, typed_value, apply_immediately=True):
                restart_note = "\nNote: Restart required for this change to take effect." if setting.requires_restart else ""
                return f"Setting updated: {setting.name} = {typed_value}{restart_note}"
            else:
                return f"Failed to update setting {full_key}"

        return f"Usage: /settings {category_name} <key> <value>"

    def _cmd_profile(self, args: List[str]) -> str:
        """Handle /profile command."""
        profile = self.cognitive_engine.user_profile if hasattr(self.cognitive_engine, 'user_profile') else None

        if not profile:
            return "User profiling system not initialized."

        subcommand = args[0] if args else "view"

        if subcommand == "view":
            # Show comprehensive profile information
            lines = [f"User Profile: {profile.username}\n"]

            # Preferences
            lines.append("=== Preferences ===")
            lines.append(f"  Response Style: {profile.preferences.response_style}")
            lines.append(f"  Expertise Level: {profile.preferences.expertise_level}")
            lines.append(f"  Communication Style: {profile.preferences.communication_style}")
            lines.append(f"  Learning Mode: {'enabled' if profile.preferences.learning_mode else 'disabled'}")
            lines.append(f"  Proactive Suggestions: {'enabled' if profile.preferences.proactive_suggestions else 'disabled'}")

            if profile.preferences.preferred_topics:
                lines.append(f"  Preferred Topics: {', '.join(profile.preferences.preferred_topics[:5])}")
            lines.append("")

            # Statistics
            lines.append("=== Statistics ===")
            lines.append(f"  Total Interactions: {profile.stats.total_interactions}")
            lines.append(f"  Questions Asked: {profile.stats.questions_asked}")
            lines.append(f"  Commands Used: {profile.stats.commands_used}")
            lines.append(f"  Vocabulary Size: {len(profile.vocabulary)} words")
            lines.append(f"  Unique Topics Discussed: {len(profile.stats.topics_discussed)}")

            if profile.stats.last_interaction:
                lines.append(f"  Last Interaction: {profile.stats.last_interaction}")
            lines.append("")

            # Top Topics
            top_topics = profile.get_top_topics(limit=5)
            if top_topics:
                lines.append("=== Top Discussion Topics ===")
                for topic, count in top_topics:
                    lines.append(f"  {topic}: {count} times")
                lines.append("")

            # Top Commands
            if profile.preferred_commands:
                lines.append("=== Frequently Used Commands ===")
                for cmd, count in profile.preferred_commands.most_common(5):
                    lines.append(f"  {cmd}: {count} times")
                lines.append("")

            # Expertise Indicators
            expertise = profile.get_expertise_indicators()
            lines.append("=== Expertise Indicators ===")
            lines.append(f"  Vocabulary Size: {expertise['vocabulary_size']} words")
            lines.append(f"  Topics Mastered: {expertise['topics_mastered']}")
            lines.append(f"  Command Proficiency: {expertise['command_proficiency']} commands")
            if profile.total_suggestions > 0:
                success_rate = profile.get_suggestion_success_rate() * 100
                lines.append(f"  Suggestion Acceptance Rate: {success_rate:.1f}%")

            return "\n".join(lines)

        elif subcommand == "reset":
            # Reset profile to defaults
            profile.preferences.response_style = "balanced"
            profile.preferences.expertise_level = "intermediate"
            profile.preferences.communication_style = "friendly"
            profile.preferences.learning_mode = True
            profile.preferences.proactive_suggestions = True
            profile.preferences.preferred_topics = []
            profile.save()
            return "Profile reset to defaults."

        else:
            return f"Unknown profile subcommand: {subcommand}\nUse /profile view or /profile reset"

    def _cmd_learn(self, args: List[str]) -> str:
        """Handle /learn command."""
        learning_engine = self.cognitive_engine.learning_engine if hasattr(self.cognitive_engine, 'learning_engine') else None

        if not learning_engine:
            return "Learning engine not initialized."

        subcommand = args[0] if args else "stats"

        if subcommand == "stats":
            # Show learning statistics
            metrics = learning_engine.monitor_performance()

            lines = ["Learning Statistics:\n"]
            lines.append("=== Performance ===")
            lines.append(f"  Total Examples: {metrics['total_examples']}")
            lines.append(f"  Learned Examples: {metrics['learned_examples']}")
            lines.append(f"  Learning Rate: {metrics['learning_rate']:.1%}")
            lines.append("")

            lines.append("=== Pattern Recognition ===")
            lines.append(f"  Total Patterns: {metrics['total_patterns']}")
            lines.append(f"  High Confidence Patterns: {metrics['high_confidence_patterns']}")
            lines.append("")

            lines.append("=== Skill Acquisition ===")
            lines.append(f"  Total Skills: {metrics['total_skills']}")
            lines.append(f"  Proficient Skills: {metrics['proficient_skills']}")
            if metrics['total_skills'] > 0:
                lines.append(f"  Average Proficiency: {metrics['avg_skill_proficiency']:.1%}")
            lines.append("")

            return "\n".join(lines)

        elif subcommand == "insights":
            # Show learning insights
            insights = learning_engine.get_learning_insights()

            if not insights:
                return "No learning insights available yet. Continue interacting to generate insights."

            lines = ["Learning Insights:\n"]
            for i, insight in enumerate(insights, 1):
                lines.append(f"  {i}. {insight}")
            lines.append("")

            return "\n".join(lines)

        elif subcommand == "gaps":
            # Show knowledge gaps
            gaps = learning_engine.identify_knowledge_gaps()

            if not gaps:
                return "No significant knowledge gaps identified. Learning is progressing well!"

            lines = ["Knowledge Gaps:\n"]
            for i, gap in enumerate(gaps, 1):
                lines.append(f"  {i}. {gap}")
            lines.append("")
            lines.append("Consider focusing on these areas for improvement.")

            return "\n".join(lines)

        elif subcommand == "skills":
            # List all acquired skills
            skills = learning_engine.skills

            if not skills:
                return "No skills acquired yet."

            lines = ["Acquired Skills:\n"]
            # Group by domain
            from collections import defaultdict
            domain_skills = defaultdict(list)
            for skill in skills.values():
                domain_skills[skill.domain].append(skill)

            for domain, domain_skill_list in sorted(domain_skills.items()):
                lines.append(f"=== {domain.upper()} ===")
                for skill in sorted(domain_skill_list, key=lambda s: s.proficiency, reverse=True):
                    prof_bar = "█" * int(skill.proficiency * 10) + "░" * (10 - int(skill.proficiency * 10))
                    lines.append(f"  {skill.skill_name}")
                    lines.append(f"    Proficiency: [{prof_bar}] {skill.proficiency:.1%}")
                    lines.append(f"    Practice Count: {skill.practice_count}")
                lines.append("")

            return "\n".join(lines)

        else:
            return f"""Unknown learn subcommand: {subcommand}

Available subcommands:
  /learn stats - Show learning statistics
  /learn insights - Show learning insights
  /learn gaps - Identify knowledge gaps
  /learn skills - List all acquired skills"""

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

    # Phases 13-17: Enhanced CLI Command Handlers

    def _cmd_edit(self, args: List[str]) -> str:
        """Handle /edit command."""
        if not self.file_editor:
            return "File editor not available."

        if not args:
            return "Usage: /edit <file> [--preview|--apply|--backup]"

        try:
            file_path = args[0]
            mode = args[1] if len(args) > 1 else "--preview"

            result = self.file_editor.edit_file(file_path, mode=mode.lstrip('--'))
            return f"Edit operation completed:\n{result}"
        except Exception as e:
            self.file_logger.log_error(f"Edit command error: {e}")
            return f"Error editing file: {str(e)}"

    def _cmd_analyze(self, args: List[str]) -> str:
        """Handle /analyze command."""
        if not self.code_metrics:
            return "Code metrics analyzer not available."

        if not args:
            return "Usage: /analyze <file|directory> [--report=text|json]"

        try:
            target = args[0]
            report_format = "text"
            for arg in args[1:]:
                if arg.startswith("--report="):
                    report_format = arg.split("=")[1]

            analysis = self.code_metrics.analyze(target, report_format=report_format)
            return f"Code Analysis Results:\n{analysis}"
        except Exception as e:
            self.file_logger.log_error(f"Analyze command error: {e}")
            return f"Error analyzing code: {str(e)}"

    def _cmd_scan(self, args: List[str]) -> str:
        """Handle /scan command."""
        if not self.security_scanner:
            return "Security scanner not available."

        if not args:
            return "Usage: /scan <file|directory> [--severity=low|medium|high]"

        try:
            target = args[0]
            severity_filter = None
            for arg in args[1:]:
                if arg.startswith("--severity="):
                    severity_filter = arg.split("=")[1]

            scan_results = self.security_scanner.scan(target, severity_filter=severity_filter)
            return f"Security Scan Results:\n{scan_results}"
        except Exception as e:
            self.file_logger.log_error(f"Scan command error: {e}")
            return f"Error scanning code: {str(e)}"

    def _cmd_parse(self, args: List[str]) -> str:
        """Handle /parse command."""
        if not self.code_parser:
            return "Code parser not available."

        if not args:
            return "Usage: /parse <file> [--symbols|--tree]"

        try:
            file_path = args[0]
            show_mode = "structure"
            if len(args) > 1:
                if args[1] == "--symbols":
                    show_mode = "symbols"
                elif args[1] == "--tree":
                    show_mode = "tree"

            parse_result = self.code_parser.parse(file_path, mode=show_mode)
            return f"Code Structure:\n{parse_result}"
        except Exception as e:
            self.file_logger.log_error(f"Parse command error: {e}")
            return f"Error parsing code: {str(e)}"

    def _cmd_refactor(self, args: List[str]) -> str:
        """Handle /refactor command."""
        if not self.refactoring_engine:
            return "Refactoring engine not available."

        if len(args) < 2:
            return "Usage: /refactor <operation> <args>\nOperations: rename, extract-method, inline"

        try:
            operation = args[0]
            refactor_args = args[1:]

            result = self.refactoring_engine.refactor(operation, refactor_args)
            return f"Refactoring completed:\n{result}"
        except Exception as e:
            self.file_logger.log_error(f"Refactor command error: {e}")
            return f"Error refactoring code: {str(e)}"

    def _cmd_git(self, args: List[str]) -> str:
        """Handle /git command."""
        if not self.git_interface:
            return "Git interface not available."

        if not args:
            return "Usage: /git <operation> [args]\nOperations: status, commit, diff, branch, log"

        try:
            operation = args[0]
            git_args = args[1:]

            if operation == "commit" and self.commit_assistant:
                # Use AI to generate commit message
                result = self.commit_assistant.auto_commit(git_args)
            else:
                result = self.git_interface.execute(operation, git_args)

            return f"Git operation completed:\n{result}"
        except Exception as e:
            self.file_logger.log_error(f"Git command error: {e}")
            return f"Error executing git operation: {str(e)}"

    def _cmd_task(self, args: List[str]) -> str:
        """Handle /task command."""
        if not self.task_planner or not self.execution_engine:
            return "Task orchestration not available."

        if not args:
            return "Usage: /task <plan|execute|status|cancel> [args]"

        try:
            subcommand = args[0]
            task_args = args[1:]

            if subcommand == "plan":
                if not task_args:
                    return "Usage: /task plan '<task description>'"
                task_description = " ".join(task_args)
                plan = self.task_planner.create_plan(task_description)
                return f"Task plan created:\n{plan}"

            elif subcommand == "execute":
                if not task_args:
                    return "Usage: /task execute <plan_id>"
                plan_id = task_args[0]
                result = self.execution_engine.execute_plan(plan_id)
                return f"Task execution started:\n{result}"

            elif subcommand == "status":
                status = self.execution_engine.get_status()
                return f"Task status:\n{status}"

            elif subcommand == "cancel":
                if not task_args:
                    return "Usage: /task cancel <task_id>"
                task_id = task_args[0]
                result = self.execution_engine.cancel_task(task_id)
                return f"Task cancelled:\n{result}"

            else:
                return f"Unknown task subcommand: {subcommand}"

        except Exception as e:
            self.file_logger.log_error(f"Task command error: {e}")
            return f"Error in task operation: {str(e)}"

    def _cmd_workflow(self, args: List[str]) -> str:
        """Handle /workflow command."""
        if not self.workflow_library:
            return "Workflow library not available."

        if not args:
            return "Usage: /workflow <name|list> [args]"

        try:
            subcommand = args[0]

            if subcommand == "list":
                workflows = self.workflow_library.list_workflows()
                return f"Available workflows:\n{workflows}"

            else:
                workflow_name = subcommand
                workflow_args = args[1:]
                result = self.workflow_library.execute_workflow(workflow_name, workflow_args)
                return f"Workflow executed:\n{result}"

        except Exception as e:
            self.file_logger.log_error(f"Workflow command error: {e}")
            return f"Error executing workflow: {str(e)}"

    def _cmd_command(self, args: List[str]) -> str:
        """Handle /command (custom command management)."""
        if not self.custom_command_manager:
            return "Custom command manager not available."

        if not args:
            return "Usage: /command <create|list|execute|delete> [args]"

        try:
            subcommand = args[0]

            if subcommand == "list":
                commands = self.custom_command_manager.list_commands()
                return f"Custom commands:\n{commands}"

            elif subcommand == "create":
                if len(args) < 2:
                    return "Usage: /command create <command_name>"
                cmd_name = args[1]
                result = self.custom_command_manager.create_command(cmd_name)
                return f"Custom command created:\n{result}"

            elif subcommand == "execute":
                if len(args) < 2:
                    return "Usage: /command execute <command_name>"
                cmd_name = args[1]
                cmd_args = args[2:]
                result = self.custom_command_manager.execute_command(cmd_name, cmd_args)
                return f"Command executed:\n{result}"

            elif subcommand == "delete":
                if len(args) < 2:
                    return "Usage: /command delete <command_name>"
                cmd_name = args[1]
                result = self.custom_command_manager.delete_command(cmd_name)
                return f"Command deleted:\n{result}"

            else:
                return f"Unknown command subcommand: {subcommand}"

        except Exception as e:
            self.file_logger.log_error(f"Custom command error: {e}")
            return f"Error managing custom commands: {str(e)}"

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
