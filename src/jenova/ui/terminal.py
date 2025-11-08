# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for the terminal UI of the JENOVA Cognitive Architecture."""

import getpass
import itertools
import os

import sys
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.ui.logger import UILogger
from jenova.ui.health_display import HealthDisplay, CompactHealthDisplay
from jenova.ui.commands import CommandRegistry

BANNER = """
     ██╗███████╗███╗   ██╗ ██████╗ ██╗   ██╗ █████╗
     ██║██╔════╝████╗  ██║██╔═══██╗██║   ██║██╔══██╗
     ██║█████╗  ██╔██╗ ██║██║   ██║██║   ██║███████║
██   ██║██╔══╝  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
╚█████╔╝███████╗██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
 ╚════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝"""
ATTRIBUTION = "Designed and Developed by orpheus497 - https://github.com/orpheus497"


class TerminalUI:
    def __init__(
        self,
        cognitive_engine: CognitiveEngine,
        logger: UILogger,
        health_monitor=None,
        metrics=None,
        **kwargs,
    ):
        """
        Initialize TerminalUI.

        Args:
            cognitive_engine: The cognitive engine instance
            logger: UI logger instance
            health_monitor: Optional health monitor (Phase 2)
            metrics: Optional metrics collector (Phase 2)
            **kwargs: Optional CLI enhancement modules (Phases 13-17, 19)
                - Backup: backup_manager (Phase 19)
                - Analysis: context_optimizer, code_metrics, security_scanner, intent_classifier, command_disambiguator
                - Code Tools: file_editor, code_parser, refactoring_engine, syntax_highlighter, codebase_mapper, interactive_terminal
                - Git Tools: git_interface, commit_assistant, diff_analyzer, hooks_manager, branch_manager
                - Orchestration: task_planner, subagent_manager, execution_engine, checkpoint_manager, background_task_manager
                - Automation: custom_command_manager, hooks_system, template_engine, workflow_library
        """
        self.engine = cognitive_engine
        self.logger = logger
        self.health_monitor = health_monitor
        self.metrics = metrics
        self.username = getpass.getuser()
        history_path = os.path.join(
            self.engine.config["user_data_root"], ".jenova_history"
        )
        self.session = PromptSession(
            history=FileHistory(history_path), auto_suggest=AutoSuggestFromHistory()
        )
        self.prompt_style = Style.from_dict(
            {
                "username": "#44ff44 bold",
                "at": "#888888",
                "hostname": "#ff00ff bold",
                "prompt": "#888888",
            }
        )
        self.verifying_assumption = None
        self._spinner_running = False
        self._spinner_thread = None
        self.message_queue = self.logger.message_queue

        # Phase 19: Store backup manager
        self.backup_manager = kwargs.get("backup_manager")

        # Phases 13-17: Store CLI enhancement modules for command handlers
        # Analysis Module
        self.context_optimizer = kwargs.get("context_optimizer")
        self.code_metrics = kwargs.get("code_metrics")
        self.security_scanner = kwargs.get("security_scanner")
        self.intent_classifier = kwargs.get("intent_classifier")
        self.command_disambiguator = kwargs.get("command_disambiguator")
        # Code Tools Module
        self.file_editor = kwargs.get("file_editor")
        self.code_parser = kwargs.get("code_parser")
        self.refactoring_engine = kwargs.get("refactoring_engine")
        self.syntax_highlighter = kwargs.get("syntax_highlighter")
        self.codebase_mapper = kwargs.get("codebase_mapper")
        self.interactive_terminal = kwargs.get("interactive_terminal")
        # Git Tools Module
        self.git_interface = kwargs.get("git_interface")
        self.commit_assistant = kwargs.get("commit_assistant")
        self.diff_analyzer = kwargs.get("diff_analyzer")
        self.hooks_manager = kwargs.get("hooks_manager")
        self.branch_manager = kwargs.get("branch_manager")
        # Orchestration Module
        self.task_planner = kwargs.get("task_planner")
        self.subagent_manager = kwargs.get("subagent_manager")
        self.execution_engine = kwargs.get("execution_engine")
        self.checkpoint_manager = kwargs.get("checkpoint_manager")
        self.background_task_manager = kwargs.get("background_task_manager")
        # Automation Module
        self.custom_command_manager = kwargs.get("custom_command_manager")
        self.hooks_system = kwargs.get("hooks_system")
        self.template_engine = kwargs.get("template_engine")
        self.workflow_library = kwargs.get("workflow_library")

        # Phase 9: Integrated Command Registry (pass CLI enhancements)
        # Phase 19: Also pass backup_manager
        self.command_registry = CommandRegistry(
            cognitive_engine,
            logger,
            cognitive_engine.file_logger,
            # Phase 19: Backup capabilities
            backup_manager=self.backup_manager,
            # Pass CLI enhancements to command registry
            context_optimizer=self.context_optimizer,
            code_metrics=self.code_metrics,
            security_scanner=self.security_scanner,
            file_editor=self.file_editor,
            code_parser=self.code_parser,
            refactoring_engine=self.refactoring_engine,
            git_interface=self.git_interface,
            commit_assistant=self.commit_assistant,
            task_planner=self.task_planner,
            execution_engine=self.execution_engine,
            custom_command_manager=self.custom_command_manager,
            workflow_library=self.workflow_library,
        )
        self.commands = self._register_commands()

        # Phase 6: Health Display
        self.health_display = (
            HealthDisplay(health_monitor, metrics, self.logger.console)
            if health_monitor
            else None
        )
        self.compact_health = (
            CompactHealthDisplay(health_monitor) if health_monitor else None
        )

    def _register_commands(self):
        return {
            "/insight": self._run_command_in_thread(
                self.engine.develop_insights_from_conversation, needs_spinner=True
            ),
            "/reflect": self._run_command_in_thread(
                self.engine.reflect_on_insights, needs_spinner=True
            ),
            "/memory-insight": self._run_command_in_thread(
                self.engine.develop_insights_from_memory, needs_spinner=True
            ),
            "/meta": self._run_command_in_thread(
                self.engine.generate_meta_insight, needs_spinner=True
            ),
            "/verify": self._verify_assumption,
            "/train": self._show_train_help,
            "/develop_insight": self._develop_insight,
            "/learn_procedure": self._learn_procedure,
            "/help": self._show_help,
            # Phase 6: Health and Metrics Commands
            "/health": self._show_health,
            "/metrics": self._show_metrics,
            "/status": self._show_status,
            "/cache": self._show_cache_stats,
        }

    def _spinner(self):
        spinner_chars = itertools.cycle(["   ", ".  ", ".. ", "..."])
        color_code = "\033[93m"  # Yellow color
        reset_code = "\033[0m"
        while self._spinner_running:
            sys.stdout.write(f"{color_code}\r{next(spinner_chars)}{reset_code}")
            sys.stdout.flush()
            time.sleep(0.2)
        sys.stdout.write("\r" + " " * 5 + "\r")
        sys.stdout.flush()

    def start_spinner(self):
        """
        Start the loading spinner in a separate thread.

        Creates and starts a daemon thread that displays an animated spinner
        to indicate that processing is in progress.

        Example:
            >>> terminal = TerminalUI(engine, logger, username)
            >>> terminal.start_spinner()
            >>> # Long-running operation here
            >>> terminal.stop_spinner()
        """
        self._spinner_running = True
        self._spinner_thread = threading.Thread(target=self._spinner)
        self._spinner_thread.daemon = True
        self._spinner_thread.start()

    def stop_spinner(self):
        """
        Stop the loading spinner and clean up the spinner thread.

        Signals the spinner thread to stop, waits for it to finish, and
        cleans up the thread reference.

        Example:
            >>> terminal = TerminalUI(engine, logger, username)
            >>> terminal.start_spinner()
            >>> # Long-running operation here
            >>> terminal.stop_spinner()
        """
        self._spinner_running = False
        if self._spinner_thread and self._spinner_thread.is_alive():
            self._spinner_thread.join()
        self._spinner_thread = None

    def run(self):
        """
        Start the main terminal interaction loop.

        This is the main entry point for the terminal UI. It displays the banner,
        enters the main input loop, and handles user input until the user exits.

        The loop handles:
            - User text input and command execution
            - Assumption verification prompts
            - Special commands (starting with /)
            - Keyboard interrupts (Ctrl+C)
            - End-of-file (Ctrl+D)

        Example:
            >>> terminal = TerminalUI(engine, logger, username)
            >>> terminal.run()  # Blocks until user exits
        """
        if self.logger:
            self.logger.banner(BANNER, ATTRIBUTION)
            self.logger.info("Initialized and Ready.")
            self.logger.info(
                "Type your message, use a command, or type 'exit' to quit."
            )
            self.logger.info("Type /help to see a list of available commands.\n")
            self.logger.process_queued_messages()

        while True:
            try:
                if self.verifying_assumption:
                    prompt_message = [("class:prompt", "Your answer (yes/no): ")]
                    user_input = self.session.prompt(
                        prompt_message, style=self.prompt_style
                    ).strip()
                    if self.engine:
                        self.engine.assumption_manager.resolve_assumption(
                            self.verifying_assumption, user_input, self.username
                        )
                    self.verifying_assumption = None
                    if self.logger:
                        self.logger.system_message("")
                        self.logger.process_queued_messages()
                    continue

                prompt_message = [
                    ("class:username", self.username),
                    ("class:at", "@"),
                    ("class:hostname", "JENOVA"),
                    ("class:prompt", "> "),
                ]
                user_input = self.session.prompt(
                    prompt_message, style=self.prompt_style
                ).strip()

                if not user_input:
                    if self.logger:
                        self.logger.system_message("")
                        self.logger.process_queued_messages()
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    break

                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self._handle_conversation(user_input)

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                if self.logger:
                    self.logger.system_message(
                        f"An unexpected error occurred in the UI loop: {e}"
                    )
                    self.logger.process_queued_messages()

    def _handle_command(self, user_input: str):
        # Parse command and arguments
        parts = user_input.split(" ", 1)
        command_name = parts[0]
        args = parts[1].split() if len(parts) > 1 else []

        # Try Phase 9 command registry first
        result = self.command_registry.execute(command_name, args)
        if result and not result.startswith("Unknown command"):
            if self.logger:
                self.logger.system_message(result)
                self.logger.system_message("")
                self.logger.process_queued_messages()
            return

        # Fallback to legacy commands for backward compatibility
        command_func = self.commands.get(command_name.lower())
        if command_func:
            command_func(args)
        else:
            if self.logger:
                self.logger.system_message(
                    f"Unknown command: {command_name}. Type /help for available commands."
                )

        if self.logger:
            self.logger.system_message("")
            self.logger.process_queued_messages()

    def _handle_conversation(self, user_input: str):
        self.start_spinner()
        response_container = []

        def task():
            response_container.append(self.engine.think(user_input, self.username))

        thread = threading.Thread(target=task)
        thread.start()

        while thread.is_alive():
            if self.logger:
                self.logger.process_queued_messages()
            time.sleep(0.1)
        thread.join()

        self.stop_spinner()

        response = (
            response_container[0]
            if response_container
            else "Error: No response generated"
        )

        if self.logger:
            self.logger.process_queued_messages()

        if not isinstance(response, str):
            response = str(response)
        if self.logger:
            self.logger.system_message("")
            self.logger.jenova_response(response)
            self.logger.process_queued_messages()
            self.logger.system_message("")

    def _process_and_log_messages(self, returned_messages):
        if not returned_messages:
            return
        messages = (
            returned_messages
            if isinstance(returned_messages, list)
            else [returned_messages]
        )
        for msg in messages:
            if self.logger:
                self.logger.system_message(str(msg))

    def _run_command_in_thread(self, target_func, needs_spinner=False):
        def wrapper(args):
            if needs_spinner:
                self.start_spinner()

            result_container = []
            # Pass username for functions that need it
            func_args = (self.username,) + tuple(args)

            def task():
                result_container.append(target_func(*func_args))

            thread = threading.Thread(target=task)
            thread.start()

            while thread.is_alive():
                if self.logger:
                    self.logger.process_queued_messages()
                time.sleep(0.1)
            thread.join()

            if needs_spinner:
                self.stop_spinner()

            if self.logger:
                self.logger.process_queued_messages()
            returned_messages = result_container[0] if result_container else []
            self._process_and_log_messages(returned_messages)

        return wrapper

    def _verify_assumption(self, args):
        self.start_spinner()
        result_container = []

        def task():
            result_container.append(self.engine.verify_assumptions(self.username))

        thread = threading.Thread(target=task)
        thread.start()

        while thread.is_alive():
            if self.logger:
                self.logger.process_queued_messages()
            time.sleep(0.1)
        thread.join()

        self.stop_spinner()
        if self.logger:
            self.logger.process_queued_messages()

        assumption, question = result_container[0] if result_container else (None, None)

        if question:
            if self.logger:
                self.logger.system_message(
                    f"Jenova is asking for clarification: {question}"
                )
                self.logger.process_queued_messages()
            if assumption:
                self.verifying_assumption = assumption

    def _develop_insight(self, args):
        target_func = (
            self.engine.cortex.develop_insight
            if args
            else self.engine.cortex.develop_insights_from_docs
        )
        func_args = (args[0], self.username) if args else (self.username,)
        self._run_command_in_thread(target_func, needs_spinner=True)(func_args)

    def _learn_procedure(self, args):
        if self.logger:
            self.logger.system_message("Initiating interactive procedure learning...")

        procedure_name = self.session.prompt(
            [("class:prompt", "Procedure Name: ")]
        ).strip()
        if not procedure_name:
            if self.logger:
                self.logger.system_message("Procedure name cannot be empty. Aborting.")
            return

        steps = []
        if self.logger:
            self.logger.system_message(
                "Enter procedure steps one by one. Type 'done' when finished."
            )
        while True:
            step = self.session.prompt(
                [("class:prompt", f"Step {len(steps) + 1}: ")]
            ).strip()
            if step.lower() == "done":
                break
            if step:
                steps.append(step)

        if not steps:
            if self.logger:
                self.logger.system_message("No steps entered. Aborting.")
            return

        expected_outcome = self.session.prompt(
            [("class:prompt", "Expected Outcome: ")]
        ).strip()
        if not expected_outcome:
            if self.logger:
                self.logger.system_message(
                    "Expected outcome cannot be empty. Aborting."
                )
            return

        procedure_data = {
            "name": procedure_name,
            "steps": steps,
            "outcome": expected_outcome,
        }
        self._run_command_in_thread(self.engine.learn_procedure, needs_spinner=True)(
            (procedure_data, self.username)
        )

    # Phase 6: Health and Metrics Command Implementations

    def _show_health(self, args):
        """Display current system health status."""
        if not self.health_display:
            if self.logger:
                self.logger.warning(
                    "Health monitoring not available (disabled or not initialized)"
                )
            return

        try:
            self.health_display.show_health()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error displaying health: {e}")

    def _show_metrics(self, args):
        """Display performance metrics."""
        if not self.metrics:
            if self.logger:
                self.logger.warning(
                    "Metrics collection not available (disabled or not initialized)"
                )
            return

        try:
            all_stats = self.metrics.get_all_stats()
            if not all_stats:
                if self.logger:
                    self.logger.system_message("No metrics data available yet.")
                return

            # Convert stats to display format
            metrics_data = {
                operation: {
                    "count": stats.count,
                    "avg_time": stats.avg_time,
                    "total_time": stats.total_time,
                }
                for operation, stats in all_stats.items()
            }

            if self.logger:
                self.logger.metrics_table(metrics_data)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error displaying metrics: {e}")

    def _show_status(self, args):
        """Display complete system status (health + metrics + cognitive)."""
        if not self.health_display:
            if self.logger:
                self.logger.warning(
                    "Status display not available (health monitor not initialized)"
                )
            return

        try:
            self.health_display.show_full_status()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error displaying status: {e}")

    def _show_cache_stats(self, args):
        """Display RAG cache statistics."""
        try:
            # Try to get cache stats from RAG system
            if hasattr(self.engine, "rag_system") and hasattr(
                self.engine.rag_system, "get_cache_stats"
            ):
                stats = self.engine.rag_system.get_cache_stats()
                if self.logger:
                    self.logger.cache_stats(stats)
            else:
                if self.logger:
                    self.logger.warning(
                        "Cache statistics not available (RAG caching may be disabled)"
                    )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error displaying cache stats: {e}")

    def _show_help(self, args):
        if self.logger:
            self.logger.system_message("JENOVA Available Commands:")
            self.logger.system_message("")
            self.logger.system_message("=== Cognitive Commands ===")
            self.logger.system_message("/help              - Show this help message")
            self.logger.system_message(
                "/insight           - Develop insights from conversation history"
            )
            self.logger.system_message(
                "/reflect           - Reflect on existing insights"
            )
            self.logger.system_message(
                "/memory-insight    - Develop insights from memory"
            )
            self.logger.system_message(
                "/meta              - Generate meta-insight from all insights"
            )
            self.logger.system_message(
                "/verify            - Verify pending assumptions"
            )
            self.logger.system_message(
                "/develop_insight   - Develop insight from docs or specific input"
            )
            self.logger.system_message(
                "/learn_procedure   - Learn a new procedure interactively"
            )
            self.logger.system_message(
                "/train             - Show fine-tuning data generation help"
            )
            self.logger.system_message("")
            self.logger.system_message("=== System Monitoring (Phase 6) ===")
            self.logger.system_message(
                "/health            - Show system health (CPU, memory, GPU)"
            )
            self.logger.system_message("/metrics           - Show performance metrics")
            self.logger.system_message(
                "/status            - Show complete system status"
            )
            self.logger.system_message("/cache             - Show RAG cache statistics")
            self.logger.system_message("")

    def _show_train_help(self, args):
        if self.logger:
            self.logger.system_message("Fine-Tuning Data Generation")
            self.logger.system_message("")
            self.logger.system_message(
                "To generate comprehensive training data from your cognitive architecture, run:"
            )
            self.logger.system_message("  python finetune/train.py")
            self.logger.system_message("")
            self.logger.system_message(
                "This creates 'finetune_train.jsonl' from all knowledge sources."
            )
            self.logger.system_message(
                "Use this file with external fine-tuning tools like llama.cpp or Axolotl."
            )
            self.logger.system_message("")
            self.logger.system_message(
                "See finetune/README.md for detailed instructions."
            )
