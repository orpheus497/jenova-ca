# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 21: Full Project Remediation & Modernization
# Main Application Class

"""
Main application class for The JENOVA Cognitive Architecture.

This module provides the top-level Application class that orchestrates
the entire system lifecycle, from initialization through shutdown.

Example:
    >>> from jenova.core import Application
    >>> app = Application()
    >>> exit_code = app.run()
    >>> sys.exit(exit_code)
"""

import signal
import sys
import traceback
from typing import Optional

from jenova.core.bootstrap import ApplicationBootstrapper
from jenova.core.container import DependencyContainer
from jenova.core.lifecycle import ComponentLifecycle


class Application:
    """
    Main application class for JENOVA Cognitive Architecture.

    This class provides the top-level entry point for the application,
    managing the complete lifecycle from bootstrapping through shutdown.

    Attributes:
        container: Dependency injection container
        lifecycle: Component lifecycle manager
        bootstrapper: Application bootstrapper
        ui_logger: UI logger instance
        file_logger: File logger instance

    Example:
        >>> app = Application()
        >>> exit_code = app.run()
    """

    def __init__(self, username: Optional[str] = None):
        """
        Initialize application.

        Args:
            username: Optional username (defaults to current system user)
        """
        self.container: Optional[DependencyContainer] = None
        self.lifecycle: Optional[ComponentLifecycle] = None
        self.bootstrapper = ApplicationBootstrapper(username=username)
        self.ui_logger = None
        self.file_logger = None
        self._shutdown_requested = False

    def run(self) -> int:
        """
        Run the application.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            # Register signal handlers
            self._register_signal_handlers()

            # Bootstrap application
            self.container = self.bootstrapper.bootstrap()

            # Get logger instances
            self.ui_logger = self.container.resolve('ui_logger')
            self.file_logger = self.container.resolve('file_logger')

            # Initialize component lifecycle (if needed)
            # Note: Most components are already initialized by bootstrapper
            # This is mainly for future extensibility

            # Get cognitive engine and terminal UI
            cognitive_engine = self.container.resolve('cognitive_engine')

            # Import TerminalUI here to avoid circular dependencies
            from jenova.ui.terminal import TerminalUI

            # Get all optional components for TerminalUI
            health_monitor = self.container.resolve('health_monitor') if self.container.is_registered('health_monitor') else None
            metrics = self.container.resolve('metrics') if self.container.is_registered('metrics') else None
            backup_manager = self.container.resolve('backup_manager') if self.container.is_registered('backup_manager') else None

            # Get CLI enhancement modules (Phase 13-17)
            context_optimizer = self.container.resolve('context_optimizer') if self.container.is_registered('context_optimizer') else None
            code_metrics = self.container.resolve('code_metrics') if self.container.is_registered('code_metrics') else None
            security_scanner = self.container.resolve('security_scanner') if self.container.is_registered('security_scanner') else None
            intent_classifier = self.container.resolve('intent_classifier') if self.container.is_registered('intent_classifier') else None
            command_disambiguator = self.container.resolve('command_disambiguator') if self.container.is_registered('command_disambiguator') else None
            file_editor = self.container.resolve('file_editor') if self.container.is_registered('file_editor') else None
            code_parser = self.container.resolve('code_parser') if self.container.is_registered('code_parser') else None
            refactoring_engine = self.container.resolve('refactoring_engine') if self.container.is_registered('refactoring_engine') else None
            syntax_highlighter = self.container.resolve('syntax_highlighter') if self.container.is_registered('syntax_highlighter') else None
            codebase_mapper = self.container.resolve('codebase_mapper') if self.container.is_registered('codebase_mapper') else None
            interactive_terminal = self.container.resolve('interactive_terminal') if self.container.is_registered('interactive_terminal') else None
            git_interface = self.container.resolve('git_interface') if self.container.is_registered('git_interface') else None
            commit_assistant = self.container.resolve('commit_assistant') if self.container.is_registered('commit_assistant') else None
            diff_analyzer = self.container.resolve('diff_analyzer') if self.container.is_registered('diff_analyzer') else None
            hooks_manager = self.container.resolve('hooks_manager') if self.container.is_registered('hooks_manager') else None
            branch_manager = self.container.resolve('branch_manager') if self.container.is_registered('branch_manager') else None
            task_planner = self.container.resolve('task_planner') if self.container.is_registered('task_planner') else None
            subagent_manager = self.container.resolve('subagent_manager') if self.container.is_registered('subagent_manager') else None
            execution_engine = self.container.resolve('execution_engine') if self.container.is_registered('execution_engine') else None
            checkpoint_manager = self.container.resolve('checkpoint_manager') if self.container.is_registered('checkpoint_manager') else None
            background_task_manager = self.container.resolve('background_task_manager') if self.container.is_registered('background_task_manager') else None
            custom_command_manager = self.container.resolve('custom_command_manager') if self.container.is_registered('custom_command_manager') else None
            hooks_system = self.container.resolve('hooks_system') if self.container.is_registered('hooks_system') else None
            template_engine = self.container.resolve('template_engine') if self.container.is_registered('template_engine') else None
            workflow_library = self.container.resolve('workflow_library') if self.container.is_registered('workflow_library') else None

            # Create and run Terminal UI
            terminal_ui = TerminalUI(
                cognitive_engine,
                self.ui_logger,
                health_monitor=health_monitor,
                metrics=metrics,
                backup_manager=backup_manager,
                # Phase 13-17 CLI enhancements
                context_optimizer=context_optimizer,
                code_metrics=code_metrics,
                security_scanner=security_scanner,
                intent_classifier=intent_classifier,
                command_disambiguator=command_disambiguator,
                file_editor=file_editor,
                code_parser=code_parser,
                refactoring_engine=refactoring_engine,
                syntax_highlighter=syntax_highlighter,
                codebase_mapper=codebase_mapper,
                interactive_terminal=interactive_terminal,
                git_interface=git_interface,
                commit_assistant=commit_assistant,
                diff_analyzer=diff_analyzer,
                hooks_manager=hooks_manager,
                branch_manager=branch_manager,
                task_planner=task_planner,
                subagent_manager=subagent_manager,
                execution_engine=execution_engine,
                checkpoint_manager=checkpoint_manager,
                background_task_manager=background_task_manager,
                custom_command_manager=custom_command_manager,
                hooks_system=hooks_system,
                template_engine=template_engine,
                workflow_library=workflow_library,
            )

            # Run the terminal UI (this blocks until user exits)
            terminal_ui.run()

            # Normal shutdown
            return 0

        except KeyboardInterrupt:
            if self.ui_logger:
                self.ui_logger.info("Shutdown requested by user")
            if self.file_logger:
                self.file_logger.log_info("Application shutdown via keyboard interrupt")
            return 0

        except Exception as e:
            error_msg = f"Critical application error: {e}"
            if self.ui_logger:
                self.ui_logger.error(error_msg)
            if self.file_logger:
                self.file_logger.log_error(error_msg)
                self.file_logger.log_error(f"Traceback: {traceback.format_exc()}")
            else:
                # Fallback if logger not available
                print(f"ERROR: {error_msg}", file=sys.stderr)
                traceback.print_exc()
            return 1

        finally:
            # Cleanup
            self._cleanup()

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            self._shutdown_requested = True
            if self.ui_logger:
                self.ui_logger.info(f"Received signal {signum}, shutting down...")
            if self.file_logger:
                self.file_logger.log_info(f"Shutdown signal received: {signum}")

        # Register handlers for common shutdown signals
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _cleanup(self) -> None:
        """Cleanup application resources."""
        if not self.container:
            return

        try:
            # Log final metrics summary
            if self.file_logger:
                self.file_logger.log_info("=== Shutdown Cleanup ===")

            # Cleanup LLM resources
            try:
                if self.container.is_registered('llm_interface'):
                    llm_interface = self.container.resolve('llm_interface')
                    if hasattr(llm_interface, 'close'):
                        llm_interface.close()
                        if self.file_logger:
                            self.file_logger.log_info("LLM interface closed")
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error closing LLM interface: {e}")

            # Cleanup model manager
            try:
                if self.container.is_registered('model_manager'):
                    model_manager = self.container.resolve('model_manager')
                    if hasattr(model_manager, 'unload_model'):
                        model_manager.unload_model()
                        if self.file_logger:
                            self.file_logger.log_info("Model unloaded")
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error unloading model: {e}")

            # Cleanup embedding manager
            try:
                if self.container.is_registered('embedding_manager'):
                    embedding_manager = self.container.resolve('embedding_manager')
                    if hasattr(embedding_manager, 'unload_model'):
                        embedding_manager.unload_model()
                        if self.file_logger:
                            self.file_logger.log_info("Embedding model unloaded")
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error unloading embedding model: {e}")

            # Cleanup network layer (Phase 8)
            try:
                if self.container.is_registered('discovery_service'):
                    discovery_service = self.container.resolve('discovery_service')
                    if hasattr(discovery_service, 'stop_advertising'):
                        discovery_service.stop_advertising()
                    if hasattr(discovery_service, 'stop_browsing'):
                        discovery_service.stop_browsing()
                    if self.file_logger:
                        self.file_logger.log_info("Discovery service stopped")
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error stopping discovery service: {e}")

            try:
                if self.container.is_registered('rpc_server'):
                    rpc_server = self.container.resolve('rpc_server')
                    if hasattr(rpc_server, 'stop'):
                        rpc_server.stop(grace_period=5)
                    if self.file_logger:
                        self.file_logger.log_info("RPC server stopped")
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error stopping RPC server: {e}")

            try:
                if self.container.is_registered('rpc_client'):
                    rpc_client = self.container.resolve('rpc_client')
                    if hasattr(rpc_client, 'close_all_connections'):
                        rpc_client.close_all_connections()
                    if self.file_logger:
                        self.file_logger.log_info("RPC client connections closed")
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error closing RPC client: {e}")

            try:
                if self.container.is_registered('security_manager'):
                    security_manager = self.container.resolve('security_manager')
                    if hasattr(security_manager, 'close'):
                        security_manager.close()
                    if self.file_logger:
                        self.file_logger.log_info("Security manager closed")
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error closing security manager: {e}")

            # Log network metrics summary
            try:
                if self.container.is_registered('network_metrics'):
                    network_metrics = self.container.resolve('network_metrics')
                    if hasattr(network_metrics, 'log_summary'):
                        network_metrics.log_summary(top_n=10)
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error logging network metrics: {e}")

            # Log final metrics summary
            try:
                if self.container.is_registered('metrics'):
                    metrics = self.container.resolve('metrics')
                    if hasattr(metrics, 'log_summary'):
                        if self.file_logger:
                            self.file_logger.log_info("=== Session Metrics Summary ===")
                        metrics.log_summary(top_n=10)
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(f"Error logging metrics: {e}")

            # Final shutdown message
            if self.ui_logger:
                self.ui_logger.info("JENOVA shutdown complete.")
            if self.file_logger:
                self.file_logger.log_info("=== Application Shutdown Complete ===")

        except Exception as e:
            # Suppress cleanup errors but log them
            if self.file_logger:
                self.file_logger.log_error(f"Error during cleanup: {e}")
