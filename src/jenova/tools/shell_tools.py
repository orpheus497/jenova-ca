# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Shell Tools

"""
Shell command execution tool module.

This module provides secure shell command execution with whitelist validation,
timeout protection, and comprehensive error handling.
"""

import shlex
import subprocess
from typing import Any, Dict, List, Optional

from jenova.tools.base import BaseTool, ToolResult
from jenova.config.constants import DEFAULT_COMMAND_TIMEOUT_SECONDS


class ShellTools(BaseTool):
    """
    Secure shell command execution tool.

    Provides shell command execution with security controls:
    - Command whitelist validation
    - Argument parsing with shlex
    - Timeout protection
    - Safe subprocess execution (no shell=True)

    Methods:
        execute: Execute shell command with security controls
        is_command_allowed: Check if command is whitelisted
        get_whitelisted_commands: Get list of allowed commands
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ui_logger: Any,
        file_logger: Any,
        whitelist: Optional[List[str]] = None
    ):
        """
        Initialize shell tools.

        Args:
            config: Configuration dictionary
            ui_logger: UI logger instance
            file_logger: File logger instance
            whitelist: List of allowed commands (from config if None)
        """
        super().__init__(
            name="shell_tools",
            description="Secure shell command execution",
            config=config,
            ui_logger=ui_logger,
            file_logger=file_logger
        )

        # Load whitelist from config or use provided list
        if whitelist is not None:
            self.whitelist = whitelist
        else:
            tools_config = config.get('tools', {})
            self.whitelist = tools_config.get('shell_command_whitelist', [
                'ls', 'cat', 'grep', 'find', 'echo', 'date',
                'whoami', 'pwd', 'uname'
            ])

        self.timeout = config.get('tools', {}).get(
            'command_timeout',
            DEFAULT_COMMAND_TIMEOUT_SECONDS
        )

    def execute(
        self,
        command: str,
        timeout: Optional[int] = None
    ) -> ToolResult:
        """
        Execute shell command with security controls.

        Args:
            command: Shell command string to execute
            timeout: Optional timeout in seconds (uses default if None)

        Returns:
            ToolResult with command output or error

        Example:
            >>> tools = ShellTools(config, ui_logger, file_logger)
            >>> result = tools.execute("ls -la /tmp")
            >>> if result.success:
            >>>     print(result.data['stdout'])

        Raises:
            ToolError: If command execution fails
        """
        timeout = timeout or self.timeout

        try:
            # Parse command safely with shlex
            command_args = shlex.split(command)

            if not command_args:
                return self._create_error_result(
                    error="Empty command provided",
                    metadata={'command': command}
                )

            # Extract command name (first argument)
            command_name = command_args[0]

            # Validate against whitelist
            if not self.is_command_allowed(command_name):
                error_msg = (
                    f"Command '{command_name}' not allowed. "
                    f"Allowed commands: {', '.join(self.whitelist)}"
                )
                result = self._create_error_result(
                    error=error_msg,
                    metadata={
                        'command': command,
                        'command_name': command_name,
                        'reason': 'not_whitelisted'
                    }
                )
                self._log_execution({'command': command}, result)
                return result

            # Execute command safely (no shell=True)
            proc_result = subprocess.run(
                command_args,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )

            # Create result data
            data = {
                'stdout': proc_result.stdout,
                'stderr': proc_result.stderr,
                'returncode': proc_result.returncode,
                'command': command,
                'success': proc_result.returncode == 0
            }

            result = self._create_success_result(
                data=data,
                metadata={
                    'command_name': command_name,
                    'args_count': len(command_args) - 1,
                    'execution_time': timeout  # Could track actual time
                }
            )

            self._log_execution({'command': command}, result)
            return result

        except FileNotFoundError:
            error_msg = f"Command not found: {command_args[0]}"
            result = self._create_error_result(
                error=error_msg,
                metadata={'command': command, 'reason': 'command_not_found'}
            )
            self._log_execution({'command': command}, result)
            return result

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds"
            result = self._create_error_result(
                error=error_msg,
                metadata={'command': command, 'timeout': timeout, 'reason': 'timeout'}
            )
            self._log_execution({'command': command}, result)
            return result

        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            result = self._create_error_result(
                error=error_msg,
                metadata={'command': command, 'exception': type(e).__name__}
            )
            self._log_execution({'command': command}, result)
            return result

    def is_command_allowed(self, command_name: str) -> bool:
        """
        Check if command is in whitelist.

        Args:
            command_name: Name of command to check

        Returns:
            True if command is whitelisted

        Example:
            >>> tools = ShellTools(config, ui_logger, file_logger)
            >>> tools.is_command_allowed('ls')  # True
            >>> tools.is_command_allowed('rm')  # False
        """
        return command_name in self.whitelist

    def get_whitelisted_commands(self) -> List[str]:
        """
        Get list of allowed commands.

        Returns:
            List of whitelisted command names

        Example:
            >>> tools = ShellTools(config, ui_logger, file_logger)
            >>> commands = tools.get_whitelisted_commands()
            >>> print(commands)  # ['ls', 'cat', 'grep', ...]
        """
        return self.whitelist.copy()

    def add_to_whitelist(self, command: str) -> None:
        """
        Add command to whitelist (runtime only, not persisted).

        Args:
            command: Command name to whitelist

        Example:
            >>> tools = ShellTools(config, ui_logger, file_logger)
            >>> tools.add_to_whitelist('tree')
        """
        if command not in self.whitelist:
            self.whitelist.append(command)
            if self.file_logger:
                self.file_logger.log_info(
                    f"Added '{command}' to shell command whitelist (runtime only)"
                )

    def remove_from_whitelist(self, command: str) -> None:
        """
        Remove command from whitelist (runtime only).

        Args:
            command: Command name to remove

        Example:
            >>> tools = ShellTools(config, ui_logger, file_logger)
            >>> tools.remove_from_whitelist('echo')
        """
        if command in self.whitelist:
            self.whitelist.remove(command)
            if self.file_logger:
                self.file_logger.log_info(
                    f"Removed '{command}' from shell command whitelist (runtime only)"
                )
