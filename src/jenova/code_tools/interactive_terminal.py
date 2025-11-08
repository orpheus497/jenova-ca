# The JENOVA Cognitive Architecture - Interactive Terminal
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Interactive terminal support using PTY.

Provides pseudo-terminal support for interactive commands like vim, git rebase -i, etc.
"""

import os
import sys
import select
import subprocess
import threading
from typing import Optional, List, Callable


class InteractiveTerminal:
    """
    Interactive terminal handler using PTY.

    Capabilities:
    - PTY support for interactive commands
    - Real-time output streaming
    - Background process management
    - Terminal state preservation
    """

    def __init__(self, ui_logger=None, file_logger=None):
        """
        Initialize interactive terminal.

        Args:
            ui_logger: UI logger for user feedback
            file_logger: File logger for operation logging
        """
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.active_processes = {}
        self.process_counter = 0

    def run_interactive(
        self, command: List[str], cwd: Optional[str] = None, env: Optional[dict] = None
    ) -> int:
        """
        Run interactive command with PTY support.

        Args:
            command: Command and arguments
            cwd: Working directory
            env: Environment variables

        Returns:
            Exit code
        """
        try:
            import pty
            import termios
            import tty

            # Save terminal state
            if sys.stdin.isatty():
                old_settings = termios.tcgetattr(sys.stdin)
            else:
                old_settings = None

            try:
                # Create PTY
                master, slave = pty.openpty()

                # Start process
                process = subprocess.Popen(
                    command,
                    stdin=slave,
                    stdout=slave,
                    stderr=slave,
                    cwd=cwd,
                    env=env or os.environ.copy(),
                )

                os.close(slave)

                if old_settings:
                    # Set terminal to raw mode
                    tty.setraw(sys.stdin.fileno())

                # Handle I/O
                self._pty_io_loop(master, process)

                # Wait for process
                exit_code = process.wait()

                return exit_code

            finally:
                # Restore terminal state
                if old_settings:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                os.close(master)

        except ImportError:
            # Fallback to regular subprocess if PTY not available
            if self.file_logger:
                self.file_logger.log_warning("PTY not available, using subprocess")

            return self.run_command(command, cwd, env)

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error running interactive command: {e}")
            return 1

    def _pty_io_loop(self, master: int, process: subprocess.Popen):
        """
        Handle bidirectional I/O for PTY.

        Args:
            master: Master PTY file descriptor
            process: Process object
        """
        while True:
            try:
                # Check if process is still running
                if process.poll() is not None:
                    # Read any remaining output
                    try:
                        while True:
                            chunk = os.read(master, 1024)
                            if not chunk:
                                break
                            os.write(sys.stdout.fileno(), chunk)
                    except OSError:
                        pass
                    break

                # Use select to check for input/output
                rlist, _, _ = select.select([master, sys.stdin], [], [], 0.1)

                if master in rlist:
                    # Read from PTY and write to stdout
                    try:
                        chunk = os.read(master, 1024)
                        if chunk:
                            os.write(sys.stdout.fileno(), chunk)
                    except OSError:
                        break

                if sys.stdin in rlist and sys.stdin.isatty():
                    # Read from stdin and write to PTY
                    try:
                        chunk = os.read(sys.stdin.fileno(), 1024)
                        if chunk:
                            os.write(master, chunk)
                    except OSError:
                        break

            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Error in PTY I/O loop: {e}")
                break

    def run_command(
        self, command: List[str], cwd: Optional[str] = None, env: Optional[dict] = None
    ) -> int:
        """
        Run command without PTY (fallback).

        Args:
            command: Command and arguments
            cwd: Working directory
            env: Environment variables

        Returns:
            Exit code
        """
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env or os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in process.stdout:
                print(line, end="")

            exit_code = process.wait()
            return exit_code

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error running command: {e}")
            return 1

    def run_background(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        Run command in background and return process ID.

        Args:
            command: Command and arguments
            cwd: Working directory
            env: Environment variables
            output_callback: Callback for output lines

        Returns:
            Process ID
        """
        process_id = self.process_counter
        self.process_counter += 1

        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env or os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Store process
            self.active_processes[process_id] = {
                "process": process,
                "command": command,
                "output": [],
            }

            # Start output monitoring thread
            def monitor_output():
                for line in process.stdout:
                    self.active_processes[process_id]["output"].append(line)
                    if output_callback:
                        output_callback(line)

            thread = threading.Thread(target=monitor_output, daemon=True)
            thread.start()

            if self.file_logger:
                self.file_logger.log_info(
                    f"Started background process {process_id}: {' '.join(command)}"
                )

            return process_id

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error starting background process: {e}")
            return -1

    def get_process_status(self, process_id: int) -> Optional[dict]:
        """
        Get status of background process.

        Args:
            process_id: Process ID

        Returns:
            Status dict or None if not found
        """
        if process_id not in self.active_processes:
            return None

        proc_info = self.active_processes[process_id]
        process = proc_info["process"]

        return {
            "id": process_id,
            "command": " ".join(proc_info["command"]),
            "running": process.poll() is None,
            "exit_code": process.poll(),
            "output_lines": len(proc_info["output"]),
        }

    def get_process_output(
        self, process_id: int, last_n: Optional[int] = None
    ) -> Optional[List[str]]:
        """
        Get output from background process.

        Args:
            process_id: Process ID
            last_n: Return only last N lines

        Returns:
            Output lines or None if not found
        """
        if process_id not in self.active_processes:
            return None

        output = self.active_processes[process_id]["output"]

        if last_n:
            return output[-last_n:]

        return output

    def kill_process(self, process_id: int) -> bool:
        """
        Kill background process.

        Args:
            process_id: Process ID

        Returns:
            True if successful, False otherwise
        """
        if process_id not in self.active_processes:
            return False

        try:
            process = self.active_processes[process_id]["process"]
            process.terminate()

            # Wait for termination
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

            del self.active_processes[process_id]

            if self.file_logger:
                self.file_logger.log_info(f"Killed process {process_id}")

            return True

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error killing process {process_id}: {e}")
            return False

    def list_processes(self) -> List[dict]:
        """
        List all background processes.

        Returns:
            List of process status dicts
        """
        statuses = []

        for process_id in list(self.active_processes.keys()):
            status = self.get_process_status(process_id)
            if status:
                statuses.append(status)

        return statuses

    def cleanup(self):
        """Clean up all background processes."""
        for process_id in list(self.active_processes.keys()):
            self.kill_process(process_id)
