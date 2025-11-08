"""
JENOVA Cognitive Architecture - Background Task Manager Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides background task management with comprehensive thread safety,
process monitoring, and resource tracking.

Phase 20 Enhancements:
- Fixed race conditions with proper locking on all shared state
- Thread-safe task registry using threading.RLock
- Atomic operations for task state transitions
- Safe concurrent access to task output streams
- Enhanced error handling and resource cleanup
- Complete type hints for all functions and methods
"""

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Deque
import psutil

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a background task."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class BackgroundTask:
    """
    Represents a background task.

    Thread-safety: All access to this class should be protected by
    BackgroundTaskManager._lock.
    """

    task_id: int
    command: List[str]
    process: Optional[subprocess.Popen] = None
    status: TaskStatus = TaskStatus.STARTING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: Deque[str] = field(default_factory=deque)
    stderr: Deque[str] = field(default_factory=deque)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary.

        Thread-safe: Creates snapshot of current state.
        """
        with self._lock:
            return {
                "task_id": self.task_id,
                "command": " ".join(self.command),
                "status": self.status.value,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "exit_code": self.exit_code,
                "pid": self.process.pid if self.process else None,
                "metadata": self.metadata.copy(),
                "stdout_lines": len(self.stdout),
                "stderr_lines": len(self.stderr),
            }

    def append_stdout(self, line: str, max_lines: int) -> None:
        """
        Append line to stdout with size limit.

        Thread-safe: Protected by task lock.
        """
        with self._lock:
            self.stdout.append(line)
            while len(self.stdout) > max_lines:
                self.stdout.popleft()

    def append_stderr(self, line: str, max_lines: int) -> None:
        """
        Append line to stderr with size limit.

        Thread-safe: Protected by task lock.
        """
        with self._lock:
            self.stderr.append(line)
            while len(self.stderr) > max_lines:
                self.stderr.popleft()

    def get_output_copy(self) -> Dict[str, List[str]]:
        """
        Get copy of stdout and stderr.

        Thread-safe: Creates snapshot.
        """
        with self._lock:
            return {
                "stdout": list(self.stdout),
                "stderr": list(self.stderr)
            }


class BackgroundTaskManager:
    """
    Manages background tasks with process monitoring using psutil.

    Thread-safety: All public methods are thread-safe. Internal state
    is protected by self._lock.

    Features:
    - Start and monitor background processes
    - Capture stdout/stderr with size limits
    - Process resource monitoring (CPU, memory)
    - Graceful termination with timeout
    - Automatic cleanup
    - Thread-safe concurrent access
    """

    def __init__(self, max_output_lines: int = 1000):
        """
        Initialize the background task manager.

        Args:
            max_output_lines: Maximum lines of output to keep per task
        """
        self._tasks: Dict[int, BackgroundTask] = {}
        self._max_output_lines = max_output_lines
        self._next_task_id = 0
        self._lock = threading.RLock()  # Reentrant lock for nested access
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False

    def start(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Start a background task.

        Thread-safe: Can be called from multiple threads.

        Args:
            command: Command and arguments as list
            cwd: Working directory for the command
            env: Environment variables
            metadata: Optional metadata for the task

        Returns:
            Task ID

        Raises:
            RuntimeError: If task failed to start
        """
        # Allocate task ID under lock
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1

        # Create task object
        task = BackgroundTask(
            task_id=task_id,
            command=command,
            metadata=metadata or {}
        )

        try:
            # Start process (outside lock - I/O operation)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env,
                bufsize=1,
            )

            # Update task state under lock
            with task._lock:
                task.process = process
                task.status = TaskStatus.RUNNING

            # Register task under lock
            with self._lock:
                self._tasks[task_id] = task

            logger.info(
                f"Started background task {task_id}: {' '.join(command)} (PID: {process.pid})"
            )

            # Ensure monitoring is running
            self._ensure_monitoring()

            return task_id

        except Exception as e:
            with task._lock:
                task.status = TaskStatus.FAILED
                task.end_time = datetime.now()
            logger.error(f"Failed to start task {task_id}: {e}")
            raise RuntimeError(f"Failed to start task: {e}")

    def _ensure_monitoring(self) -> None:
        """
        Ensure the monitoring thread is running.

        Thread-safe: Protected by manager lock.
        """
        with self._lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_tasks,
                    daemon=True,
                    name="BackgroundTaskMonitor"
                )
                self._monitor_thread.start()
                logger.debug("Started task monitoring thread")

    def _monitor_tasks(self) -> None:
        """
        Monitor running tasks and collect output.

        Runs in separate thread. Uses proper locking for all shared state access.
        """
        while self._monitoring:
            # Get snapshot of tasks to check
            with self._lock:
                tasks_snapshot = list(self._tasks.values())

            # Check each task (outside lock to avoid blocking)
            for task in tasks_snapshot:
                # Skip if not running
                with task._lock:
                    if task.status != TaskStatus.RUNNING or not task.process:
                        continue
                    process = task.process  # Capture reference

                try:
                    # Check if process is still running
                    poll_result = process.poll()

                    if poll_result is not None:
                        # Process has ended - update state under lock
                        with task._lock:
                            task.status = (
                                TaskStatus.COMPLETED
                                if poll_result == 0
                                else TaskStatus.FAILED
                            )
                            task.exit_code = poll_result
                            task.end_time = datetime.now()

                        # Collect remaining output
                        self._collect_output(task)

                        logger.info(
                            f"Task {task.task_id} ended with exit code {poll_result}"
                        )
                    else:
                        # Process still running, collect output
                        self._collect_output(task)

                except Exception as e:
                    logger.error(f"Error monitoring task {task.task_id}: {e}")

            # Sleep to avoid busy-waiting
            time.sleep(0.1)  # Check every 100ms

    def _collect_output(self, task: BackgroundTask) -> None:
        """
        Collect output from a task.

        Thread-safe: Uses task lock for output access.

        Args:
            task: Task to collect output from
        """
        with task._lock:
            if not task.process:
                return
            process = task.process

        # Read stdout (outside lock - I/O operation)
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                task.append_stdout(line.rstrip(), self._max_output_lines)
        except Exception:
            pass  # Process may have terminated

        # Read stderr (outside lock - I/O operation)
        try:
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                task.append_stderr(line.rstrip(), self._max_output_lines)
        except Exception:
            pass  # Process may have terminated

    def stop(self, task_id: int, timeout: float = 5.0) -> bool:
        """
        Stop a running task gracefully.

        Thread-safe: Can be called from multiple threads.

        Args:
            task_id: Task ID to stop
            timeout: Timeout for graceful termination

        Returns:
            True if task was stopped, False otherwise
        """
        # Get task under lock
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

        # Check status and get process under task lock
        with task._lock:
            if task.status != TaskStatus.RUNNING or not task.process:
                return False
            process = task.process
            pid = process.pid

        # Terminate outside locks (I/O operation)
        try:
            logger.info(f"Terminating task {task_id} (PID: {pid})")
            process.terminate()

            try:
                process.wait(timeout=timeout)

                # Update state under lock
                with task._lock:
                    task.status = TaskStatus.TERMINATED
                    task.end_time = datetime.now()
                    task.exit_code = process.returncode

                logger.info(f"Task {task_id} terminated gracefully")
                return True

            except subprocess.TimeoutExpired:
                # Force kill
                logger.warning(
                    f"Task {task_id} did not terminate gracefully, forcing kill"
                )
                process.kill()
                process.wait()

                # Update state under lock
                with task._lock:
                    task.status = TaskStatus.TERMINATED
                    task.end_time = datetime.now()
                    task.exit_code = process.returncode

                return True

        except Exception as e:
            logger.error(f"Error stopping task {task_id}: {e}")
            return False

    def get_status(self, task_id: int) -> Optional[TaskStatus]:
        """
        Get status of a task.

        Thread-safe: Returns snapshot of status.

        Args:
            task_id: Task ID

        Returns:
            TaskStatus or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

        with task._lock:
            return task.status

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a task by ID.

        Thread-safe: Returns snapshot of task state.

        Args:
            task_id: Task ID

        Returns:
            Task dictionary or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

        return task.to_dict()

    def get_output(self, task_id: int) -> Optional[Dict[str, List[str]]]:
        """
        Get stdout and stderr of a task.

        Thread-safe: Returns snapshot of output.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with 'stdout' and 'stderr' keys, or None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

        return task.get_output_copy()

    def get_resource_usage(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get resource usage of a running task using psutil.

        Thread-safe: Safe to call concurrently.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with CPU and memory usage, or None
        """
        # Get task and process under lock
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

        with task._lock:
            if task.status != TaskStatus.RUNNING or not task.process:
                return None
            pid = task.process.pid

        # Get resource usage outside lock (I/O operation)
        try:
            process = psutil.Process(pid)

            return {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(
                    process.create_time()
                ).isoformat(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Error getting resource usage for task {task_id}: {e}")
            return None

    def list_tasks(
        self, status_filter: Optional[TaskStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List all tasks.

        Thread-safe: Returns snapshot of tasks.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of task dictionaries
        """
        # Get snapshot of tasks under lock
        with self._lock:
            tasks_snapshot = list(self._tasks.values())

        # Build result list
        result = []
        for task in tasks_snapshot:
            if status_filter is None or task.status == status_filter:
                result.append(task.to_dict())

        return result

    def cleanup_completed(self) -> int:
        """
        Remove completed tasks from memory.

        Thread-safe: Safe to call concurrently.

        Returns:
            Number of tasks removed
        """
        with self._lock:
            tasks_to_remove = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.TERMINATED
                )
            ]

            for task_id in tasks_to_remove:
                del self._tasks[task_id]

            count = len(tasks_to_remove)
            if count > 0:
                logger.info(f"Cleaned up {count} completed tasks")

            return count

    def stop_all(self, timeout: float = 5.0) -> int:
        """
        Stop all running tasks.

        Thread-safe: Safe to call concurrently.

        Args:
            timeout: Timeout for each task

        Returns:
            Number of tasks stopped
        """
        # Get snapshot of running tasks
        with self._lock:
            running_tasks = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status == TaskStatus.RUNNING
            ]

        # Stop each task
        count = 0
        for task_id in running_tasks:
            if self.stop(task_id, timeout=timeout):
                count += 1

        if count > 0:
            logger.info(f"Stopped {count} running tasks")

        return count

    def shutdown(self) -> None:
        """
        Shutdown the task manager and stop all tasks.

        Thread-safe: Safe to call from any thread.
        """
        logger.info("Shutting down BackgroundTaskManager")

        # Stop monitoring thread
        with self._lock:
            self._monitoring = False

        # Wait for monitor thread
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        # Stop all tasks
        self.stop_all()

        logger.info("BackgroundTaskManager shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
