"""
JENOVA Cognitive Architecture - Background Task Manager Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides background task management using psutil for process monitoring.
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
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
    """Represents a background task."""

    task_id: int
    command: List[str]
    process: Optional[subprocess.Popen] = None
    status: TaskStatus = TaskStatus.STARTING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: List[str] = field(default_factory=list)
    stderr: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "command": " ".join(self.command),
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "exit_code": self.exit_code,
            "pid": self.process.pid if self.process else None,
            "metadata": self.metadata
        }


class BackgroundTaskManager:
    """
    Manages background tasks with process monitoring using psutil.

    Features:
    - Start and monitor background processes
    - Capture stdout/stderr
    - Process resource monitoring (CPU, memory)
    - Graceful termination
    - Automatic cleanup
    """

    def __init__(self, max_output_lines: int = 1000):
        """
        Initialize the background task manager.

        Args:
            max_output_lines: Maximum lines of output to keep per task
        """
        self.tasks: Dict[int, BackgroundTask] = {}
        self.max_output_lines = max_output_lines
        self._next_task_id = 0
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._monitoring = False

    def start(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Start a background task.

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
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1

        task = BackgroundTask(
            task_id=task_id,
            command=command,
            metadata=metadata or {}
        )

        try:
            # Start process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env,
                bufsize=1
            )

            task.process = process
            task.status = TaskStatus.RUNNING

            with self._lock:
                self.tasks[task_id] = task

            logger.info(f"Started background task {task_id}: {' '.join(command)} (PID: {process.pid})")

            # Start monitoring if not already running
            self._ensure_monitoring()

            return task_id

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            logger.error(f"Failed to start task {task_id}: {e}")
            raise RuntimeError(f"Failed to start task: {e}")

    def _ensure_monitoring(self) -> None:
        """Ensure the monitoring thread is running."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_tasks, daemon=True)
            self._monitor_thread.start()
            logger.debug("Started task monitoring thread")

    def _monitor_tasks(self) -> None:
        """Monitor running tasks and collect output."""
        while self._monitoring:
            tasks_to_check = list(self.tasks.values())

            for task in tasks_to_check:
                if task.status != TaskStatus.RUNNING or not task.process:
                    continue

                try:
                    # Check if process is still running
                    poll_result = task.process.poll()

                    if poll_result is not None:
                        # Process has ended
                        task.status = TaskStatus.COMPLETED if poll_result == 0 else TaskStatus.FAILED
                        task.exit_code = poll_result
                        task.end_time = datetime.now()

                        # Collect remaining output
                        self._collect_output(task)

                        logger.info(f"Task {task.task_id} ended with exit code {poll_result}")
                    else:
                        # Process still running, collect output
                        self._collect_output(task)

                except Exception as e:
                    logger.error(f"Error monitoring task {task.task_id}: {e}")

            time.sleep(0.1)  # Check every 100ms

    def _collect_output(self, task: BackgroundTask) -> None:
        """Collect output from a task."""
        if not task.process:
            return

        # Read stdout
        try:
            while True:
                line = task.process.stdout.readline()
                if not line:
                    break
                task.stdout.append(line.rstrip())

                # Limit output size
                if len(task.stdout) > self.max_output_lines:
                    task.stdout.pop(0)
        except Exception:
            pass

        # Read stderr
        try:
            while True:
                line = task.process.stderr.readline()
                if not line:
                    break
                task.stderr.append(line.rstrip())

                # Limit output size
                if len(task.stderr) > self.max_output_lines:
                    task.stderr.pop(0)
        except Exception:
            pass

    def stop(self, task_id: int, timeout: float = 5.0) -> bool:
        """
        Stop a running task gracefully.

        Args:
            task_id: Task ID to stop
            timeout: Timeout for graceful termination

        Returns:
            True if task was stopped, False otherwise
        """
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        if not task.process:
            return False

        try:
            # Try graceful termination first
            logger.info(f"Terminating task {task_id} (PID: {task.process.pid})")
            task.process.terminate()

            try:
                task.process.wait(timeout=timeout)
                task.status = TaskStatus.TERMINATED
                task.end_time = datetime.now()
                task.exit_code = task.process.returncode
                logger.info(f"Task {task_id} terminated gracefully")
                return True
            except subprocess.TimeoutExpired:
                # Force kill
                logger.warning(f"Task {task_id} did not terminate gracefully, forcing kill")
                task.process.kill()
                task.process.wait()
                task.status = TaskStatus.TERMINATED
                task.end_time = datetime.now()
                task.exit_code = task.process.returncode
                return True

        except Exception as e:
            logger.error(f"Error stopping task {task_id}: {e}")
            return False

    def get_status(self, task_id: int) -> Optional[TaskStatus]:
        """
        Get status of a task.

        Args:
            task_id: Task ID

        Returns:
            TaskStatus or None if not found
        """
        task = self.tasks.get(task_id)
        return task.status if task else None

    def get_task(self, task_id: int) -> Optional[BackgroundTask]:
        """
        Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            BackgroundTask or None if not found
        """
        return self.tasks.get(task_id)

    def get_output(self, task_id: int) -> Optional[Dict[str, List[str]]]:
        """
        Get stdout and stderr of a task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with 'stdout' and 'stderr' keys, or None
        """
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            "stdout": task.stdout.copy(),
            "stderr": task.stderr.copy()
        }

    def get_resource_usage(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get resource usage of a running task using psutil.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with CPU and memory usage, or None
        """
        task = self.tasks.get(task_id)
        if not task or not task.process or task.status != TaskStatus.RUNNING:
            return None

        try:
            process = psutil.Process(task.process.pid)

            return {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Error getting resource usage for task {task_id}: {e}")
            return None

    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """
        List all tasks.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of task dictionaries
        """
        tasks = []

        for task in self.tasks.values():
            if status_filter is None or task.status == status_filter:
                tasks.append(task.to_dict())

        return tasks

    def cleanup_completed(self) -> int:
        """
        Remove completed tasks from memory.

        Returns:
            Number of tasks removed
        """
        with self._lock:
            tasks_to_remove = [
                task_id for task_id, task in self.tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TERMINATED)
            ]

            for task_id in tasks_to_remove:
                del self.tasks[task_id]

            logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
            return len(tasks_to_remove)

    def stop_all(self, timeout: float = 5.0) -> int:
        """
        Stop all running tasks.

        Args:
            timeout: Timeout for each task

        Returns:
            Number of tasks stopped
        """
        running_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.status == TaskStatus.RUNNING
        ]

        count = 0
        for task_id in running_tasks:
            if self.stop(task_id, timeout=timeout):
                count += 1

        logger.info(f"Stopped {count} running tasks")
        return count

    def shutdown(self) -> None:
        """Shutdown the task manager and stop all tasks."""
        logger.info("Shutting down BackgroundTaskManager")

        # Stop monitoring
        self._monitoring = False
        if self._monitor_thread:
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
