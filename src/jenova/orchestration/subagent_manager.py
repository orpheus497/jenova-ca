"""
JENOVA Cognitive Architecture - Subagent Manager Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides subagent lifecycle management for parallel task execution.
"""

import logging
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future


logger = logging.getLogger(__name__)


class SubagentStatus(Enum):
    """Status of a subagent."""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class SubagentTask:
    """Represents a task assigned to a subagent."""

    task_id: str
    description: str
    callable: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None


@dataclass
class Subagent:
    """Represents a subagent worker."""

    id: int
    status: SubagentStatus = SubagentStatus.IDLE
    current_task: Optional[SubagentTask] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert subagent to dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "current_task": self.current_task.task_id if self.current_task else None,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


class SubagentManager:
    """
    Manages subagent lifecycle and task distribution.

    Provides:
    - Dynamic subagent pool management
    - Task queue with priority
    - Parallel task execution
    - Resource management
    - Error handling and recovery
    """

    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        """
        Initialize the subagent manager.

        Args:
            max_workers: Maximum number of concurrent subagents
            queue_size: Maximum size of the task queue
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.subagents: Dict[int, Subagent] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self.active_futures: Dict[int, Future] = {}
        self.completed_tasks: List[SubagentTask] = []
        self.failed_tasks: List[SubagentTask] = []
        self._next_agent_id = 0
        self._lock = threading.Lock()
        self._shutdown = False

        # Initialize subagents
        for _ in range(max_workers):
            self._create_subagent()

    def _create_subagent(self) -> int:
        """
        Create a new subagent.

        Returns:
            Subagent ID
        """
        with self._lock:
            agent_id = self._next_agent_id
            self._next_agent_id += 1

            subagent = Subagent(id=agent_id)
            self.subagents[agent_id] = subagent

            logger.info(f"Created subagent {agent_id}")
            return agent_id

    def spawn(
        self,
        task_callable: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: int = 1,
        **kwargs
    ) -> str:
        """
        Spawn a task on a subagent.

        Args:
            task_callable: Function to execute
            *args: Positional arguments for the callable
            task_id: Optional task ID (auto-generated if not provided)
            priority: Task priority (higher = more important)
            **kwargs: Keyword arguments for the callable

        Returns:
            Task ID

        Raises:
            RuntimeError: If manager is shutdown or queue is full
        """
        if self._shutdown:
            raise RuntimeError("SubagentManager is shutdown")

        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{datetime.now().timestamp()}"

        # Create task
        task = SubagentTask(
            task_id=task_id,
            description=str(task_callable.__name__),
            callable=task_callable,
            args=args,
            kwargs=kwargs,
            priority=priority
        )

        # Add to queue (lower priority number = higher priority)
        try:
            self.task_queue.put((-priority, task), block=False)
            logger.info(f"Queued task {task_id} with priority {priority}")
        except queue.Full:
            raise RuntimeError("Task queue is full")

        # Try to execute immediately if workers available
        self._try_execute_next()

        return task_id

    def _try_execute_next(self) -> None:
        """Try to execute the next task in queue."""
        if self._shutdown:
            return

        # Find idle subagent
        idle_agent = None
        with self._lock:
            for agent in self.subagents.values():
                if agent.status == SubagentStatus.IDLE:
                    idle_agent = agent
                    break

        if idle_agent is None:
            return

        # Get next task from queue
        try:
            priority, task = self.task_queue.get(block=False)
        except queue.Empty:
            return

        # Execute task
        self._execute_task(idle_agent, task)

    def _execute_task(self, agent: Subagent, task: SubagentTask) -> None:
        """
        Execute a task on a subagent.

        Args:
            agent: Subagent to use
            task: Task to execute
        """
        with self._lock:
            agent.status = SubagentStatus.BUSY
            agent.current_task = task
            agent.last_activity = datetime.now()

        task.started_at = datetime.now()

        # Submit task to executor
        future = self.executor.submit(self._run_task, agent, task)
        self.active_futures[agent.id] = future

        # Add callback for completion
        future.add_done_callback(lambda f: self._task_completed(agent, task, f))

        logger.info(f"Subagent {agent.id} executing task {task.task_id}")

    def _run_task(self, agent: Subagent, task: SubagentTask) -> Any:
        """
        Run the task callable.

        Args:
            agent: Subagent running the task
            task: Task to run

        Returns:
            Task result
        """
        try:
            return task.callable(*task.args, **task.kwargs)
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            raise

    def _task_completed(self, agent: Subagent, task: SubagentTask, future: Future) -> None:
        """
        Handle task completion.

        Args:
            agent: Subagent that ran the task
            task: Completed task
            future: Future object
        """
        task.completed_at = datetime.now()

        with self._lock:
            # Remove from active futures
            if agent.id in self.active_futures:
                del self.active_futures[agent.id]

            # Get result or error
            try:
                result = future.result()
                task.result = result
                agent.completed_tasks += 1
                self.completed_tasks.append(task)
                logger.info(f"Task {task.task_id} completed successfully")
            except Exception as e:
                task.error = e
                agent.failed_tasks += 1
                agent.status = SubagentStatus.ERROR
                self.failed_tasks.append(task)
                logger.error(f"Task {task.task_id} failed: {e}")

            # Mark agent as idle
            agent.current_task = None
            if agent.status != SubagentStatus.ERROR:
                agent.status = SubagentStatus.IDLE
            agent.last_activity = datetime.now()

        # Try to execute next task
        self._try_execute_next()

    def get_status(self) -> Dict[str, Any]:
        """
        Get manager status.

        Returns:
            Dictionary with status information
        """
        with self._lock:
            idle_count = sum(1 for a in self.subagents.values() if a.status == SubagentStatus.IDLE)
            busy_count = sum(1 for a in self.subagents.values() if a.status == SubagentStatus.BUSY)
            error_count = sum(1 for a in self.subagents.values() if a.status == SubagentStatus.ERROR)

            return {
                "total_subagents": len(self.subagents),
                "idle_subagents": idle_count,
                "busy_subagents": busy_count,
                "error_subagents": error_count,
                "queued_tasks": self.task_queue.qsize(),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "is_shutdown": self._shutdown
            }

    def get_subagent_status(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific subagent.

        Args:
            agent_id: Subagent ID

        Returns:
            Subagent status dictionary or None
        """
        agent = self.subagents.get(agent_id)
        return agent.to_dict() if agent else None

    def pause_subagent(self, agent_id: int) -> bool:
        """
        Pause a subagent (it will finish current task but won't take new ones).

        Args:
            agent_id: Subagent ID

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            agent = self.subagents.get(agent_id)
            if agent and agent.status == SubagentStatus.IDLE:
                agent.status = SubagentStatus.PAUSED
                logger.info(f"Paused subagent {agent_id}")
                return True
            return False

    def resume_subagent(self, agent_id: int) -> bool:
        """
        Resume a paused subagent.

        Args:
            agent_id: Subagent ID

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            agent = self.subagents.get(agent_id)
            if agent and agent.status == SubagentStatus.PAUSED:
                agent.status = SubagentStatus.IDLE
                logger.info(f"Resumed subagent {agent_id}")
                # Try to execute queued tasks
                self._try_execute_next()
                return True
            return False

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout
        """
        import time
        start_time = time.time()

        while True:
            status = self.get_status()
            if status["queued_tasks"] == 0 and status["busy_subagents"] == 0:
                return True

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            time.sleep(0.1)

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get the result of a completed task.

        Args:
            task_id: Task ID

        Returns:
            Task result or None if not found/completed
        """
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task.result
        return None

    def get_task_error(self, task_id: str) -> Optional[Exception]:
        """
        Get the error of a failed task.

        Args:
            task_id: Task ID

        Returns:
            Task error or None if not found/failed
        """
        for task in self.failed_tasks:
            if task.task_id == task_id:
                return task.error
        return None

    def clear_completed(self) -> None:
        """Clear completed and failed task history."""
        with self._lock:
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            logger.info("Cleared completed and failed task history")

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Shutdown the subagent manager.

        Args:
            wait: Whether to wait for running tasks to complete
            timeout: Maximum time to wait in seconds
        """
        logger.info("Shutting down SubagentManager")
        self._shutdown = True

        # Mark all subagents as terminated
        with self._lock:
            for agent in self.subagents.values():
                if agent.status == SubagentStatus.IDLE:
                    agent.status = SubagentStatus.TERMINATED

        # Shutdown executor
        self.executor.shutdown(wait=wait, cancel_futures=not wait)
        logger.info("SubagentManager shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown(wait=True)
        return False
