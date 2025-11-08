"""
JENOVA Cognitive Architecture - Execution Engine Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides task execution with comprehensive error handling and recovery.
"""

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of execution."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of task execution."""

    task_id: str
    status: ExecutionStatus
    result: Optional[Any] = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": str(self.error) if self.error else None,
            "error_traceback": self.error_traceback,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration(),
            "retry_count": self.retry_count,
            "metadata": self.metadata
        }


class ExecutionEngine:
    """
    Task execution engine with error handling and recovery.

    Features:
    - Automatic retry with exponential backoff
    - Error categorization and handling
    - Execution history tracking
    - Resource cleanup
    - Pause/resume capability
    """

    def __init__(self, max_retries: int = 3, enable_retry: bool = True):
        """
        Initialize the execution engine.

        Args:
            max_retries: Maximum number of retry attempts
            enable_retry: Whether to enable automatic retries
        """
        self.max_retries = max_retries
        self.enable_retry = enable_retry
        self.running_tasks: Dict[str, ExecutionResult] = {}
        self.execution_history: List[ExecutionResult] = []
        self._paused = False

    def execute(
        self,
        task_callable: Callable,
        *args,
        task_id: Optional[str] = None,
        retryable_exceptions: Optional[List[type]] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a task with error handling.

        Args:
            task_callable: Function to execute
            *args: Positional arguments
            task_id: Optional task ID
            retryable_exceptions: Exceptions that should trigger retry
            **kwargs: Keyword arguments

        Returns:
            ExecutionResult
        """
        if task_id is None:
            task_id = f"task_{datetime.now().timestamp()}"

        result = ExecutionResult(
            task_id=task_id,
            status=ExecutionStatus.NOT_STARTED,
            start_time=datetime.now()
        )

        self.running_tasks[task_id] = result

        try:
            # Check if paused
            if self._paused:
                result.status = ExecutionStatus.PAUSED
                logger.info(f"Execution paused for task {task_id}")
                return result

            result.status = ExecutionStatus.RUNNING
            logger.info(f"Executing task {task_id}")

            # Execute with or without retry
            if self.enable_retry and retryable_exceptions:
                output = self._execute_with_retry(
                    task_callable,
                    args,
                    kwargs,
                    retryable_exceptions,
                    result
                )
            else:
                output = task_callable(*args, **kwargs)

            result.result = output
            result.status = ExecutionStatus.COMPLETED
            result.end_time = datetime.now()

            logger.info(f"Task {task_id} completed successfully in {result.duration():.2f}s")

        except Exception as e:
            result.error = e
            result.error_traceback = traceback.format_exc()
            result.status = ExecutionStatus.FAILED
            result.end_time = datetime.now()

            logger.error(f"Task {task_id} failed: {e}")
            logger.debug(f"Traceback:\n{result.error_traceback}")

        finally:
            # Move to history
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            self.execution_history.append(result)

        return result

    def _execute_with_retry(
        self,
        task_callable: Callable,
        args: tuple,
        kwargs: Dict[str, Any],
        retryable_exceptions: List[type],
        result: ExecutionResult
    ) -> Any:
        """
        Execute task with automatic retry.

        Args:
            task_callable: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            retryable_exceptions: Exceptions that trigger retry
            result: ExecutionResult to update

        Returns:
            Task result
        """
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(tuple(retryable_exceptions)),
            reraise=True
        )
        def _retry_wrapper():
            result.retry_count += 1
            if result.retry_count > 1:
                logger.info(f"Retry attempt {result.retry_count} for task {result.task_id}")
            return task_callable(*args, **kwargs)

        return _retry_wrapper()

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, ExecutionResult]:
        """
        Execute a task plan.

        Args:
            plan: Task plan dictionary with 'steps' and 'dependencies'

        Returns:
            Dictionary mapping task IDs to ExecutionResults
        """
        steps = plan.get("steps", [])
        dependencies = plan.get("dependencies", {})
        results = {}

        logger.info(f"Executing plan with {len(steps)} steps")

        # Execute steps
        for step in steps:
            task_id = step.get("id", f"step_{len(results)}")
            callable_func = step.get("callable")
            args = step.get("args", ())
            kwargs = step.get("kwargs", {})

            if not callable_func:
                logger.error(f"No callable provided for step {task_id}")
                continue

            # Check dependencies
            if task_id in dependencies:
                for dep_id in dependencies[task_id]:
                    dep_result = results.get(dep_id)
                    if not dep_result or dep_result.status != ExecutionStatus.COMPLETED:
                        logger.error(f"Dependency {dep_id} not completed for {task_id}")
                        results[task_id] = ExecutionResult(
                            task_id=task_id,
                            status=ExecutionStatus.FAILED,
                            error=Exception(f"Dependency {dep_id} failed")
                        )
                        continue

            # Execute step
            result = self.execute(callable_func, *args, task_id=task_id, **kwargs)
            results[task_id] = result

            # Stop on failure if required
            if result.status == ExecutionStatus.FAILED and plan.get("stop_on_failure", True):
                logger.error(f"Stopping plan execution due to failure in {task_id}")
                break

        return results

    def pause(self) -> None:
        """Pause execution of new tasks."""
        self._paused = True
        logger.info("Execution engine paused")

    def resume(self) -> None:
        """Resume execution of tasks."""
        self._paused = False
        logger.info("Execution engine resumed")

    def cancel_task(self, task_id: str) -> bool:
        """
        Attempt to cancel a running task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id in self.running_tasks:
            result = self.running_tasks[task_id]
            result.status = ExecutionStatus.CANCELLED
            result.end_time = datetime.now()
            logger.info(f"Cancelled task {task_id}")
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[ExecutionStatus]:
        """
        Get status of a task.

        Args:
            task_id: Task ID

        Returns:
            ExecutionStatus or None if not found
        """
        if task_id in self.running_tasks:
            return self.running_tasks[task_id].status

        for result in reversed(self.execution_history):
            if result.task_id == task_id:
                return result.status

        return None

    def get_task_result(self, task_id: str) -> Optional[ExecutionResult]:
        """
        Get result of a task.

        Args:
            task_id: Task ID

        Returns:
            ExecutionResult or None if not found
        """
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]

        for result in reversed(self.execution_history):
            if result.task_id == task_id:
                return result

        return None

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of executions.

        Returns:
            Dictionary with execution statistics
        """
        total = len(self.execution_history)
        completed = sum(1 for r in self.execution_history if r.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for r in self.execution_history if r.status == ExecutionStatus.FAILED)
        cancelled = sum(1 for r in self.execution_history if r.status == ExecutionStatus.CANCELLED)

        return {
            "total_executions": total,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "running": len(self.running_tasks),
            "success_rate": (completed / total * 100) if total > 0 else 0,
            "is_paused": self._paused
        }

    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        logger.info("Cleared execution history")

    def get_failed_tasks(self) -> List[ExecutionResult]:
        """Get all failed task results."""
        return [r for r in self.execution_history if r.status == ExecutionStatus.FAILED]

    def get_completed_tasks(self) -> List[ExecutionResult]:
        """Get all completed task results."""
        return [r for r in self.execution_history if r.status == ExecutionStatus.COMPLETED]
