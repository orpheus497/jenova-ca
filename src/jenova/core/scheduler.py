##Script function and purpose: Cognitive scheduler for turn-based background task execution
"""
Cognitive Scheduler - Turn-based scheduling of cognitive tasks.

This module provides scheduling infrastructure for background cognitive tasks
that run during conversation intervals. Tasks include insight generation,
assumption verification, and reflection cycles.

Reference: .devdocs/resources/src/jenova/cognitive_engine/scheduler.py
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, Callable, Sequence

import structlog

from jenova.exceptions import SchedulerError

##Class purpose: Define logger for scheduler operations
logger = structlog.get_logger(__name__)


##Class purpose: Enumerate types of cognitive tasks that can be scheduled
class TaskType(Enum):
    """Types of cognitive tasks that can be scheduled."""
    GENERATE_INSIGHT = auto()
    GENERATE_ASSUMPTION = auto()
    VERIFY_ASSUMPTION = auto()
    REFLECT = auto()
    PRUNE_GRAPH = auto()
    LINK_ORPHANS = auto()


##Class purpose: Configuration for task scheduling intervals
@dataclass(frozen=True)
class TaskSchedule:
    """Configuration for a scheduled task.
    
    Attributes:
        task_type: The type of task to schedule
        interval: Number of turns between task executions
        priority: Higher priority tasks run first (default 0)
        enabled: Whether this task is enabled
    """
    task_type: TaskType
    interval: int
    priority: int = 0
    enabled: bool = True


##Class purpose: Track state of a scheduled task
@dataclass
class TaskState:
    """Runtime state for a scheduled task.
    
    Attributes:
        schedule: The task schedule configuration
        turns_since_last: Number of turns since last execution
        execution_count: Total number of times task has executed
        last_error: Last error message if task failed
    """
    schedule: TaskSchedule
    turns_since_last: int = 0
    execution_count: int = 0
    last_error: str | None = None


##Class purpose: Protocol for task executors
class TaskExecutorProtocol(Protocol):
    """Protocol for cognitive task executors."""
    
    ##Method purpose: Execute a cognitive task
    def execute_task(self, task_type: TaskType, username: str) -> bool:
        """Execute a cognitive task.
        
        Args:
            task_type: The type of task to execute
            username: The user context for the task
            
        Returns:
            True if task completed successfully, False otherwise
        """
        ...


##Class purpose: Configuration for the cognitive scheduler
@dataclass
class SchedulerConfig:
    """Configuration for the CognitiveScheduler.
    
    Attributes:
        insight_interval: Turns between insight generation (default 5)
        assumption_interval: Turns between assumption generation (default 7)
        verify_interval: Turns between assumption verification (default 3)
        reflect_interval: Turns between reflection cycles (default 10)
        prune_interval: Turns between graph pruning (default 20)
        link_orphans_interval: Turns between orphan linking (default 15)
        accelerate_verification: Speed up verification if many unverified (default True)
        unverified_threshold: Threshold for acceleration (default 5)
    """
    insight_interval: int = 5
    assumption_interval: int = 7
    verify_interval: int = 3
    reflect_interval: int = 10
    prune_interval: int = 20
    link_orphans_interval: int = 15
    accelerate_verification: bool = True
    unverified_threshold: int = 5


##Class purpose: Main scheduler for cognitive background tasks
class CognitiveScheduler:
    """Scheduler for turn-based cognitive background tasks.
    
    The scheduler tracks conversation turns and executes cognitive tasks
    at configured intervals. Tasks are prioritized and executed in order.
    
    Example:
        >>> scheduler = CognitiveScheduler(config, executor)
        >>> scheduler.on_turn_complete(username="user1")
        >>> # Tasks are automatically executed based on turn count
    """
    
    ##Method purpose: Initialize scheduler with configuration
    def __init__(
        self,
        config: SchedulerConfig,
        executor: TaskExecutorProtocol | None = None,
    ) -> None:
        """Initialize the cognitive scheduler.
        
        Args:
            config: Scheduler configuration
            executor: Optional task executor (can be set later)
        """
        ##Step purpose: Store configuration
        self._config = config
        self._executor = executor
        self._turn_count = 0
        
        ##Step purpose: Initialize task states from config
        self._tasks: list[TaskState] = self._create_task_states()
        
        logger.info(
            "scheduler_initialized",
            task_count=len(self._tasks),
            enabled_tasks=[t.schedule.task_type.name for t in self._tasks if t.schedule.enabled],
        )
    
    ##Method purpose: Create task states from configuration
    def _create_task_states(self) -> list[TaskState]:
        """Create task states from configuration.
        
        Returns:
            List of TaskState instances initialized from scheduler configuration
        """
        schedules = [
            TaskSchedule(TaskType.GENERATE_INSIGHT, self._config.insight_interval, priority=2),
            TaskSchedule(TaskType.GENERATE_ASSUMPTION, self._config.assumption_interval, priority=1),
            TaskSchedule(TaskType.VERIFY_ASSUMPTION, self._config.verify_interval, priority=3),
            TaskSchedule(TaskType.REFLECT, self._config.reflect_interval, priority=0),
            TaskSchedule(TaskType.PRUNE_GRAPH, self._config.prune_interval, priority=-1),
            TaskSchedule(TaskType.LINK_ORPHANS, self._config.link_orphans_interval, priority=-1),
        ]
        return [TaskState(schedule=s) for s in schedules]
    
    ##Method purpose: Set the task executor
    def set_executor(self, executor: TaskExecutorProtocol) -> None:
        """Set the task executor.
        
        Args:
            executor: The task executor to use
        """
        self._executor = executor
        logger.debug("scheduler_executor_set")
    
    ##Method purpose: Called when a conversation turn completes
    def on_turn_complete(
        self,
        username: str,
        unverified_count: int = 0,
    ) -> list[TaskType]:
        """Called when a conversation turn completes.
        
        Increments turn counter and executes any due tasks.
        
        Args:
            username: The user context for task execution
            unverified_count: Count of unverified assumptions (for acceleration)
            
        Returns:
            List of task types that were executed
        """
        ##Step purpose: Increment turn counter
        self._turn_count += 1
        
        ##Step purpose: Increment turns since last for all tasks
        for task in self._tasks:
            task.turns_since_last += 1
        
        ##Step purpose: Get and execute due tasks
        due_tasks = self._get_due_tasks(unverified_count)
        executed: list[TaskType] = []
        
        ##Loop purpose: Execute each due task
        for task in due_tasks:
            ##Condition purpose: Check if executor is available
            if self._executor is None:
                logger.warning(
                    "scheduler_no_executor",
                    task_type=task.schedule.task_type.name,
                )
                continue
            
            ##Error purpose: Handle task execution errors
            try:
                success = self._executor.execute_task(
                    task.schedule.task_type,
                    username,
                )
                
                ##Condition purpose: Update task state based on success
                if success:
                    task.turns_since_last = 0
                    task.execution_count += 1
                    task.last_error = None
                    executed.append(task.schedule.task_type)
                    logger.debug(
                        "scheduler_task_executed",
                        task_type=task.schedule.task_type.name,
                        execution_count=task.execution_count,
                    )
                else:
                    task.last_error = "Task returned False"
                    
            except Exception as e:
                task.last_error = str(e)
                logger.error(
                    "scheduler_task_failed",
                    task_type=task.schedule.task_type.name,
                    error=str(e),
                )
        
        return executed
    
    ##Method purpose: Get tasks that are due for execution
    def _get_due_tasks(self, unverified_count: int) -> list[TaskState]:
        """Get tasks that are due for execution.
        
        Args:
            unverified_count: Count of unverified assumptions
            
        Returns:
            List of due task states, sorted by priority
        """
        due: list[TaskState] = []
        
        ##Loop purpose: Check each task for due status
        for task in self._tasks:
            ##Condition purpose: Skip disabled tasks
            if not task.schedule.enabled:
                continue
            
            ##Step purpose: Calculate effective interval
            interval = task.schedule.interval
            
            ##Condition purpose: Accelerate verification if many unverified
            if (
                task.schedule.task_type == TaskType.VERIFY_ASSUMPTION
                and self._config.accelerate_verification
                and unverified_count >= self._config.unverified_threshold
            ):
                interval = max(1, interval // 2)
            
            ##Condition purpose: Check if task is due
            if task.turns_since_last >= interval:
                due.append(task)
        
        ##Step purpose: Sort by priority (higher first)
        due.sort(key=lambda t: t.schedule.priority, reverse=True)
        
        return due
    
    ##Method purpose: Get current turn count
    @property
    def turn_count(self) -> int:
        """Get the current turn count.
        
        Returns:
            Current number of conversation turns completed
        """
        return self._turn_count
    
    ##Method purpose: Reset the scheduler state
    def reset(self) -> None:
        """Reset the scheduler to initial state."""
        self._turn_count = 0
        for task in self._tasks:
            task.turns_since_last = 0
            task.execution_count = 0
            task.last_error = None
        logger.info("scheduler_reset")
    
    ##Method purpose: Get scheduler status for diagnostics
    def get_status(self) -> dict[str, object]:
        """Get scheduler status for diagnostics.
        
        Returns:
            Dictionary with scheduler state information
        """
        return {
            "turn_count": self._turn_count,
            "executor_set": self._executor is not None,
            "tasks": [
                {
                    "type": t.schedule.task_type.name,
                    "interval": t.schedule.interval,
                    "turns_since_last": t.turns_since_last,
                    "execution_count": t.execution_count,
                    "enabled": t.schedule.enabled,
                    "last_error": t.last_error,
                }
                for t in self._tasks
            ],
        }
