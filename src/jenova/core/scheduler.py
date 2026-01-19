##Script function and purpose: Cognitive Scheduler - Turn-based background task scheduling for cognitive operations
##Dependency purpose: Schedules cognitive tasks (insights, assumptions, verification, reflection) based on conversation turn count and context
"""Cognitive Scheduler for JENOVA.

This module schedules cognitive tasks based on conversation context and
configurable intervals. Tasks are triggered based on turn count and can be
adjusted dynamically based on cognitive state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from jenova.config.models import JenovaConfig

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for graph operations needed by scheduler
class GraphProtocol(Protocol):
    """Protocol for graph operations."""
    
    ##Method purpose: Get nodes by type and username
    def get_nodes_by_user(self, username: str) -> list[object]:
        """Get all nodes for a user."""
        ...


##Class purpose: Protocol for insight manager operations
class InsightManagerProtocol(Protocol):
    """Protocol for insight manager operations."""
    
    ##Method purpose: Generate insight from history
    def generate_insight_from_history(self, username: str) -> None:
        """Generate insight from conversation history."""
        ...


##Class purpose: Protocol for assumption manager operations
class AssumptionManagerProtocol(Protocol):
    """Protocol for assumption manager operations."""
    
    ##Method purpose: Generate assumption from history
    def generate_assumption_from_history(self, username: str) -> None:
        """Generate assumption from conversation history."""
        ...
    
    ##Method purpose: Proactively verify assumptions
    def proactively_verify_assumption(self, username: str) -> None:
        """Proactively verify an assumption."""
        ...


##Class purpose: Configuration for cognitive scheduler
@dataclass
class SchedulerConfig:
    """Configuration for cognitive scheduler.
    
    Attributes:
        generate_insight_interval: Turn interval for insight generation (default: 5).
        generate_assumption_interval: Turn interval for assumption generation (default: 7).
        proactively_verify_assumption_interval: Turn interval for assumption verification (default: 8).
        reflect_interval: Turn interval for reflection (default: 10).
        enabled: Whether scheduler is enabled (default: True).
    """
    
    generate_insight_interval: int = 5
    generate_assumption_interval: int = 7
    proactively_verify_assumption_interval: int = 8
    reflect_interval: int = 10
    enabled: bool = True


##Class purpose: Task definition for scheduled cognitive operations
@dataclass
class ScheduledTask:
    """A scheduled cognitive task.
    
    Attributes:
        name: Task name/identifier.
        params: Task parameters as dict.
        turn_triggered: Turn count when task was triggered.
    """
    
    name: str
    params: dict[str, str] = field(default_factory=dict)
    turn_triggered: int = 0


##Class purpose: Schedules cognitive functions based on turn count and context
class CognitiveScheduler:
    """Schedules cognitive functions based on conversation context.
    
    The scheduler determines which cognitive tasks should run based on:
    - Turn count intervals
    - Cognitive state (e.g., number of unverified assumptions)
    - Configurable intervals
    
    Attributes:
        config: Scheduler configuration.
        graph: Graph protocol for state queries.
        insight_manager: Insight manager protocol.
        assumption_manager: Assumption manager protocol.
        last_execution_times: Track when tasks last ran.
    """
    
    ##Method purpose: Initialize scheduler with configuration and cognitive components
    def __init__(
        self,
        config: SchedulerConfig,
        graph: GraphProtocol,
        insight_manager: InsightManagerProtocol | None = None,
        assumption_manager: AssumptionManagerProtocol | None = None,
    ) -> None:
        """Initialize the cognitive scheduler.
        
        Args:
            config: Scheduler configuration.
            graph: Graph for state queries.
            insight_manager: Optional insight manager.
            assumption_manager: Optional assumption manager.
        """
        ##Step purpose: Store configuration and dependencies
        self.config = config
        self.graph = graph
        self.insight_manager = insight_manager
        self.assumption_manager = assumption_manager
        
        ##Step purpose: Track last execution times
        self.last_execution_times: dict[str, datetime | None] = {
            "generate_insight": None,
            "generate_assumption": None,
            "proactively_verify_assumption": None,
            "reflect": None,
        }
        
        ##Action purpose: Log initialization
        logger.info(
            "cognitive_scheduler_initialized",
            enabled=self.config.enabled,
            insight_interval=self.config.generate_insight_interval,
            assumption_interval=self.config.generate_assumption_interval,
        )
    
    ##Method purpose: Determine which cognitive tasks should run based on context
    def get_cognitive_tasks(
        self,
        turn_count: int,
        user_input: str,
        username: str,
    ) -> list[ScheduledTask]:
        """Determine which cognitive tasks to run based on current context.
        
        Args:
            turn_count: Current conversation turn count.
            user_input: Current user input (for context).
            username: Username for user-specific operations.
            
        Returns:
            List of scheduled tasks to execute.
        """
        ##Condition purpose: Return empty if scheduler disabled
        if not self.config.enabled:
            return []
        
        tasks: list[ScheduledTask] = []
        
        ##Step purpose: Get intervals from config
        generate_insight_interval = self.config.generate_insight_interval
        generate_assumption_interval = self.config.generate_assumption_interval
        proactively_verify_assumption_interval = self.config.proactively_verify_assumption_interval
        reflect_interval = self.config.reflect_interval
        
        ##Step purpose: Context-aware adjustments
        ##Condition purpose: Adjust verification interval if many unverified assumptions
        unverified_count = self._count_unverified_assumptions(username)
        if unverified_count > 5:
            proactively_verify_assumption_interval = 3
            logger.debug(
                "scheduler_adjusted_verification_interval",
                unverified_count=unverified_count,
                new_interval=proactively_verify_assumption_interval,
            )
        
        ##Condition purpose: Schedule insight generation if interval reached
        if self._should_run("generate_insight", turn_count, generate_insight_interval):
            tasks.append(ScheduledTask(
                name="generate_insight_from_history",
                params={"username": username},
                turn_triggered=turn_count,
            ))
        
        ##Condition purpose: Schedule assumption generation if interval reached
        if self._should_run("generate_assumption", turn_count, generate_assumption_interval):
            tasks.append(ScheduledTask(
                name="generate_assumption_from_history",
                params={"username": username},
                turn_triggered=turn_count,
            ))
        
        ##Condition purpose: Schedule assumption verification if interval reached
        if self._should_run("proactively_verify_assumption", turn_count, proactively_verify_assumption_interval):
            tasks.append(ScheduledTask(
                name="proactively_verify_assumption",
                params={"username": username},
                turn_triggered=turn_count,
            ))
        
        ##Condition purpose: Schedule reflection if interval reached
        if self._should_run("reflect", turn_count, reflect_interval):
            tasks.append(ScheduledTask(
                name="reflect",
                params={"username": username},
                turn_triggered=turn_count,
            ))
        
        ##Action purpose: Log scheduled tasks
        if tasks:
            logger.debug(
                "scheduler_tasks_scheduled",
                turn_count=turn_count,
                task_count=len(tasks),
                task_names=[t.name for t in tasks],
            )
        
        return tasks
    
    ##Method purpose: Check if a task should run based on turn count and interval
    def _should_run(self, task_name: str, turn_count: int, interval: int) -> bool:
        """Check if a task should be run based on turn count and interval.
        
        Args:
            task_name: Name of the task.
            turn_count: Current turn count.
            interval: Interval in turns.
            
        Returns:
            True if task should run, False otherwise.
        """
        ##Condition purpose: Check if turn count matches interval
        if turn_count > 0 and turn_count % interval == 0:
            self.last_execution_times[task_name] = datetime.now()
            return True
        return False
    
    ##Method purpose: Count unverified assumptions for a user
    def _count_unverified_assumptions(self, username: str) -> int:
        """Count unverified assumptions for a user.
        
        Args:
            username: Username to count for.
            
        Returns:
            Count of unverified assumptions.
        """
        ##Error purpose: Handle errors gracefully
        try:
            ##Step purpose: Get all user nodes
            user_nodes = self.graph.get_nodes_by_user(username)
            
            ##Step purpose: Count assumption nodes with unverified status
            count = 0
            for node in user_nodes:
                ##Condition purpose: Check if node is an unverified assumption
                node_type = getattr(node, "node_type", "")
                metadata = getattr(node, "metadata", {})
                if node_type == "assumption" and metadata.get("status") == "unverified":
                    count += 1
            
            return count
        except Exception as e:
            logger.warning("scheduler_count_assumptions_failed", error=str(e), username=username)
            return 0
    
    ##Method purpose: Execute a scheduled task
    def execute_task(self, task: ScheduledTask) -> bool:
        """Execute a scheduled cognitive task.
        
        Args:
            task: Scheduled task to execute.
            
        Returns:
            True if task executed successfully, False otherwise.
        """
        ##Error purpose: Handle task execution errors
        try:
            ##Condition purpose: Route to appropriate handler
            if task.name == "generate_insight_from_history":
                if self.insight_manager:
                    self.insight_manager.generate_insight_from_history(task.params["username"])
                    return True
                logger.warning("scheduler_insight_manager_unavailable", task=task.name)
                return False
            
            elif task.name == "generate_assumption_from_history":
                if self.assumption_manager:
                    self.assumption_manager.generate_assumption_from_history(task.params["username"])
                    return True
                logger.warning("scheduler_assumption_manager_unavailable", task=task.name)
                return False
            
            elif task.name == "proactively_verify_assumption":
                if self.assumption_manager:
                    self.assumption_manager.proactively_verify_assumption(task.params["username"])
                    return True
                logger.warning("scheduler_assumption_manager_unavailable", task=task.name)
                return False
            
            elif task.name == "reflect":
                ##Step purpose: Reflection is a placeholder for future implementation
                logger.debug("scheduler_reflection_placeholder", username=task.params.get("username"))
                return True
            
            else:
                logger.warning("scheduler_unknown_task", task=task.name)
                return False
                
        except Exception as e:
            logger.error("scheduler_task_execution_failed", task=task.name, error=str(e))
            return False
