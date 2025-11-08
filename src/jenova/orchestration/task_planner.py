"""
JENOVA Cognitive Architecture - Task Planner Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides task planning and decomposition with dependency graph management.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime
import json


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task in the plan."""

    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a single task in the plan."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "dependencies": list(self.dependencies),
            "metadata": self.metadata,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        task = cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 2)),
            dependencies=set(data.get("dependencies", [])),
            metadata=data.get("metadata", {}),
            result=data.get("result"),
            error=data.get("error"),
        )

        if data.get("created_at"):
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])

        return task


@dataclass
class TaskPlan:
    """Represents a complete task plan with dependency graph."""

    id: str
    description: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_task(self, task: Task) -> None:
        """Add a task to the plan."""
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute (dependencies met)."""
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                if all(
                    self.tasks.get(dep_id)
                    and self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                ):
                    ready.append(task)

        # Sort by priority (highest first)
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        return ready

    def get_blocked_tasks(self) -> List[Task]:
        """Get all tasks that are blocked by dependencies."""
        blocked = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if any dependency is not completed
                has_incomplete_deps = any(
                    self.tasks.get(dep_id)
                    and self.tasks[dep_id].status != TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if has_incomplete_deps:
                    blocked.append(task)
        return blocked

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for task in self.tasks.values()
        )

    def has_failures(self) -> bool:
        """Check if any tasks have failed."""
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary format."""
        return {
            "id": self.id,
            "description": self.description,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        """Create plan from dictionary."""
        plan = cls(
            id=data["id"],
            description=data["description"],
            metadata=data.get("metadata", {}),
        )

        if data.get("created_at"):
            plan.created_at = datetime.fromisoformat(data["created_at"])

        for task_data in data.get("tasks", {}).values():
            task = Task.from_dict(task_data)
            plan.add_task(task)

        return plan


class TaskPlanner:
    """
    Task planning and decomposition system with dependency graph management.

    This planner can:
    - Decompose complex tasks into subtasks
    - Manage task dependencies
    - Detect circular dependencies
    - Order tasks topologically
    - Track task execution progress
    """

    def __init__(self, llm_interface=None):
        """
        Initialize the task planner.

        Args:
            llm_interface: Optional LLM interface for AI-assisted planning
        """
        self.llm = llm_interface
        self.plans: Dict[str, TaskPlan] = {}
        self._task_counter = 0

    def plan_task(
        self, description: str, metadata: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        """
        Create a task plan from a description.

        Args:
            description: Description of the task to plan
            metadata: Optional metadata for the plan

        Returns:
            TaskPlan with decomposed tasks and dependencies
        """
        plan_id = f"plan_{self._generate_id()}"
        plan = TaskPlan(id=plan_id, description=description, metadata=metadata or {})

        # If LLM is available, use it for intelligent decomposition
        if self.llm:
            tasks = self._decompose_with_llm(description)
        else:
            # Fallback to heuristic decomposition
            tasks = self._decompose_heuristic(description)

        # Add tasks to plan
        for task_data in tasks:
            task = Task(
                id=f"task_{self._generate_id()}",
                description=task_data["description"],
                priority=task_data.get("priority", TaskPriority.MEDIUM),
                dependencies=set(task_data.get("dependencies", [])),
                metadata=task_data.get("metadata", {}),
            )
            plan.add_task(task)

        # Validate the plan
        self._validate_plan(plan)

        # Store the plan
        self.plans[plan_id] = plan

        logger.info(f"Created task plan '{plan_id}' with {len(plan.tasks)} tasks")
        return plan

    def _decompose_with_llm(self, description: str) -> List[Dict[str, Any]]:
        """
        Decompose task using LLM.

        Args:
            description: Task description

        Returns:
            List of task dictionaries
        """
        # This would use the LLM to intelligently break down tasks
        # For now, fall back to heuristic
        logger.info("LLM-based decomposition not yet implemented, using heuristic")
        return self._decompose_heuristic(description)

    def _decompose_heuristic(self, description: str) -> List[Dict[str, Any]]:
        """
        Decompose task using heuristic rules.

        Args:
            description: Task description

        Returns:
            List of task dictionaries
        """
        tasks = []
        description_lower = description.lower()

        # Pattern-based decomposition
        if "test" in description_lower or "testing" in description_lower:
            tasks.extend(
                [
                    {
                        "description": "Set up test environment",
                        "priority": TaskPriority.HIGH,
                    },
                    {
                        "description": "Write test cases",
                        "priority": TaskPriority.MEDIUM,
                    },
                    {"description": "Run tests", "priority": TaskPriority.MEDIUM},
                    {
                        "description": "Analyze test results",
                        "priority": TaskPriority.MEDIUM,
                    },
                ]
            )
        elif "deploy" in description_lower or "deployment" in description_lower:
            tasks.extend(
                [
                    {"description": "Build application", "priority": TaskPriority.HIGH},
                    {
                        "description": "Run pre-deployment tests",
                        "priority": TaskPriority.HIGH,
                    },
                    {
                        "description": "Deploy to staging",
                        "priority": TaskPriority.MEDIUM,
                    },
                    {
                        "description": "Verify staging deployment",
                        "priority": TaskPriority.MEDIUM,
                    },
                    {
                        "description": "Deploy to production",
                        "priority": TaskPriority.LOW,
                    },
                ]
            )
        elif "refactor" in description_lower or "restructure" in description_lower:
            tasks.extend(
                [
                    {
                        "description": "Analyze current code structure",
                        "priority": TaskPriority.HIGH,
                    },
                    {
                        "description": "Identify refactoring opportunities",
                        "priority": TaskPriority.MEDIUM,
                    },
                    {
                        "description": "Apply refactoring changes",
                        "priority": TaskPriority.MEDIUM,
                    },
                    {"description": "Update tests", "priority": TaskPriority.MEDIUM},
                    {
                        "description": "Verify functionality",
                        "priority": TaskPriority.HIGH,
                    },
                ]
            )
        else:
            # Generic decomposition
            tasks.append({"description": description, "priority": TaskPriority.MEDIUM})

        return tasks

    def add_task(
        self,
        plan_id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Add a task to an existing plan.

        Args:
            plan_id: ID of the plan to add task to
            description: Task description
            dependencies: List of task IDs this task depends on
            priority: Task priority
            metadata: Optional task metadata

        Returns:
            Task ID if successful, None otherwise
        """
        plan = self.plans.get(plan_id)
        if not plan:
            logger.error(f"Plan not found: {plan_id}")
            return None

        task = Task(
            id=f"task_{self._generate_id()}",
            description=description,
            priority=priority,
            dependencies=set(dependencies or []),
            metadata=metadata or {},
        )

        plan.add_task(task)
        logger.info(f"Added task '{task.id}' to plan '{plan_id}'")
        return task.id

    def add_dependency(self, plan_id: str, task_id: str, depends_on: str) -> bool:
        """
        Add a dependency between tasks.

        Args:
            plan_id: Plan ID
            task_id: Task that has the dependency
            depends_on: Task that must complete first

        Returns:
            True if successful, False otherwise
        """
        plan = self.plans.get(plan_id)
        if not plan:
            logger.error(f"Plan not found: {plan_id}")
            return False

        task = plan.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return False

        if depends_on not in plan.tasks:
            logger.error(f"Dependency task not found: {depends_on}")
            return False

        task.dependencies.add(depends_on)

        # Check for circular dependencies
        if self._has_circular_dependency(plan, task_id):
            logger.error(f"Circular dependency detected, removing dependency")
            task.dependencies.remove(depends_on)
            return False

        logger.info(f"Added dependency: {task_id} depends on {depends_on}")
        return True

    def _has_circular_dependency(self, plan: TaskPlan, start_task_id: str) -> bool:
        """
        Check if adding a dependency would create a circular dependency.

        Args:
            plan: The task plan
            start_task_id: Task ID to check from

        Returns:
            True if circular dependency exists, False otherwise
        """
        visited = set()
        stack = [start_task_id]

        while stack:
            task_id = stack.pop()

            if task_id in visited:
                # Found a cycle
                return True

            visited.add(task_id)

            task = plan.get_task(task_id)
            if task:
                stack.extend(task.dependencies)

        return False

    def topological_sort(self, plan_id: str) -> Optional[List[str]]:
        """
        Get tasks in topological order (dependencies first).

        Args:
            plan_id: Plan ID

        Returns:
            List of task IDs in execution order, or None if circular dependency
        """
        plan = self.plans.get(plan_id)
        if not plan:
            logger.error(f"Plan not found: {plan_id}")
            return None

        # Calculate in-degree for each task
        in_degree = {task_id: 0 for task_id in plan.tasks}
        for task in plan.tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Find tasks with no dependencies
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            task_id = queue.popleft()
            result.append(task_id)

            # Remove this task from dependency lists
            task = plan.get_task(task_id)
            if task:
                for other_task in plan.tasks.values():
                    if task_id in other_task.dependencies:
                        in_degree[other_task.id] -= 1
                        if in_degree[other_task.id] == 0:
                            queue.append(other_task.id)

        # If not all tasks processed, there's a circular dependency
        if len(result) != len(plan.tasks):
            logger.error(f"Circular dependency detected in plan {plan_id}")
            return None

        return result

    def get_execution_order(self, plan_id: str) -> List[List[str]]:
        """
        Get tasks grouped by execution level (parallel execution possible within level).

        Args:
            plan_id: Plan ID

        Returns:
            List of lists of task IDs, where each inner list can be executed in parallel
        """
        plan = self.plans.get(plan_id)
        if not plan:
            logger.error(f"Plan not found: {plan_id}")
            return []

        levels = []
        remaining_tasks = set(plan.tasks.keys())
        completed_tasks = set()

        while remaining_tasks:
            # Find tasks that can execute now (all dependencies met)
            current_level = []
            for task_id in remaining_tasks:
                task = plan.get_task(task_id)
                if task and all(dep in completed_tasks for dep in task.dependencies):
                    current_level.append(task_id)

            if not current_level:
                # Circular dependency or other issue
                logger.error(f"Unable to determine execution order for plan {plan_id}")
                break

            levels.append(current_level)
            completed_tasks.update(current_level)
            remaining_tasks -= set(current_level)

        return levels

    def _validate_plan(self, plan: TaskPlan) -> None:
        """
        Validate a task plan.

        Args:
            plan: Plan to validate

        Raises:
            ValueError: If plan is invalid
        """
        # Check all dependencies exist
        for task in plan.tasks.values():
            for dep in task.dependencies:
                if dep not in plan.tasks:
                    raise ValueError(
                        f"Task {task.id} depends on non-existent task {dep}"
                    )

        # Check for circular dependencies
        for task_id in plan.tasks:
            if self._has_circular_dependency(plan, task_id):
                raise ValueError(
                    f"Circular dependency detected starting from task {task_id}"
                )

    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Get a plan by ID."""
        return self.plans.get(plan_id)

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Get the status of a plan.

        Args:
            plan_id: Plan ID

        Returns:
            Dictionary with plan status information
        """
        plan = self.plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}

        total = len(plan.tasks)
        completed = sum(
            1 for t in plan.tasks.values() if t.status == TaskStatus.COMPLETED
        )
        failed = sum(1 for t in plan.tasks.values() if t.status == TaskStatus.FAILED)
        in_progress = sum(
            1 for t in plan.tasks.values() if t.status == TaskStatus.IN_PROGRESS
        )
        pending = sum(1 for t in plan.tasks.values() if t.status == TaskStatus.PENDING)

        return {
            "plan_id": plan_id,
            "description": plan.description,
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "progress": (completed / total * 100) if total > 0 else 0,
            "is_complete": plan.is_complete(),
            "has_failures": plan.has_failures(),
        }

    def save_plan(self, plan_id: str, filepath: str) -> bool:
        """
        Save a plan to a file.

        Args:
            plan_id: Plan ID
            filepath: Path to save file

        Returns:
            True if successful, False otherwise
        """
        plan = self.plans.get(plan_id)
        if not plan:
            logger.error(f"Plan not found: {plan_id}")
            return False

        try:
            with open(filepath, "w") as f:
                json.dump(plan.to_dict(), f, indent=2)
            logger.info(f"Saved plan to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving plan: {e}")
            return False

    def load_plan(self, filepath: str) -> Optional[str]:
        """
        Load a plan from a file.

        Args:
            filepath: Path to plan file

        Returns:
            Plan ID if successful, None otherwise
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            plan = TaskPlan.from_dict(data)
            self.plans[plan.id] = plan
            logger.info(f"Loaded plan {plan.id} from {filepath}")
            return plan.id
        except Exception as e:
            logger.error(f"Error loading plan: {e}")
            return None

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        self._task_counter += 1
        return f"{self._task_counter:06d}"
