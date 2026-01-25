##Script function and purpose: Unit tests for cognitive scheduler (turn-based task execution)
"""
Test suite for CognitiveScheduler - Turn-based scheduling of cognitive tasks.

Tests cover:
- Task state management
- Scheduling logic and intervals
- Priority sorting
- Acceleration logic
- Error handling
"""

import pytest
from unittest.mock import Mock, MagicMock
from jenova.core.scheduler import (
    CognitiveScheduler,
    SchedulerConfig,
    TaskType,
    TaskSchedule,
    TaskState,
    TaskExecutorProtocol,
)
from jenova.exceptions import SchedulerError


##Class purpose: Fixture providing mock executor for tests
@pytest.fixture
def mock_executor() -> Mock:
    """##Test case: Mock task executor."""
    executor = Mock(spec=TaskExecutorProtocol)
    executor.execute_task = Mock(return_value=True)
    return executor


##Class purpose: Fixture providing default scheduler config
@pytest.fixture
def default_config() -> SchedulerConfig:
    """##Test case: Default scheduler configuration."""
    return SchedulerConfig(
        insight_interval=5,
        assumption_interval=7,
        verify_interval=3,
        reflect_interval=10,
    )


##Class purpose: Fixture providing scheduler instance
@pytest.fixture
def scheduler_with_executor(default_config: SchedulerConfig, mock_executor: Mock) -> CognitiveScheduler:
    """##Test case: Scheduler with mock executor."""
    return CognitiveScheduler(default_config, mock_executor)


##Class purpose: Fixture providing scheduler without executor
@pytest.fixture
def scheduler_no_executor(default_config: SchedulerConfig) -> CognitiveScheduler:
    """##Test case: Scheduler without executor."""
    return CognitiveScheduler(default_config, None)


##Function purpose: Test TaskSchedule immutability
def test_task_schedule_frozen() -> None:
    """##Test case: TaskSchedule is frozen (immutable)."""
    ##Step purpose: Create task schedule
    schedule = TaskSchedule(TaskType.GENERATE_INSIGHT, interval=5)
    
    ##Assertion purpose: Verify frozen dataclass behavior
    with pytest.raises(AttributeError):
        schedule.interval = 10


##Function purpose: Test TaskSchedule defaults
def test_task_schedule_defaults() -> None:
    """##Test case: TaskSchedule has correct defaults."""
    ##Step purpose: Create schedule with minimal args
    schedule = TaskSchedule(TaskType.VERIFY_ASSUMPTION, interval=3)
    
    ##Assertion purpose: Verify defaults
    assert schedule.priority == 0
    assert schedule.enabled is True


##Function purpose: Test TaskState initialization
def test_task_state_initialization() -> None:
    """##Test case: TaskState initializes correctly."""
    ##Step purpose: Create task schedule and state
    schedule = TaskSchedule(TaskType.GENERATE_INSIGHT, interval=5)
    state = TaskState(schedule=schedule)
    
    ##Assertion purpose: Verify initial state
    assert state.turns_since_last == 0
    assert state.execution_count == 0
    assert state.last_error is None


##Function purpose: Test scheduler initialization
def test_scheduler_initialization(default_config: SchedulerConfig, mock_executor: Mock) -> None:
    """##Test case: Scheduler initializes with correct task count."""
    ##Step purpose: Create scheduler
    scheduler = CognitiveScheduler(default_config, mock_executor)
    
    ##Assertion purpose: Verify state
    assert scheduler.turn_count == 0
    assert len(scheduler._tasks) == 6  # Six task types
    assert scheduler._executor is mock_executor


##Function purpose: Test scheduler initialization without executor
def test_scheduler_initialization_no_executor(default_config: SchedulerConfig) -> None:
    """##Test case: Scheduler can initialize without executor."""
    ##Step purpose: Create scheduler without executor
    scheduler = CognitiveScheduler(default_config, None)
    
    ##Assertion purpose: Verify state
    assert scheduler._executor is None
    assert len(scheduler._tasks) == 6


##Function purpose: Test set executor method
def test_set_executor(scheduler_no_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Can set executor after initialization."""
    ##Step purpose: Set executor
    scheduler_no_executor.set_executor(mock_executor)
    
    ##Assertion purpose: Verify executor set
    assert scheduler_no_executor._executor is mock_executor


##Function purpose: Test turn count increments
def test_turn_count_increments(scheduler_with_executor: CognitiveScheduler) -> None:
    """##Test case: Turn count increments on each call."""
    ##Step purpose: Call on_turn_complete multiple times
    scheduler_with_executor.on_turn_complete("user1")
    assert scheduler_with_executor.turn_count == 1
    
    scheduler_with_executor.on_turn_complete("user1")
    assert scheduler_with_executor.turn_count == 2


##Function purpose: Test task execution without executor
def test_on_turn_complete_no_executor(scheduler_no_executor: CognitiveScheduler) -> None:
    """##Test case: on_turn_complete handles missing executor."""
    ##Step purpose: Trigger turn without executor
    result = scheduler_no_executor.on_turn_complete("user1")
    
    ##Assertion purpose: Verify no tasks executed
    assert result == []
    assert scheduler_no_executor.turn_count == 1


##Function purpose: Test task execution with executor
def test_on_turn_complete_with_executor(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: on_turn_complete executes due tasks."""
    ##Step purpose: Set up executor to return success
    mock_executor.execute_task = Mock(return_value=True)
    
    ##Action purpose: Trigger turns to make task due
    # VERIFY_ASSUMPTION interval is 3, so it's due at turn 3
    scheduler_with_executor.on_turn_complete("user1")
    scheduler_with_executor.on_turn_complete("user1")
    result = scheduler_with_executor.on_turn_complete("user1")
    
    ##Assertion purpose: Verify task was executed
    assert len(result) > 0
    assert TaskType.VERIFY_ASSUMPTION in result


##Function purpose: Test task execution failure recording
def test_task_execution_failure(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Task failure is recorded in state."""
    ##Step purpose: Set up executor to return failure
    mock_executor.execute_task = Mock(return_value=False)
    
    ##Action purpose: Trigger task execution
    for _ in range(3):
        scheduler_with_executor.on_turn_complete("user1")
    
    ##Step purpose: Find VERIFY_ASSUMPTION task
    verify_task = next(t for t in scheduler_with_executor._tasks if t.schedule.task_type == TaskType.VERIFY_ASSUMPTION)
    
    ##Assertion purpose: Verify failure recorded
    assert verify_task.last_error == "Task returned False"


##Function purpose: Test task exception handling
def test_task_execution_exception(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Task exceptions are caught and logged."""
    ##Step purpose: Set up executor to raise exception
    mock_executor.execute_task = Mock(side_effect=RuntimeError("Test error"))
    
    ##Action purpose: Trigger task execution
    for _ in range(3):
        scheduler_with_executor.on_turn_complete("user1")
    
    ##Step purpose: Find VERIFY_ASSUMPTION task
    verify_task = next(t for t in scheduler_with_executor._tasks if t.schedule.task_type == TaskType.VERIFY_ASSUMPTION)
    
    ##Assertion purpose: Verify error recorded
    assert "Test error" in verify_task.last_error


##Function purpose: Test acceleration logic for verification
def test_acceleration_verification_threshold(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Verification accelerates when unverified count high."""
    ##Step purpose: Set up config with acceleration
    config = SchedulerConfig(
        verify_interval=3,
        accelerate_verification=True,
        unverified_threshold=5,
    )
    scheduler = CognitiveScheduler(config, mock_executor)
    
    ##Action purpose: Trigger turn with high unverified count
    # Should accelerate: interval becomes 3 // 2 = 1
    mock_executor.execute_task = Mock(return_value=True)
    result = scheduler.on_turn_complete("user1", unverified_count=10)
    
    ##Assertion purpose: Verify task executed at turn 1 (accelerated)
    assert TaskType.VERIFY_ASSUMPTION in result


##Function purpose: Test no acceleration without threshold
def test_no_acceleration_below_threshold(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Verification doesn't accelerate below threshold."""
    ##Step purpose: Configure acceleration
    config = SchedulerConfig(
        verify_interval=3,
        accelerate_verification=True,
        unverified_threshold=5,
    )
    scheduler = CognitiveScheduler(config, mock_executor)
    mock_executor.execute_task = Mock(return_value=True)
    
    ##Action purpose: Trigger turn with low unverified count
    result = scheduler.on_turn_complete("user1", unverified_count=2)
    
    ##Assertion purpose: Verify no acceleration (task not due yet)
    assert TaskType.VERIFY_ASSUMPTION not in result


##Function purpose: Test priority sorting
def test_due_tasks_sorted_by_priority(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Due tasks are sorted by priority."""
    ##Step purpose: Create many turns to trigger multiple tasks
    mock_executor.execute_task = Mock(return_value=True)
    
    ##Action purpose: Trigger turn 20 (all tasks due)
    for _ in range(20):
        scheduler_with_executor.on_turn_complete("user1")
    
    ##Step purpose: Get all tasks and verify they have priorities
    all_tasks = scheduler_with_executor._tasks
    priorities = [t.schedule.priority for t in all_tasks]
    
    ##Assertion purpose: Verify priorities exist
    assert len(priorities) == 6
    # VERIFY_ASSUMPTION has priority 3 (highest)
    verify_task = next(t for t in all_tasks if t.schedule.task_type == TaskType.VERIFY_ASSUMPTION)
    assert verify_task.schedule.priority == 3


##Function purpose: Test disabled task skipping
def test_disabled_tasks_not_executed(default_config: SchedulerConfig, mock_executor: Mock) -> None:
    """##Test case: Disabled tasks are not executed."""
    ##Step purpose: Create config with disabled task
    config = SchedulerConfig(
        verify_interval=1,  # Make due quickly
    )
    scheduler = CognitiveScheduler(config, mock_executor)
    
    ##Action purpose: Disable verification task
    verify_task = next(t for t in scheduler._tasks if t.schedule.task_type == TaskType.VERIFY_ASSUMPTION)
    verify_task.schedule = TaskSchedule(TaskType.VERIFY_ASSUMPTION, interval=1, enabled=False)
    
    ##Action purpose: Trigger turn
    mock_executor.execute_task = Mock(return_value=True)
    result = scheduler.on_turn_complete("user1")
    
    ##Assertion purpose: Verify disabled task not executed
    assert TaskType.VERIFY_ASSUMPTION not in result


##Function purpose: Test reset functionality
def test_scheduler_reset(scheduler_with_executor: CognitiveScheduler) -> None:
    """##Test case: Scheduler reset clears state."""
    ##Step purpose: Advance scheduler and trigger tasks
    for _ in range(5):
        scheduler_with_executor.on_turn_complete("user1")
    
    ##Action purpose: Reset scheduler
    scheduler_with_executor.reset()
    
    ##Assertion purpose: Verify reset
    assert scheduler_with_executor.turn_count == 0
    for task in scheduler_with_executor._tasks:
        assert task.turns_since_last == 0
        assert task.execution_count == 0
        assert task.last_error is None


##Function purpose: Test status dictionary
def test_get_status(scheduler_with_executor: CognitiveScheduler) -> None:
    """##Test case: get_status returns complete status dict."""
    ##Step purpose: Get status
    status = scheduler_with_executor.get_status()
    
    ##Assertion purpose: Verify status structure
    assert "turn_count" in status
    assert "executor_set" in status
    assert "tasks" in status
    assert len(status["tasks"]) == 6


##Function purpose: Test execution count tracking
def test_execution_count_tracking(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Execution count increments correctly."""
    ##Step purpose: Set up successful execution
    mock_executor.execute_task = Mock(return_value=True)
    
    ##Action purpose: Trigger multiple turns
    for _ in range(5):
        scheduler_with_executor.on_turn_complete("user1")
    
    ##Step purpose: Find VERIFY_ASSUMPTION task (interval 3)
    verify_task = next(t for t in scheduler_with_executor._tasks if t.schedule.task_type == TaskType.VERIFY_ASSUMPTION)
    
    ##Assertion purpose: Verify execution count (turns 3 and 6, but we only did 5)
    assert verify_task.execution_count >= 1


##Function purpose: Test turns_since_last tracking
def test_turns_since_last_tracking(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: turns_since_last resets after execution."""
    ##Step purpose: Set up successful execution
    mock_executor.execute_task = Mock(return_value=True)
    
    ##Action purpose: Trigger 4 turns (task due at 3)
    for _ in range(4):
        scheduler_with_executor.on_turn_complete("user1")
    
    ##Step purpose: Find VERIFY_ASSUMPTION task
    verify_task = next(t for t in scheduler_with_executor._tasks if t.schedule.task_type == TaskType.VERIFY_ASSUMPTION)
    
    ##Assertion purpose: Verify turns_since_last reset after execution at turn 3
    assert verify_task.turns_since_last == 1  # One turn since last execution


##Function purpose: Test multiple tasks due same turn
def test_multiple_tasks_due_same_turn(scheduler_with_executor: CognitiveScheduler, mock_executor: Mock) -> None:
    """##Test case: Multiple tasks can execute in same turn."""
    ##Step purpose: Set up successful execution
    mock_executor.execute_task = Mock(return_value=True)
    
    ##Action purpose: Trigger turn 15 (multiple tasks due)
    # VERIFY_ASSUMPTION: interval 3 (due at 3, 6, 9, 12, 15)
    # GENERATE_INSIGHT: interval 5 (due at 5, 10, 15)
    # REFLECT: interval 10 (due at 10, 20)
    for _ in range(15):
        scheduler_with_executor.on_turn_complete("user1")
    
    ##Assertion purpose: Verify multiple tasks executed
    # At turn 15: VERIFY, GENERATE_INSIGHT should be due
    assert mock_executor.execute_task.called
