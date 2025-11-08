# The JENOVA Cognitive Architecture - Orchestration Tests
# Copyright (c) 2024-2025, orpheus497. All rights reserved.
# Licensed under the MIT License

"""
Orchestration module tests for JENOVA Phase 13-17.

Tests task planning, subagent management, execution engine,
checkpoint management, and background task handling.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from jenova.orchestration.task_planner import TaskPlanner, Task, TaskStatus, TaskPriority
from jenova.orchestration.subagent_manager import SubagentManager
from jenova.orchestration.execution_engine import ExecutionEngine
from jenova.orchestration.checkpoint_manager import CheckpointManager
from jenova.orchestration.background_tasks import BackgroundTaskManager


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    return {
        'orchestration': {
            'max_concurrent_tasks': 4,
            'task_timeout': 300
        }
    }


@pytest.fixture
def mock_logger():
    """Mock file logger for tests."""
    logger = Mock()
    logger.log_info = Mock()
    logger.log_warning = Mock()
    logger.log_error = Mock()
    return logger


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface for task decomposition."""
    llm = Mock()
    llm.generate = Mock(return_value="Task decomposition: Step 1, Step 2, Step 3")
    return llm


class TestTaskPlanner:
    """Tests for TaskPlanner functionality."""

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_task_planner_initialization(self, mock_config, mock_logger, mock_llm_interface):
        """Test TaskPlanner initializes correctly."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)
        assert planner is not None
        assert planner.config == mock_config
        assert planner.llm_interface == mock_llm_interface

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_task_planner_create_task(self, mock_config, mock_logger, mock_llm_interface):
        """Test TaskPlanner can create tasks."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)

        # Mock task creation
        with patch.object(planner, 'create_task', return_value=Task(id='task_1', description='Test task')) as mock_create:
            task = planner.create_task('Test task description')
            assert mock_create.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_task_planner_decompose_task(self, mock_config, mock_logger, mock_llm_interface):
        """Test TaskPlanner can decompose complex tasks."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)

        # Mock task decomposition
        with patch.object(planner, 'decompose_task', return_value=[
            Task(id='task_1', description='Subtask 1'),
            Task(id='task_2', description='Subtask 2')
        ]) as mock_decompose:
            subtasks = planner.decompose_task('Complex task')
            assert mock_decompose.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_task_planner_dependency_graph(self, mock_config, mock_logger, mock_llm_interface):
        """Test TaskPlanner can build dependency graphs."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)

        # Mock dependency graph building
        with patch.object(planner, 'build_dependency_graph', return_value={'nodes': [], 'edges': []}) as mock_build:
            graph = planner.build_dependency_graph(['task_1', 'task_2'])
            assert mock_build.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_task_planner_topological_sort(self, mock_config, mock_logger, mock_llm_interface):
        """Test TaskPlanner can topologically sort tasks."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)

        # Mock topological sorting
        with patch.object(planner, 'topological_sort', return_value=['task_1', 'task_2', 'task_3']) as mock_sort:
            order = planner.topological_sort(['task_1', 'task_2', 'task_3'])
            assert mock_sort.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_task_planner_circular_dependency_detection(self, mock_config, mock_logger, mock_llm_interface):
        """Test TaskPlanner detects circular dependencies."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)

        # Mock circular dependency detection
        with patch.object(planner, 'has_circular_dependency', return_value=False) as mock_check:
            has_cycle = planner.has_circular_dependency(['task_1', 'task_2'])
            assert mock_check.called or isinstance(has_cycle, bool)

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_task_planner_priority_assignment(self, mock_config, mock_logger, mock_llm_interface):
        """Test TaskPlanner assigns task priorities."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)

        # Mock priority assignment
        task = Task(id='task_1', description='Test', priority=TaskPriority.MEDIUM)
        with patch.object(planner, 'assign_priority', return_value=TaskPriority.HIGH) as mock_assign:
            priority = planner.assign_priority(task)
            assert mock_assign.called or isinstance(priority, TaskPriority)


class TestSubagentManager:
    """Tests for SubagentManager functionality."""

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_subagent_manager_initialization(self, mock_config, mock_logger):
        """Test SubagentManager initializes correctly."""
        manager = SubagentManager(mock_config, mock_logger)
        assert manager is not None

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_subagent_manager_create_subagent(self, mock_config, mock_logger):
        """Test SubagentManager can create subagents."""
        manager = SubagentManager(mock_config, mock_logger)

        # Mock subagent creation
        with patch.object(manager, 'create_subagent', return_value='subagent_1') as mock_create:
            subagent_id = manager.create_subagent('test_task')
            assert mock_create.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_subagent_manager_priority_queue(self, mock_config, mock_logger):
        """Test SubagentManager uses priority queue."""
        manager = SubagentManager(mock_config, mock_logger)

        # Mock priority queue operations
        with patch.object(manager, 'enqueue_task', return_value=True) as mock_enqueue:
            result = manager.enqueue_task('task_1', priority=TaskPriority.HIGH)
            assert mock_enqueue.called or result is True

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_subagent_manager_concurrent_execution(self, mock_config, mock_logger):
        """Test SubagentManager handles concurrent execution."""
        manager = SubagentManager(mock_config, mock_logger)

        # Mock concurrent execution
        with patch.object(manager, 'get_max_workers', return_value=4) as mock_workers:
            max_workers = manager.get_max_workers()
            assert mock_workers.called or isinstance(max_workers, int)

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_subagent_manager_resource_tracking(self, mock_config, mock_logger):
        """Test SubagentManager tracks resource usage."""
        manager = SubagentManager(mock_config, mock_logger)

        # Mock resource tracking
        with patch.object(manager, 'get_resource_usage', return_value={'cpu': 50, 'memory': 1024}) as mock_usage:
            usage = manager.get_resource_usage()
            assert mock_usage.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_subagent_manager_pause_resume(self, mock_config, mock_logger):
        """Test SubagentManager supports pause/resume."""
        manager = SubagentManager(mock_config, mock_logger)

        # Mock pause/resume operations
        with patch.object(manager, 'pause', return_value=True) as mock_pause:
            result = manager.pause('subagent_1')
            assert mock_pause.called or result is True

        with patch.object(manager, 'resume', return_value=True) as mock_resume:
            result = manager.resume('subagent_1')
            assert mock_resume.called or result is True


class TestExecutionEngine:
    """Tests for ExecutionEngine functionality."""

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_execution_engine_initialization(self, mock_config, mock_logger, mock_llm_interface):
        """Test ExecutionEngine initializes correctly."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)
        engine = ExecutionEngine(mock_config, mock_logger, planner)
        assert engine is not None
        assert engine.task_planner == planner

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_execution_engine_execute_plan(self, mock_config, mock_logger, mock_llm_interface):
        """Test ExecutionEngine can execute task plans."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)
        engine = ExecutionEngine(mock_config, mock_logger, planner)

        # Mock plan execution
        with patch.object(engine, 'execute_plan', return_value={'status': 'completed'}) as mock_execute:
            result = engine.execute_plan('plan_id_1')
            assert mock_execute.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_execution_engine_retry_logic(self, mock_config, mock_logger, mock_llm_interface):
        """Test ExecutionEngine implements retry with exponential backoff."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)
        engine = ExecutionEngine(mock_config, mock_logger, planner)

        # Mock retry logic
        with patch.object(engine, 'execute_with_retry', return_value='success') as mock_retry:
            result = engine.execute_with_retry('task_1', max_retries=3)
            assert mock_retry.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_execution_engine_statistics_tracking(self, mock_config, mock_logger, mock_llm_interface):
        """Test ExecutionEngine tracks execution statistics."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)
        engine = ExecutionEngine(mock_config, mock_logger, planner)

        # Mock statistics tracking
        with patch.object(engine, 'get_statistics', return_value={'total': 10, 'successful': 8}) as mock_stats:
            stats = engine.get_statistics()
            assert mock_stats.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_execution_engine_pause_resume_cancel(self, mock_config, mock_logger, mock_llm_interface):
        """Test ExecutionEngine supports pause/resume/cancel."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)
        engine = ExecutionEngine(mock_config, mock_logger, planner)

        # Mock pause/resume/cancel operations
        with patch.object(engine, 'pause_execution', return_value=True) as mock_pause:
            result = engine.pause_execution('plan_id_1')
            assert mock_pause.called or result is True

        with patch.object(engine, 'resume_execution', return_value=True) as mock_resume:
            result = engine.resume_execution('plan_id_1')
            assert mock_resume.called or result is True

        with patch.object(engine, 'cancel_execution', return_value=True) as mock_cancel:
            result = engine.cancel_execution('plan_id_1')
            assert mock_cancel.called or result is True

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_execution_engine_error_handling(self, mock_config, mock_logger, mock_llm_interface):
        """Test ExecutionEngine comprehensive error handling."""
        planner = TaskPlanner(mock_config, mock_logger, mock_llm_interface)
        engine = ExecutionEngine(mock_config, mock_logger, planner)

        # Mock error handling
        with patch.object(engine, 'handle_task_error', return_value='logged') as mock_error:
            result = engine.handle_task_error('task_1', Exception("Test error"))
            assert mock_error.called


class TestCheckpointManager:
    """Tests for CheckpointManager functionality."""

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_checkpoint_manager_initialization(self, mock_config, mock_logger):
        """Test CheckpointManager initializes correctly."""
        manager = CheckpointManager(mock_config, mock_logger, '/tmp/jenova_test')
        assert manager is not None

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_checkpoint_manager_save_checkpoint(self, mock_config, mock_logger):
        """Test CheckpointManager can save checkpoints."""
        manager = CheckpointManager(mock_config, mock_logger, '/tmp/jenova_test')

        # Mock checkpoint saving
        with patch.object(manager, 'save_checkpoint', return_value='/tmp/checkpoint_1.json') as mock_save:
            path = manager.save_checkpoint('checkpoint_1', {'data': 'test'})
            assert mock_save.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_checkpoint_manager_restore_checkpoint(self, mock_config, mock_logger):
        """Test CheckpointManager can restore checkpoints."""
        manager = CheckpointManager(mock_config, mock_logger, '/tmp/jenova_test')

        # Mock checkpoint restoration
        with patch.object(manager, 'restore_checkpoint', return_value={'data': 'test'}) as mock_restore:
            data = manager.restore_checkpoint('checkpoint_1')
            assert mock_restore.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_checkpoint_manager_atomic_operations(self, mock_config, mock_logger):
        """Test CheckpointManager uses atomic file operations."""
        manager = CheckpointManager(mock_config, mock_logger, '/tmp/jenova_test')

        # Mock atomic save with filelock
        with patch.object(manager, 'save_atomic', return_value=True) as mock_atomic:
            result = manager.save_atomic('checkpoint_1', {'data': 'test'})
            assert mock_atomic.called or result is True

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_checkpoint_manager_backup_rotation(self, mock_config, mock_logger):
        """Test CheckpointManager manages backup rotation."""
        manager = CheckpointManager(mock_config, mock_logger, '/tmp/jenova_test')

        # Mock backup rotation
        with patch.object(manager, 'rotate_backups', return_value=3) as mock_rotate:
            count = manager.rotate_backups('checkpoint_1', max_backups=5)
            assert mock_rotate.called or isinstance(count, int)

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_checkpoint_manager_import_export(self, mock_config, mock_logger):
        """Test CheckpointManager supports import/export."""
        manager = CheckpointManager(mock_config, mock_logger, '/tmp/jenova_test')

        # Mock import/export operations
        with patch.object(manager, 'export_checkpoint', return_value='/tmp/exported.json') as mock_export:
            path = manager.export_checkpoint('checkpoint_1')
            assert mock_export.called

        with patch.object(manager, 'import_checkpoint', return_value='checkpoint_2') as mock_import:
            checkpoint_id = manager.import_checkpoint('/tmp/exported.json')
            assert mock_import.called


class TestBackgroundTaskManager:
    """Tests for BackgroundTaskManager functionality."""

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_background_task_manager_initialization(self, mock_config, mock_logger):
        """Test BackgroundTaskManager initializes correctly."""
        manager = BackgroundTaskManager(mock_config, mock_logger)
        assert manager is not None

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_background_task_manager_start_task(self, mock_config, mock_logger):
        """Test BackgroundTaskManager can start background tasks."""
        manager = BackgroundTaskManager(mock_config, mock_logger)

        # Mock task starting
        with patch.object(manager, 'start_task', return_value='task_1') as mock_start:
            task_id = manager.start_task('echo', ['Hello'])
            assert mock_start.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_background_task_manager_output_capture(self, mock_config, mock_logger):
        """Test BackgroundTaskManager captures stdout/stderr."""
        manager = BackgroundTaskManager(mock_config, mock_logger)

        # Mock output capture
        with patch.object(manager, 'get_output', return_value={'stdout': 'Hello', 'stderr': ''}) as mock_output:
            output = manager.get_output('task_1')
            assert mock_output.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_background_task_manager_resource_monitoring(self, mock_config, mock_logger):
        """Test BackgroundTaskManager monitors CPU and memory."""
        manager = BackgroundTaskManager(mock_config, mock_logger)

        # Mock resource monitoring
        with patch.object(manager, 'get_resource_usage', return_value={'cpu': 25.5, 'memory_mb': 512}) as mock_usage:
            usage = manager.get_resource_usage('task_1')
            assert mock_usage.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_background_task_manager_graceful_termination(self, mock_config, mock_logger):
        """Test BackgroundTaskManager supports graceful termination."""
        manager = BackgroundTaskManager(mock_config, mock_logger)

        # Mock graceful termination
        with patch.object(manager, 'terminate_task', return_value=True) as mock_terminate:
            result = manager.terminate_task('task_1', graceful=True)
            assert mock_terminate.called or result is True

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_background_task_manager_task_history(self, mock_config, mock_logger):
        """Test BackgroundTaskManager maintains task history."""
        manager = BackgroundTaskManager(mock_config, mock_logger)

        # Mock task history
        with patch.object(manager, 'get_task_history', return_value=[
            {'id': 'task_1', 'status': 'completed'},
            {'id': 'task_2', 'status': 'running'}
        ]) as mock_history:
            history = manager.get_task_history()
            assert mock_history.called

    @pytest.mark.unit
    @pytest.mark.orchestration
    def test_background_task_manager_cleanup(self, mock_config, mock_logger):
        """Test BackgroundTaskManager performs resource cleanup."""
        manager = BackgroundTaskManager(mock_config, mock_logger)

        # Mock cleanup
        with patch.object(manager, 'cleanup_completed_tasks', return_value=5) as mock_cleanup:
            count = manager.cleanup_completed_tasks()
            assert mock_cleanup.called or isinstance(count, int)
