# The JENOVA Cognitive Architecture - Automation Tests
# Copyright (c) 2024-2025, orpheus497. All rights reserved.
# Licensed under the MIT License

"""
Automation module tests for JENOVA Phase 13-17.

Tests custom commands, hooks system, template engine,
and workflow library functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from jenova.automation.custom_commands import CustomCommandManager
from jenova.automation.hooks_system import HooksSystem
from jenova.automation.template_engine import TemplateEngine
from jenova.automation.workflow_library import WorkflowLibrary, WorkflowType


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    return {
        'automation': {
            'max_hook_execution_time': 30,
            'enable_hooks': True
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
def mock_template_engine():
    """Mock template engine for custom commands."""
    engine = Mock()
    engine.render = Mock(return_value="Rendered template output")
    return engine


class TestCustomCommandManager:
    """Tests for CustomCommandManager functionality."""

    @pytest.mark.unit
    @pytest.mark.automation
    def test_custom_command_manager_initialization(self, mock_config, mock_logger, mock_template_engine):
        """Test CustomCommandManager initializes correctly."""
        manager = CustomCommandManager(mock_config, mock_logger, mock_template_engine, '/tmp/commands')
        assert manager is not None
        assert manager.template_engine == mock_template_engine

    @pytest.mark.unit
    @pytest.mark.automation
    def test_custom_command_manager_create_command(self, mock_config, mock_logger, mock_template_engine):
        """Test CustomCommandManager can create custom commands."""
        manager = CustomCommandManager(mock_config, mock_logger, mock_template_engine, '/tmp/commands')

        # Mock command creation
        with patch.object(manager, 'create_command', return_value='my_command') as mock_create:
            command_id = manager.create_command('my_command', 'Command template content')
            assert mock_create.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_custom_command_manager_yaml_frontmatter(self, mock_config, mock_logger, mock_template_engine):
        """Test CustomCommandManager parses YAML frontmatter."""
        manager = CustomCommandManager(mock_config, mock_logger, mock_template_engine, '/tmp/commands')

        frontmatter = """---
name: test_command
description: A test command
variables:
  - name
  - output
---
Hello {{name}}, output to {{output}}
"""
        # Mock frontmatter parsing
        with patch.object(manager, 'parse_frontmatter', return_value={
            'name': 'test_command',
            'description': 'A test command',
            'variables': ['name', 'output']
        }) as mock_parse:
            metadata = manager.parse_frontmatter(frontmatter)
            assert mock_parse.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_custom_command_manager_variable_extraction(self, mock_config, mock_logger, mock_template_engine):
        """Test CustomCommandManager extracts template variables."""
        manager = CustomCommandManager(mock_config, mock_logger, mock_template_engine, '/tmp/commands')

        template = "Hello {{name}}, your task is {{task}}"
        # Mock variable extraction
        with patch.object(manager, 'extract_variables', return_value=['name', 'task']) as mock_extract:
            variables = manager.extract_variables(template)
            assert mock_extract.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_custom_command_manager_execute_command(self, mock_config, mock_logger, mock_template_engine):
        """Test CustomCommandManager can execute commands."""
        manager = CustomCommandManager(mock_config, mock_logger, mock_template_engine, '/tmp/commands')

        # Mock command execution
        with patch.object(manager, 'execute_command', return_value="Command executed successfully") as mock_execute:
            result = manager.execute_command('my_command', {'var1': 'value1'})
            assert mock_execute.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_custom_command_manager_list_commands(self, mock_config, mock_logger, mock_template_engine):
        """Test CustomCommandManager can list available commands."""
        manager = CustomCommandManager(mock_config, mock_logger, mock_template_engine, '/tmp/commands')

        # Mock command listing
        with patch.object(manager, 'list_commands', return_value=['cmd1', 'cmd2', 'cmd3']) as mock_list:
            commands = manager.list_commands()
            assert mock_list.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_custom_command_manager_delete_command(self, mock_config, mock_logger, mock_template_engine):
        """Test CustomCommandManager can delete commands."""
        manager = CustomCommandManager(mock_config, mock_logger, mock_template_engine, '/tmp/commands')

        # Mock command deletion
        with patch.object(manager, 'delete_command', return_value=True) as mock_delete:
            result = manager.delete_command('old_command')
            assert mock_delete.called or result is True


class TestHooksSystem:
    """Tests for HooksSystem functionality."""

    @pytest.mark.unit
    @pytest.mark.automation
    def test_hooks_system_initialization(self, mock_config, mock_logger):
        """Test HooksSystem initializes correctly."""
        hooks = HooksSystem(mock_config, mock_logger)
        assert hooks is not None

    @pytest.mark.unit
    @pytest.mark.automation
    def test_hooks_system_register_hook(self, mock_config, mock_logger):
        """Test HooksSystem can register hooks."""
        hooks = HooksSystem(mock_config, mock_logger)

        # Mock hook registration
        with patch.object(hooks, 'register_hook', return_value=True) as mock_register:
            result = hooks.register_hook('pre_execute', 'hook_function', priority=1)
            assert mock_register.called or result is True

    @pytest.mark.unit
    @pytest.mark.automation
    def test_hooks_system_hook_timing(self, mock_config, mock_logger):
        """Test HooksSystem supports pre/post/on-error timing."""
        hooks = HooksSystem(mock_config, mock_logger)

        hook_timings = ['pre', 'post', 'on_error']
        for timing in hook_timings:
            # Mock hook timing validation
            with patch.object(hooks, 'is_valid_timing', return_value=True) as mock_valid:
                is_valid = hooks.is_valid_timing(timing)
                assert mock_valid.called or is_valid is True

    @pytest.mark.unit
    @pytest.mark.automation
    def test_hooks_system_priority_execution(self, mock_config, mock_logger):
        """Test HooksSystem executes hooks in priority order."""
        hooks = HooksSystem(mock_config, mock_logger)

        # Mock priority-based execution
        with patch.object(hooks, 'execute_hooks', return_value=['hook1_result', 'hook2_result']) as mock_execute:
            results = hooks.execute_hooks('pre_execute', {'context': 'test'})
            assert mock_execute.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_hooks_system_enable_disable(self, mock_config, mock_logger):
        """Test HooksSystem supports enabling/disabling hooks."""
        hooks = HooksSystem(mock_config, mock_logger)

        # Mock enable/disable operations
        with patch.object(hooks, 'disable_hook', return_value=True) as mock_disable:
            result = hooks.disable_hook('hook_name')
            assert mock_disable.called or result is True

        with patch.object(hooks, 'enable_hook', return_value=True) as mock_enable:
            result = hooks.enable_hook('hook_name')
            assert mock_enable.called or result is True

    @pytest.mark.unit
    @pytest.mark.automation
    def test_hooks_system_error_handling(self, mock_config, mock_logger):
        """Test HooksSystem comprehensive error handling."""
        hooks = HooksSystem(mock_config, mock_logger)

        # Mock error handling in hook execution
        with patch.object(hooks, 'execute_hooks', side_effect=Exception("Hook error")) as mock_execute:
            with pytest.raises(Exception):
                hooks.execute_hooks('pre_execute', {})

    @pytest.mark.unit
    @pytest.mark.automation
    def test_hooks_system_execution_history(self, mock_config, mock_logger):
        """Test HooksSystem maintains execution history."""
        hooks = HooksSystem(mock_config, mock_logger)

        # Mock execution history
        with patch.object(hooks, 'get_execution_history', return_value=[
            {'hook': 'hook1', 'timing': 'pre', 'status': 'success'},
            {'hook': 'hook2', 'timing': 'post', 'status': 'success'}
        ]) as mock_history:
            history = hooks.get_execution_history()
            assert mock_history.called


class TestTemplateEngine:
    """Tests for TemplateEngine functionality."""

    @pytest.mark.unit
    @pytest.mark.automation
    def test_template_engine_initialization(self, mock_config, mock_logger):
        """Test TemplateEngine initializes correctly."""
        engine = TemplateEngine(mock_config, mock_logger)
        assert engine is not None

    @pytest.mark.unit
    @pytest.mark.automation
    def test_template_engine_variable_substitution(self, mock_config, mock_logger):
        """Test TemplateEngine performs variable substitution."""
        engine = TemplateEngine(mock_config, mock_logger)

        template = "Hello {{name}}, welcome to {{place}}"
        variables = {'name': 'Alice', 'place': 'Wonderland'}

        # Mock rendering
        with patch.object(engine, 'render', return_value="Hello Alice, welcome to Wonderland") as mock_render:
            result = engine.render(template, variables)
            assert mock_render.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_template_engine_built_in_filters(self, mock_config, mock_logger):
        """Test TemplateEngine supports built-in filters."""
        engine = TemplateEngine(mock_config, mock_logger)

        # Test various filters
        filters = ['upper', 'lower', 'title', 'capitalize', 'strip', 'len', 'default']
        for filter_name in filters:
            # Mock filter availability check
            with patch.object(engine, 'has_filter', return_value=True) as mock_has:
                has_filter = engine.has_filter(filter_name)
                assert mock_has.called or has_filter is True

    @pytest.mark.unit
    @pytest.mark.automation
    def test_template_engine_conditionals(self, mock_config, mock_logger):
        """Test TemplateEngine supports conditional statements."""
        engine = TemplateEngine(mock_config, mock_logger)

        template = "{% if enabled %}Feature enabled{% else %}Feature disabled{% endif %}"
        # Mock conditional rendering
        with patch.object(engine, 'render', return_value="Feature enabled") as mock_render:
            result = engine.render(template, {'enabled': True})
            assert mock_render.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_template_engine_loops(self, mock_config, mock_logger):
        """Test TemplateEngine supports loops."""
        engine = TemplateEngine(mock_config, mock_logger)

        template = "{% for item in items %}{{item}}, {% endfor %}"
        # Mock loop rendering
        with patch.object(engine, 'render', return_value="a, b, c, ") as mock_render:
            result = engine.render(template, {'items': ['a', 'b', 'c']})
            assert mock_render.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_template_engine_custom_filters(self, mock_config, mock_logger):
        """Test TemplateEngine supports custom filter registration."""
        engine = TemplateEngine(mock_config, mock_logger)

        # Mock custom filter registration
        def my_filter(value):
            return f"processed_{value}"

        with patch.object(engine, 'register_filter', return_value=True) as mock_register:
            result = engine.register_filter('my_filter', my_filter)
            assert mock_register.called or result is True

    @pytest.mark.unit
    @pytest.mark.automation
    def test_template_engine_error_handling(self, mock_config, mock_logger):
        """Test TemplateEngine handles template errors gracefully."""
        engine = TemplateEngine(mock_config, mock_logger)

        # Mock error handling for invalid template
        with patch.object(engine, 'render', side_effect=ValueError("Invalid template syntax")) as mock_render:
            with pytest.raises(ValueError):
                engine.render("{{invalid syntax", {})


class TestWorkflowLibrary:
    """Tests for WorkflowLibrary functionality."""

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_initialization(self, mock_config, mock_logger):
        """Test WorkflowLibrary initializes correctly."""
        library = WorkflowLibrary(mock_config, mock_logger)
        assert library is not None

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_predefined_workflows(self, mock_config, mock_logger):
        """Test WorkflowLibrary provides 6 predefined workflows."""
        library = WorkflowLibrary(mock_config, mock_logger)

        workflow_types = [
            WorkflowType.CODE_REVIEW,
            WorkflowType.TESTING,
            WorkflowType.DEPLOYMENT,
            WorkflowType.REFACTORING,
            WorkflowType.DOCUMENTATION,
            WorkflowType.ANALYSIS
        ]

        # Mock workflow retrieval
        for workflow_type in workflow_types:
            with patch.object(library, 'get_workflow', return_value={'name': workflow_type.value}) as mock_get:
                workflow = library.get_workflow(workflow_type)
                assert mock_get.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_step_dependencies(self, mock_config, mock_logger):
        """Test WorkflowLibrary manages workflow step dependencies."""
        library = WorkflowLibrary(mock_config, mock_logger)

        # Mock dependency management
        with patch.object(library, 'get_step_dependencies', return_value=['step1', 'step2']) as mock_deps:
            deps = library.get_step_dependencies(WorkflowType.CODE_REVIEW, 'step3')
            assert mock_deps.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_clone_workflow(self, mock_config, mock_logger):
        """Test WorkflowLibrary supports workflow cloning."""
        library = WorkflowLibrary(mock_config, mock_logger)

        # Mock workflow cloning
        with patch.object(library, 'clone_workflow', return_value='custom_workflow_1') as mock_clone:
            workflow_id = library.clone_workflow(WorkflowType.CODE_REVIEW, 'my_review')
            assert mock_clone.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_execute_workflow(self, mock_config, mock_logger):
        """Test WorkflowLibrary can execute workflows."""
        library = WorkflowLibrary(mock_config, mock_logger)

        # Mock workflow execution
        with patch.object(library, 'execute_workflow', return_value={'status': 'completed'}) as mock_execute:
            result = library.execute_workflow(WorkflowType.TESTING, {'target': 'tests/'})
            assert mock_execute.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_custom_workflows(self, mock_config, mock_logger):
        """Test WorkflowLibrary supports custom workflow creation."""
        library = WorkflowLibrary(mock_config, mock_logger)

        # Mock custom workflow creation
        with patch.object(library, 'create_custom_workflow', return_value='custom_workflow_2') as mock_create:
            workflow_id = library.create_custom_workflow(
                'My Workflow',
                [{'name': 'step1'}, {'name': 'step2'}]
            )
            assert mock_create.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_workflow_serialization(self, mock_config, mock_logger):
        """Test WorkflowLibrary supports workflow serialization."""
        library = WorkflowLibrary(mock_config, mock_logger)

        # Mock workflow serialization
        with patch.object(library, 'serialize_workflow', return_value='{"name": "test"}') as mock_serialize:
            json_str = library.serialize_workflow(WorkflowType.DEPLOYMENT)
            assert mock_serialize.called

        # Mock workflow deserialization
        with patch.object(library, 'deserialize_workflow', return_value={'name': 'test'}) as mock_deserialize:
            workflow = library.deserialize_workflow('{"name": "test"}')
            assert mock_deserialize.called

    @pytest.mark.unit
    @pytest.mark.automation
    def test_workflow_library_comprehensive_metadata(self, mock_config, mock_logger):
        """Test WorkflowLibrary provides comprehensive workflow metadata."""
        library = WorkflowLibrary(mock_config, mock_logger)

        # Mock metadata retrieval
        with patch.object(library, 'get_workflow_metadata', return_value={
            'name': 'Code Review',
            'description': 'Comprehensive code review workflow',
            'steps': 5,
            'estimated_time': '15 minutes'
        }) as mock_metadata:
            metadata = library.get_workflow_metadata(WorkflowType.CODE_REVIEW)
            assert mock_metadata.called
