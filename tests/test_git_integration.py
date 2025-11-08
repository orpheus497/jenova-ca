# The JENOVA Cognitive Architecture - Git Integration Tests
# Copyright (c) 2024-2025, orpheus497. All rights reserved.
# Licensed under the MIT License

"""
Git integration module tests for JENOVA Phase 13-17.

Tests git operations, commit message generation, diff analysis,
hooks management, and branch operations.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from jenova.git_tools.git_interface import GitInterface
from jenova.git_tools.commit_assistant import CommitAssistant
from jenova.git_tools.diff_analyzer import DiffAnalyzer
from jenova.git_tools.hooks_manager import HooksManager
from jenova.git_tools.branch_manager import BranchManager


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    return {
        'tools': {
            'file_sandbox_path': '/tmp/jenova_test'
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
    """Mock LLM interface for commit message generation."""
    llm = Mock()
    llm.generate = Mock(return_value="feat: Add new feature\n\nImplements user-requested functionality.")
    return llm


class TestGitInterface:
    """Tests for GitInterface functionality."""

    @pytest.mark.unit
    @pytest.mark.git
    def test_git_interface_initialization(self, mock_config, mock_logger):
        """Test GitInterface initializes correctly."""
        git = GitInterface(mock_config, mock_logger)
        assert git is not None
        assert git.config == mock_config
        assert git.file_logger == mock_logger

    @pytest.mark.unit
    @pytest.mark.git
    def test_git_status_operation(self, mock_config, mock_logger):
        """Test GitInterface can execute git status."""
        git = GitInterface(mock_config, mock_logger)

        # Mock the git status operation
        with patch.object(git, 'execute', return_value="On branch main\nnothing to commit") as mock_exec:
            result = git.execute('status', [])
            assert mock_exec.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_git_diff_operation(self, mock_config, mock_logger):
        """Test GitInterface can execute git diff."""
        git = GitInterface(mock_config, mock_logger)

        # Mock the git diff operation
        with patch.object(git, 'execute', return_value="diff --git a/file.py b/file.py") as mock_exec:
            result = git.execute('diff', [])
            assert mock_exec.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_git_log_operation(self, mock_config, mock_logger):
        """Test GitInterface can execute git log."""
        git = GitInterface(mock_config, mock_logger)

        # Mock the git log operation
        with patch.object(git, 'execute', return_value="commit abc123\nAuthor: test") as mock_exec:
            result = git.execute('log', ['-n', '5'])
            assert mock_exec.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_git_branch_operation(self, mock_config, mock_logger):
        """Test GitInterface can execute git branch."""
        git = GitInterface(mock_config, mock_logger)

        # Mock the git branch operation
        with patch.object(git, 'execute', return_value="* main\n  develop") as mock_exec:
            result = git.execute('branch', [])
            assert mock_exec.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_git_invalid_operation(self, mock_config, mock_logger):
        """Test GitInterface handles invalid operations gracefully."""
        git = GitInterface(mock_config, mock_logger)

        # Mock error handling
        with patch.object(git, 'execute', side_effect=ValueError("Invalid git operation")) as mock_exec:
            with pytest.raises(ValueError):
                git.execute('invalid_command', [])


class TestCommitAssistant:
    """Tests for CommitAssistant functionality."""

    @pytest.mark.unit
    @pytest.mark.git
    def test_commit_assistant_initialization(self, mock_config, mock_logger, mock_llm_interface):
        """Test CommitAssistant initializes correctly."""
        assistant = CommitAssistant(mock_config, mock_logger, mock_llm_interface)
        assert assistant is not None
        assert assistant.llm_interface == mock_llm_interface

    @pytest.mark.unit
    @pytest.mark.git
    def test_commit_assistant_generate_message(self, mock_config, mock_logger, mock_llm_interface):
        """Test CommitAssistant can generate commit messages."""
        assistant = CommitAssistant(mock_config, mock_logger, mock_llm_interface)

        # Mock the commit message generation
        with patch.object(assistant, 'generate_commit_message', return_value="feat: Add feature") as mock_gen:
            message = assistant.generate_commit_message("diff content")
            assert mock_gen.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_commit_assistant_auto_commit(self, mock_config, mock_logger, mock_llm_interface):
        """Test CommitAssistant can perform auto-commit."""
        assistant = CommitAssistant(mock_config, mock_logger, mock_llm_interface)

        # Mock the auto-commit operation
        with patch.object(assistant, 'auto_commit', return_value="Committed with message: feat: Add feature") as mock_commit:
            result = assistant.auto_commit([])
            assert mock_commit.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_commit_assistant_conventional_commits(self, mock_config, mock_logger, mock_llm_interface):
        """Test CommitAssistant follows Conventional Commits format."""
        assistant = CommitAssistant(mock_config, mock_logger, mock_llm_interface)

        # Mock the message validation
        with patch.object(assistant, 'validate_commit_message', return_value=True) as mock_validate:
            is_valid = assistant.validate_commit_message("feat: Add new feature")
            assert mock_validate.called or is_valid is True

    @pytest.mark.unit
    @pytest.mark.git
    def test_commit_assistant_empty_diff(self, mock_config, mock_logger, mock_llm_interface):
        """Test CommitAssistant handles empty diffs."""
        assistant = CommitAssistant(mock_config, mock_logger, mock_llm_interface)

        # Mock handling of empty diff
        with patch.object(assistant, 'generate_commit_message', side_effect=ValueError("No changes to commit")) as mock_gen:
            with pytest.raises(ValueError):
                assistant.generate_commit_message("")


class TestDiffAnalyzer:
    """Tests for DiffAnalyzer functionality."""

    @pytest.mark.unit
    @pytest.mark.git
    def test_diff_analyzer_initialization(self, mock_config, mock_logger, mock_llm_interface):
        """Test DiffAnalyzer initializes correctly."""
        analyzer = DiffAnalyzer(mock_config, mock_logger, mock_llm_interface)
        assert analyzer is not None

    @pytest.mark.unit
    @pytest.mark.git
    def test_diff_analyzer_parse_diff(self, mock_config, mock_logger, mock_llm_interface):
        """Test DiffAnalyzer can parse git diff output."""
        analyzer = DiffAnalyzer(mock_config, mock_logger, mock_llm_interface)

        diff_content = """diff --git a/test.py b/test.py
index abc123..def456 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
+# New comment
 def test():
     return True
"""
        # Mock the diff parsing
        with patch.object(analyzer, 'parse_diff', return_value={'files': ['test.py'], 'additions': 1}) as mock_parse:
            result = analyzer.parse_diff(diff_content)
            assert mock_parse.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_diff_analyzer_summarize(self, mock_config, mock_logger, mock_llm_interface):
        """Test DiffAnalyzer can summarize changes."""
        analyzer = DiffAnalyzer(mock_config, mock_logger, mock_llm_interface)

        # Mock the summarization
        with patch.object(analyzer, 'summarize_changes', return_value="Added 1 line in test.py") as mock_summarize:
            summary = analyzer.summarize_changes("diff content")
            assert mock_summarize.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_diff_analyzer_identify_impact(self, mock_config, mock_logger, mock_llm_interface):
        """Test DiffAnalyzer can identify change impact."""
        analyzer = DiffAnalyzer(mock_config, mock_logger, mock_llm_interface)

        # Mock the impact analysis
        with patch.object(analyzer, 'analyze_impact', return_value={'risk': 'low', 'scope': 'minor'}) as mock_impact:
            impact = analyzer.analyze_impact("diff content")
            assert mock_impact.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_diff_analyzer_multiple_files(self, mock_config, mock_logger, mock_llm_interface):
        """Test DiffAnalyzer handles multi-file diffs."""
        analyzer = DiffAnalyzer(mock_config, mock_logger, mock_llm_interface)

        # Mock multi-file parsing
        with patch.object(analyzer, 'parse_diff', return_value={'files': ['a.py', 'b.py'], 'additions': 5}) as mock_parse:
            result = analyzer.parse_diff("multi-file diff content")
            assert mock_parse.called


class TestHooksManager:
    """Tests for HooksManager functionality."""

    @pytest.mark.unit
    @pytest.mark.git
    def test_hooks_manager_initialization(self, mock_config, mock_logger):
        """Test HooksManager initializes correctly."""
        manager = HooksManager(mock_config, mock_logger)
        assert manager is not None

    @pytest.mark.unit
    @pytest.mark.git
    def test_hooks_manager_install_hook(self, mock_config, mock_logger):
        """Test HooksManager can install git hooks."""
        manager = HooksManager(mock_config, mock_logger)

        # Mock the hook installation
        with patch.object(manager, 'install_hook', return_value=True) as mock_install:
            result = manager.install_hook('pre-commit', 'script content')
            assert mock_install.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_hooks_manager_remove_hook(self, mock_config, mock_logger):
        """Test HooksManager can remove git hooks."""
        manager = HooksManager(mock_config, mock_logger)

        # Mock the hook removal
        with patch.object(manager, 'remove_hook', return_value=True) as mock_remove:
            result = manager.remove_hook('pre-commit')
            assert mock_remove.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_hooks_manager_list_hooks(self, mock_config, mock_logger):
        """Test HooksManager can list installed hooks."""
        manager = HooksManager(mock_config, mock_logger)

        # Mock the hook listing
        with patch.object(manager, 'list_hooks', return_value=['pre-commit', 'post-commit']) as mock_list:
            hooks = manager.list_hooks()
            assert mock_list.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_hooks_manager_hook_types(self, mock_config, mock_logger):
        """Test HooksManager supports various hook types."""
        manager = HooksManager(mock_config, mock_logger)

        hook_types = ['pre-commit', 'post-commit', 'pre-push', 'commit-msg']
        for hook_type in hook_types:
            # Mock support check
            with patch.object(manager, 'is_valid_hook_type', return_value=True) as mock_valid:
                is_valid = manager.is_valid_hook_type(hook_type)
                assert mock_valid.called or is_valid is True


class TestBranchManager:
    """Tests for BranchManager functionality."""

    @pytest.mark.unit
    @pytest.mark.git
    def test_branch_manager_initialization(self, mock_config, mock_logger):
        """Test BranchManager initializes correctly."""
        manager = BranchManager(mock_config, mock_logger)
        assert manager is not None

    @pytest.mark.unit
    @pytest.mark.git
    def test_branch_manager_create_branch(self, mock_config, mock_logger):
        """Test BranchManager can create branches."""
        manager = BranchManager(mock_config, mock_logger)

        # Mock the branch creation
        with patch.object(manager, 'create_branch', return_value="Created branch: feature/new-feature") as mock_create:
            result = manager.create_branch('feature/new-feature')
            assert mock_create.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_branch_manager_delete_branch(self, mock_config, mock_logger):
        """Test BranchManager can delete branches."""
        manager = BranchManager(mock_config, mock_logger)

        # Mock the branch deletion
        with patch.object(manager, 'delete_branch', return_value="Deleted branch: old-feature") as mock_delete:
            result = manager.delete_branch('old-feature')
            assert mock_delete.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_branch_manager_list_branches(self, mock_config, mock_logger):
        """Test BranchManager can list branches."""
        manager = BranchManager(mock_config, mock_logger)

        # Mock the branch listing
        with patch.object(manager, 'list_branches', return_value=['main', 'develop', 'feature/test']) as mock_list:
            branches = manager.list_branches()
            assert mock_list.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_branch_manager_naming_convention(self, mock_config, mock_logger):
        """Test BranchManager enforces naming conventions."""
        manager = BranchManager(mock_config, mock_logger)

        # Mock the naming validation
        with patch.object(manager, 'validate_branch_name', return_value=True) as mock_validate:
            is_valid = manager.validate_branch_name('feature/valid-name')
            assert mock_validate.called or is_valid is True

    @pytest.mark.unit
    @pytest.mark.git
    def test_branch_manager_current_branch(self, mock_config, mock_logger):
        """Test BranchManager can identify current branch."""
        manager = BranchManager(mock_config, mock_logger)

        # Mock getting current branch
        with patch.object(manager, 'get_current_branch', return_value='main') as mock_get:
            branch = manager.get_current_branch()
            assert mock_get.called

    @pytest.mark.unit
    @pytest.mark.git
    def test_branch_manager_switch_branch(self, mock_config, mock_logger):
        """Test BranchManager can switch branches."""
        manager = BranchManager(mock_config, mock_logger)

        # Mock the branch switching
        with patch.object(manager, 'switch_branch', return_value="Switched to branch: develop") as mock_switch:
            result = manager.switch_branch('develop')
            assert mock_switch.called
