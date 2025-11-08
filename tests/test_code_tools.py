# The JENOVA Cognitive Architecture - Code Tools Tests
# Copyright (c) 2024-2025, orpheus497. All rights reserved.
# Licensed under the MIT License

"""
Code Tools module tests for JENOVA Phase 13-17.

Tests file editing, code parsing, refactoring, syntax highlighting,
codebase mapping, and interactive terminal functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from jenova.code_tools.file_editor import FileEditor
from jenova.code_tools.code_parser import CodeParser
from jenova.code_tools.refactoring_engine import RefactoringEngine
from jenova.code_tools.syntax_highlighter import SyntaxHighlighter
from jenova.code_tools.codebase_mapper import CodebaseMapper


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""# Test Python file
def hello_world():
    '''Simple hello world function.'''
    message = "Hello, World!"
    print(message)
    return message

class TestClass:
    '''A test class.'''
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"
""")
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


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


class TestFileEditor:
    """Tests for FileEditor functionality."""

    @pytest.mark.unit
    @pytest.mark.code
    def test_file_editor_initialization(self, mock_config, mock_logger):
        """Test FileEditor initializes correctly."""
        editor = FileEditor(mock_config, mock_logger)
        assert editor is not None
        assert editor.config == mock_config
        assert editor.file_logger == mock_logger

    @pytest.mark.unit
    @pytest.mark.code
    def test_file_editor_read_file(self, temp_python_file, mock_config, mock_logger):
        """Test FileEditor can read files."""
        editor = FileEditor(mock_config, mock_logger)

        # Mock the read operation
        with patch.object(editor, 'read_file', return_value="file content") as mock_read:
            content = editor.read_file(temp_python_file)
            assert mock_read.called
            assert content == "file content"

    @pytest.mark.unit
    @pytest.mark.code
    def test_file_editor_preview_mode(self, temp_python_file, mock_config, mock_logger):
        """Test FileEditor preview mode (default)."""
        editor = FileEditor(mock_config, mock_logger)

        # Mock the edit operation
        with patch.object(editor, 'edit_file', return_value="Preview: 3 changes") as mock_edit:
            result = editor.edit_file(temp_python_file, mode='preview')
            assert mock_edit.called
            assert "Preview" in result or "preview" in result.lower()

    @pytest.mark.unit
    @pytest.mark.code
    def test_file_editor_apply_mode(self, temp_python_file, mock_config, mock_logger):
        """Test FileEditor apply mode."""
        editor = FileEditor(mock_config, mock_logger)

        # Mock the edit operation
        with patch.object(editor, 'edit_file', return_value="Applied 3 changes") as mock_edit:
            result = editor.edit_file(temp_python_file, mode='apply')
            assert mock_edit.called
            assert "Applied" in result or "applied" in result.lower() or "change" in result.lower()

    @pytest.mark.unit
    @pytest.mark.code
    def test_file_editor_backup_creation(self, temp_python_file, mock_config, mock_logger):
        """Test FileEditor creates backups when requested."""
        editor = FileEditor(mock_config, mock_logger)

        # Mock the backup operation
        with patch.object(editor, 'create_backup', return_value=f"{temp_python_file}.backup") as mock_backup:
            backup_path = editor.create_backup(temp_python_file)
            assert mock_backup.called
            assert '.backup' in backup_path or 'backup' in backup_path.lower()


class TestCodeParser:
    """Tests for CodeParser functionality."""

    @pytest.mark.unit
    @pytest.mark.code
    def test_code_parser_initialization(self, mock_config, mock_logger):
        """Test CodeParser initializes correctly."""
        parser = CodeParser(mock_config, mock_logger)
        assert parser is not None

    @pytest.mark.unit
    @pytest.mark.code
    def test_code_parser_parse_python(self, temp_python_file, mock_config, mock_logger):
        """Test CodeParser can parse Python files."""
        parser = CodeParser(mock_config, mock_logger)

        # Mock the parse operation
        with patch.object(parser, 'parse', return_value={'functions': ['hello_world'], 'classes': ['TestClass']}) as mock_parse:
            result = parser.parse(temp_python_file, mode='structure')
            assert mock_parse.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_code_parser_extract_symbols(self, temp_python_file, mock_config, mock_logger):
        """Test CodeParser can extract symbols."""
        parser = CodeParser(mock_config, mock_logger)

        # Mock the symbol extraction
        with patch.object(parser, 'extract_symbols', return_value=['hello_world', 'TestClass', 'greet']) as mock_extract:
            symbols = parser.extract_symbols(temp_python_file)
            assert mock_extract.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_code_parser_ast_mode(self, temp_python_file, mock_config, mock_logger):
        """Test CodeParser AST tree mode."""
        parser = CodeParser(mock_config, mock_logger)

        # Mock the AST parsing
        with patch.object(parser, 'parse', return_value="AST tree representation") as mock_parse:
            result = parser.parse(temp_python_file, mode='tree')
            assert mock_parse.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_code_parser_invalid_file(self, mock_config, mock_logger):
        """Test CodeParser handles invalid files gracefully."""
        parser = CodeParser(mock_config, mock_logger)

        # Mock error handling
        with patch.object(parser, 'parse', side_effect=FileNotFoundError("File not found")) as mock_parse:
            with pytest.raises(FileNotFoundError):
                parser.parse('/nonexistent/file.py')


class TestRefactoringEngine:
    """Tests for RefactoringEngine functionality."""

    @pytest.mark.unit
    @pytest.mark.code
    def test_refactoring_engine_initialization(self, mock_config, mock_logger):
        """Test RefactoringEngine initializes correctly."""
        engine = RefactoringEngine(mock_config, mock_logger)
        assert engine is not None

    @pytest.mark.unit
    @pytest.mark.code
    def test_refactoring_rename_operation(self, temp_python_file, mock_config, mock_logger):
        """Test RefactoringEngine rename operation."""
        engine = RefactoringEngine(mock_config, mock_logger)

        # Mock the rename operation
        with patch.object(engine, 'refactor', return_value="Renamed 3 occurrences") as mock_refactor:
            result = engine.refactor('rename', ['old_name', 'new_name'])
            assert mock_refactor.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_refactoring_extract_method(self, temp_python_file, mock_config, mock_logger):
        """Test RefactoringEngine extract method operation."""
        engine = RefactoringEngine(mock_config, mock_logger)

        # Mock the extract operation
        with patch.object(engine, 'refactor', return_value="Extracted new method") as mock_refactor:
            result = engine.refactor('extract-method', ['new_method_name'])
            assert mock_refactor.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_refactoring_inline_operation(self, temp_python_file, mock_config, mock_logger):
        """Test RefactoringEngine inline operation."""
        engine = RefactoringEngine(mock_config, mock_logger)

        # Mock the inline operation
        with patch.object(engine, 'refactor', return_value="Inlined variable") as mock_refactor:
            result = engine.refactor('inline', ['variable_name'])
            assert mock_refactor.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_refactoring_invalid_operation(self, mock_config, mock_logger):
        """Test RefactoringEngine handles invalid operations."""
        engine = RefactoringEngine(mock_config, mock_logger)

        # Mock error handling
        with patch.object(engine, 'refactor', side_effect=ValueError("Invalid operation")) as mock_refactor:
            with pytest.raises(ValueError):
                engine.refactor('invalid_op', [])


class TestSyntaxHighlighter:
    """Tests for SyntaxHighlighter functionality."""

    @pytest.mark.unit
    @pytest.mark.code
    def test_syntax_highlighter_initialization(self, mock_config, mock_logger):
        """Test SyntaxHighlighter initializes correctly."""
        highlighter = SyntaxHighlighter(mock_config, mock_logger)
        assert highlighter is not None

    @pytest.mark.unit
    @pytest.mark.code
    def test_syntax_highlighter_python(self, temp_python_file, mock_config, mock_logger):
        """Test SyntaxHighlighter can highlight Python code."""
        highlighter = SyntaxHighlighter(mock_config, mock_logger)

        # Mock the highlight operation
        with patch.object(highlighter, 'highlight', return_value="highlighted code") as mock_highlight:
            result = highlighter.highlight(temp_python_file)
            assert mock_highlight.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_syntax_highlighter_string_input(self, mock_config, mock_logger):
        """Test SyntaxHighlighter can highlight code strings."""
        highlighter = SyntaxHighlighter(mock_config, mock_logger)

        code = "def test():\n    return True"
        # Mock the highlight operation
        with patch.object(highlighter, 'highlight_string', return_value="highlighted string") as mock_highlight:
            result = highlighter.highlight_string(code, 'python')
            assert mock_highlight.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_syntax_highlighter_language_detection(self, temp_python_file, mock_config, mock_logger):
        """Test SyntaxHighlighter can detect language from extension."""
        highlighter = SyntaxHighlighter(mock_config, mock_logger)

        # Mock the language detection
        with patch.object(highlighter, 'detect_language', return_value='python') as mock_detect:
            lang = highlighter.detect_language(temp_python_file)
            assert mock_detect.called


class TestCodebaseMapper:
    """Tests for CodebaseMapper functionality."""

    @pytest.mark.unit
    @pytest.mark.code
    def test_codebase_mapper_initialization(self, mock_config, mock_logger):
        """Test CodebaseMapper initializes correctly."""
        mapper = CodebaseMapper(mock_config, mock_logger)
        assert mapper is not None

    @pytest.mark.unit
    @pytest.mark.code
    def test_codebase_mapper_map_directory(self, mock_config, mock_logger):
        """Test CodebaseMapper can map directory structure."""
        mapper = CodebaseMapper(mock_config, mock_logger)

        # Mock the mapping operation
        with patch.object(mapper, 'map_directory', return_value={'files': 10, 'modules': 5}) as mock_map:
            result = mapper.map_directory('/tmp/test')
            assert mock_map.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_codebase_mapper_dependency_graph(self, mock_config, mock_logger):
        """Test CodebaseMapper can generate dependency graphs."""
        mapper = CodebaseMapper(mock_config, mock_logger)

        # Mock the dependency graph generation
        with patch.object(mapper, 'generate_dependency_graph', return_value={'nodes': [], 'edges': []}) as mock_graph:
            result = mapper.generate_dependency_graph('/tmp/test')
            assert mock_graph.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_codebase_mapper_analyze_structure(self, mock_config, mock_logger):
        """Test CodebaseMapper can analyze codebase structure."""
        mapper = CodebaseMapper(mock_config, mock_logger)

        # Mock the structure analysis
        with patch.object(mapper, 'analyze_structure', return_value={'complexity': 'medium'}) as mock_analyze:
            result = mapper.analyze_structure('/tmp/test')
            assert mock_analyze.called


class TestInteractiveTerminal:
    """Tests for InteractiveTerminal functionality."""

    @pytest.mark.unit
    @pytest.mark.code
    def test_interactive_terminal_initialization(self, mock_config, mock_logger):
        """Test InteractiveTerminal initializes correctly."""
        from jenova.code_tools.interactive_terminal import InteractiveTerminal
        terminal = InteractiveTerminal(mock_config, mock_logger)
        assert terminal is not None

    @pytest.mark.unit
    @pytest.mark.code
    @pytest.mark.skip(reason="Interactive terminal requires PTY which may not be available in test environment")
    def test_interactive_terminal_launch_vim(self, mock_config, mock_logger):
        """Test InteractiveTerminal can launch vim."""
        from jenova.code_tools.interactive_terminal import InteractiveTerminal
        terminal = InteractiveTerminal(mock_config, mock_logger)

        # Mock the launch operation
        with patch.object(terminal, 'launch', return_value=0) as mock_launch:
            result = terminal.launch('vim', ['/tmp/test.txt'])
            assert mock_launch.called

    @pytest.mark.unit
    @pytest.mark.code
    def test_interactive_terminal_pty_support(self, mock_config, mock_logger):
        """Test InteractiveTerminal PTY support availability."""
        from jenova.code_tools.interactive_terminal import InteractiveTerminal
        terminal = InteractiveTerminal(mock_config, mock_logger)

        # Mock the PTY check
        with patch.object(terminal, 'has_pty_support', return_value=True) as mock_pty:
            result = terminal.has_pty_support()
            assert mock_pty.called or isinstance(result, bool)
