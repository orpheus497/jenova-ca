# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Tools Module Tests

"""
Unit tests for tools module.

Tests the specialized tool implementations and centralized tool handler,
ensuring correct operation, error handling, and security controls.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from jenova.tools.base import BaseTool, ToolResult, ToolError
from jenova.tools.time_tools import TimeTools
from jenova.tools.shell_tools import ShellTools
from jenova.tools.web_tools import WebTools
from jenova.tools.file_tools import FileTools
from jenova.tools.tool_handler import ToolHandler


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful result creation."""
        result = ToolResult(success=True, data="test_data")

        assert result.success is True
        assert result.data == "test_data"
        assert result.error is None

    def test_error_result(self):
        """Test error result creation."""
        result = ToolResult(success=False, error="test_error")

        assert result.success is False
        assert result.error == "test_error"
        assert result.data is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ToolResult(
            success=True,
            data="test",
            metadata={"key": "value"}
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["data"] == "test"
        assert result_dict["metadata"]["key"] == "value"


class TestToolError:
    """Tests for ToolError exception."""

    def test_error_creation(self):
        """Test ToolError creation with context."""
        error = ToolError(
            "test_tool",
            "Test error message",
            context={"detail": "Extra info"}
        )

        assert error.tool_name == "test_tool"
        assert error.message == "Test error message"
        assert error.context["detail"] == "Extra info"
        assert "test_tool" in str(error)


class TestTimeTools:
    """Tests for TimeTools class."""

    @pytest.fixture
    def time_tools(self):
        """Provide TimeTools instance."""
        config = {}
        ui_logger = Mock()
        file_logger = Mock()
        return TimeTools(config, ui_logger, file_logger)

    def test_execute_default_format(self, time_tools):
        """Test datetime retrieval with default format."""
        result = time_tools.execute()

        assert result.success is True
        assert isinstance(result.data, str)
        # Should match format: YYYY-MM-DD HH:MM:SS
        assert len(result.data.split()) == 2
        assert result.metadata['timezone'] is not None

    def test_execute_custom_format(self, time_tools):
        """Test datetime with custom format."""
        result = time_tools.execute(format_str="%Y-%m-%d")

        assert result.success is True
        # Should match format: YYYY-MM-DD
        assert len(result.data.split('-')) == 3

    def test_execute_invalid_timezone(self, time_tools):
        """Test fallback to UTC on invalid timezone."""
        result = time_tools.execute(timezone="Invalid/Timezone")

        # Should still succeed with UTC fallback
        assert result.success is True
        assert "UTC" in result.metadata['timezone']

    def test_get_current_datetime(self, time_tools):
        """Test convenience method for current datetime."""
        dt_string = time_tools.get_current_datetime()

        assert isinstance(dt_string, str)
        assert len(dt_string) > 0

    def test_get_timestamp(self, time_tools):
        """Test Unix timestamp generation."""
        timestamp = time_tools.get_timestamp()

        assert isinstance(timestamp, int)
        assert timestamp > 0
        # Should be recent (within a year of 2025-11-08)
        assert timestamp > 1700000000

    def test_format_datetime(self, time_tools):
        """Test datetime formatting."""
        dt = datetime(2025, 11, 8, 14, 30, 0)
        formatted = time_tools.format_datetime(dt, "%B %d, %Y")

        assert formatted == "November 08, 2025"


class TestShellTools:
    """Tests for ShellTools class."""

    @pytest.fixture
    def shell_tools(self):
        """Provide ShellTools instance."""
        config = {'tools': {'shell_command_whitelist': ['echo', 'ls', 'pwd']}}
        ui_logger = Mock()
        file_logger = Mock()
        return ShellTools(config, ui_logger, file_logger)

    def test_whitelisted_command(self, shell_tools):
        """Test execution of whitelisted command."""
        result = shell_tools.execute("echo Hello")

        assert result.success is True
        assert "Hello" in result.data

    def test_command_not_whitelisted(self, shell_tools):
        """Test rejection of non-whitelisted command."""
        result = shell_tools.execute("rm -rf /")

        assert result.success is False
        assert "not allowed" in result.error

    def test_is_command_allowed(self, shell_tools):
        """Test whitelist checking."""
        assert shell_tools.is_command_allowed("echo") is True
        assert shell_tools.is_command_allowed("rm") is False

    def test_get_whitelist(self, shell_tools):
        """Test whitelist retrieval."""
        whitelist = shell_tools.get_whitelist()

        assert isinstance(whitelist, list)
        assert "echo" in whitelist
        assert "ls" in whitelist

    def test_command_with_args(self, shell_tools):
        """Test command with arguments."""
        result = shell_tools.execute("echo test argument")

        assert result.success is True
        assert "test" in result.data
        assert "argument" in result.data

    def test_timeout_parameter(self, shell_tools):
        """Test timeout parameter is accepted."""
        # Just verify it doesn't raise an exception
        result = shell_tools.execute("echo quick", timeout=5)

        assert result.success is True


class TestFileTools:
    """Tests for FileTools class."""

    @pytest.fixture
    def temp_dir(self):
        """Provide temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def file_tools(self, temp_dir):
        """Provide FileTools instance with sandbox."""
        config = {}
        ui_logger = Mock()
        file_logger = Mock()
        return FileTools(
            config,
            ui_logger,
            file_logger,
            sandbox_root=temp_dir
        )

    def test_write_and_read_file(self, file_tools, temp_dir):
        """Test writing and reading a file."""
        test_file = temp_dir / "test.txt"
        content = "Hello, JENOVA!"

        # Write file
        write_result = file_tools.execute('write', test_file, content=content)
        assert write_result.success is True

        # Read file
        read_result = file_tools.execute('read', test_file)
        assert read_result.success is True
        assert read_result.data == content

    def test_file_exists(self, file_tools, temp_dir):
        """Test file existence check."""
        test_file = temp_dir / "exists.txt"
        test_file.write_text("content")

        result = file_tools.execute('exists', test_file)

        assert result.success is True
        assert result.data is True
        assert result.metadata['is_file'] is True

    def test_file_not_exists(self, file_tools, temp_dir):
        """Test non-existent file check."""
        test_file = temp_dir / "doesnotexist.txt"

        result = file_tools.execute('exists', test_file)

        assert result.success is True
        assert result.data is False

    def test_list_directory(self, file_tools, temp_dir):
        """Test directory listing."""
        # Create test files
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")

        result = file_tools.execute('list', temp_dir)

        assert result.success is True
        assert "file1.txt" in result.data
        assert "file2.txt" in result.data
        assert result.metadata['count'] == 2

    def test_get_file_info(self, file_tools, temp_dir):
        """Test file metadata retrieval."""
        test_file = temp_dir / "info.txt"
        test_file.write_text("test content")

        result = file_tools.execute('info', test_file)

        assert result.success is True
        assert result.data['name'] == "info.txt"
        assert result.data['is_file'] is True
        assert result.data['size_bytes'] > 0

    def test_create_directory(self, file_tools, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "newdir"

        result = file_tools.execute('mkdir', new_dir)

        assert result.success is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_delete_file(self, file_tools, temp_dir):
        """Test file deletion with confirmation."""
        test_file = temp_dir / "delete.txt"
        test_file.write_text("delete me")

        result = file_tools.execute('delete', test_file, confirm=True)

        assert result.success is True
        assert not test_file.exists()

    def test_delete_without_confirmation(self, file_tools, temp_dir):
        """Test deletion requires confirmation."""
        test_file = temp_dir / "nodelete.txt"
        test_file.write_text("keep me")

        result = file_tools.execute('delete', test_file, confirm=False)

        assert result.success is False
        assert "confirm" in result.error.lower()
        assert test_file.exists()

    def test_sandbox_validation(self, file_tools, temp_dir):
        """Test sandbox prevents access outside root."""
        outside_path = Path("/tmp/outside_sandbox.txt")

        result = file_tools.execute('read', outside_path)

        assert result.success is False
        assert "validation failed" in result.error.lower()

    def test_file_size_limit(self, file_tools, temp_dir):
        """Test file size limit enforcement."""
        # Create FileTools with very small limit
        config = {}
        file_tools_limited = FileTools(
            config,
            Mock(),
            Mock(),
            sandbox_root=temp_dir,
            max_file_size_mb=0.001  # 1 KB limit
        )

        # Try to write large content (>1KB)
        large_content = "x" * 2000

        result = file_tools_limited.write_file(
            temp_dir / "large.txt",
            large_content
        )

        assert result.success is False
        assert "too large" in result.error.lower()


class TestWebTools:
    """Tests for WebTools class."""

    @pytest.fixture
    def web_tools(self):
        """Provide WebTools instance."""
        config = {}
        ui_logger = Mock()
        file_logger = Mock()
        return WebTools(config, ui_logger, file_logger)

    def test_selenium_not_available(self, web_tools):
        """Test graceful degradation when selenium unavailable."""
        # Patch selenium availability
        web_tools.selenium_available = False

        result = web_tools.execute("test query")

        assert result.success is False
        assert "selenium" in result.error.lower() or "install" in result.error.lower()

    @patch('jenova.tools.web_tools.webdriver')
    def test_execute_with_query(self, mock_webdriver, web_tools):
        """Test web search execution (mocked)."""
        # This test would require mocking selenium WebDriver
        # For now, just verify the method signature
        pass  # Complex selenium mocking - skip for basic test


class TestToolHandler:
    """Tests for ToolHandler class."""

    @pytest.fixture
    def tool_handler(self):
        """Provide ToolHandler instance."""
        config = {'tools': {'shell_command_whitelist': ['echo', 'ls']}}
        ui_logger = Mock()
        file_logger = Mock()
        return ToolHandler(config, ui_logger, file_logger)

    def test_initialization(self, tool_handler):
        """Test handler initializes all tools."""
        tools = tool_handler.list_tools()

        assert 'time_tools' in tools
        assert 'shell_tools' in tools
        assert 'web_tools' in tools
        assert 'file_tools' in tools

    def test_execute_time_tool(self, tool_handler):
        """Test executing time tool through handler."""
        result = tool_handler.execute_tool('time_tools', format_str="%Y-%m-%d")

        assert result.success is True
        assert isinstance(result.data, str)

    def test_execute_nonexistent_tool(self, tool_handler):
        """Test error on non-existent tool."""
        with pytest.raises(ToolError) as exc_info:
            tool_handler.execute_tool('nonexistent_tool')

        assert "not found" in str(exc_info.value)

    def test_get_tool(self, tool_handler):
        """Test direct tool retrieval."""
        time_tool = tool_handler.get_tool('time_tools')

        assert isinstance(time_tool, TimeTools)

    def test_get_tool_info(self, tool_handler):
        """Test tool metadata retrieval."""
        info = tool_handler.get_tool_info('time_tools')

        assert info['name'] == 'time_tools'
        assert 'description' in info
        assert info['type'] == 'TimeTools'

    def test_get_all_tools_info(self, tool_handler):
        """Test all tools metadata retrieval."""
        all_info = tool_handler.get_all_tools_info()

        assert len(all_info) == 4
        assert 'time_tools' in all_info
        assert 'shell_tools' in all_info

    def test_is_tool_available(self, tool_handler):
        """Test tool availability check."""
        assert tool_handler.is_tool_available('time_tools') is True
        assert tool_handler.is_tool_available('fake_tool') is False

    def test_register_custom_tool(self, tool_handler):
        """Test registering custom tool."""
        # Create a simple custom tool
        class CustomTool(BaseTool):
            def execute(self, **kwargs):
                return ToolResult(success=True, data="custom")

        custom = CustomTool("custom", "Custom tool", {}, Mock(), Mock())
        tool_handler.register_tool('custom_tool', custom)

        assert tool_handler.is_tool_available('custom_tool') is True

    def test_unregister_tool(self, tool_handler):
        """Test unregistering a tool."""
        # Register then unregister
        class CustomTool(BaseTool):
            def execute(self, **kwargs):
                return ToolResult(success=True, data="custom")

        custom = CustomTool("custom", "Custom tool", {}, Mock(), Mock())
        tool_handler.register_tool('custom_tool', custom)

        assert tool_handler.unregister_tool('custom_tool') is True
        assert tool_handler.is_tool_available('custom_tool') is False

    def test_unregister_nonexistent_tool(self, tool_handler):
        """Test unregistering non-existent tool returns False."""
        result = tool_handler.unregister_tool('nonexistent')

        assert result is False


# Integration test
def test_tool_handler_integration():
    """Integration test for full tool handler workflow."""
    config = {'tools': {'shell_command_whitelist': ['echo']}}
    ui_logger = Mock()
    file_logger = Mock()

    # Create handler
    handler = ToolHandler(config, ui_logger, file_logger)

    # Execute multiple tools
    time_result = handler.execute_tool('time_tools')
    assert time_result.success is True

    shell_result = handler.execute_tool('shell_tools', command='echo test')
    assert shell_result.success is True

    # Get tool info
    all_info = handler.get_all_tools_info()
    assert len(all_info) >= 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
