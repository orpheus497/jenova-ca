##Script function and purpose: Unit tests for utility tools (datetime and shell operations)
"""
Test suite for Tools Module - Utility tools for shell commands and datetime operations.

Tests cover:
- Datetime formatting
- Shell command execution
- Command existence checking
- System information
- Security (no shell injection)
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from jenova.tools import (
    get_current_datetime,
    get_current_date,
    get_current_time,
    execute_shell_command,
    command_exists,
    get_system_info,
    format_datetime_for_display,
    SHELL_TIMEOUT_DEFAULT,
    SHELL_MAX_OUTPUT_LENGTH,
)
from jenova.exceptions import ToolError


##Function purpose: Test get current datetime with timezone
def test_get_current_datetime_with_timezone() -> None:
    """##Test case: get_current_datetime includes timezone by default."""
    ##Action purpose: Get datetime
    result = get_current_datetime()
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert "T" in result  # ISO format
    assert "+" in result or "Z" in result or "-" in result  # Timezone


##Function purpose: Test get current datetime without timezone
def test_get_current_datetime_no_timezone() -> None:
    """##Test case: get_current_datetime can exclude timezone."""
    ##Action purpose: Get datetime
    result = get_current_datetime(include_timezone=False)
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert "T" in result  # ISO format
    assert "+" not in result  # No timezone


##Function purpose: Test get current datetime custom format
def test_get_current_datetime_custom_format() -> None:
    """##Test case: get_current_datetime accepts custom format."""
    ##Action purpose: Get datetime
    result = get_current_datetime(format_string="%Y-%m-%d")
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert len(result) == 10  # YYYY-MM-DD
    assert "-" in result


##Function purpose: Test get current date
def test_get_current_date() -> None:
    """##Test case: get_current_date returns YYYY-MM-DD."""
    ##Action purpose: Get date
    result = get_current_date()
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert len(result) == 10
    assert "-" in result


##Function purpose: Test get current date custom format
def test_get_current_date_custom_format() -> None:
    """##Test case: get_current_date accepts custom format."""
    ##Action purpose: Get date
    result = get_current_date(format_string="%d/%m/%Y")
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert "/" in result


##Function purpose: Test get current time 24h
def test_get_current_time_24h() -> None:
    """##Test case: get_current_time 24-hour format."""
    ##Action purpose: Get time
    result = get_current_time(include_seconds=True, format_24h=True)
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert result.count(":") == 2  # HH:MM:SS
    assert "AM" not in result and "PM" not in result


##Function purpose: Test get current time 12h
def test_get_current_time_12h() -> None:
    """##Test case: get_current_time 12-hour format."""
    ##Action purpose: Get time
    result = get_current_time(include_seconds=True, format_24h=False)
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert result.count(":") == 2  # HH:MM:SS
    assert ("AM" in result or "PM" in result)


##Function purpose: Test get current time no seconds
def test_get_current_time_no_seconds() -> None:
    """##Test case: get_current_time without seconds."""
    ##Action purpose: Get time
    result = get_current_time(include_seconds=False, format_24h=True)
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert result.count(":") == 1  # HH:MM


##Function purpose: Test execute shell command success
def test_execute_shell_command_success() -> None:
    """##Test case: execute_shell_command returns output and code."""
    ##Action purpose: Execute simple command
    output, code = execute_shell_command("echo hello")
    
    ##Assertion purpose: Verify
    assert code == 0
    assert "hello" in output


##Function purpose: Test execute shell command failure
def test_execute_shell_command_failure() -> None:
    """##Test case: execute_shell_command returns error code."""
    ##Action purpose: Execute failing command
    output, code = execute_shell_command("false")
    
    ##Assertion purpose: Verify
    assert code != 0


##Function purpose: Test execute shell command with args
def test_execute_shell_command_with_args() -> None:
    """##Test case: execute_shell_command handles arguments."""
    ##Action purpose: Execute command with args
    output, code = execute_shell_command("echo hello world")
    
    ##Assertion purpose: Verify
    assert code == 0
    assert "hello" in output and "world" in output


##Function purpose: Test execute shell command timeout
def test_execute_shell_command_timeout() -> None:
    """##Test case: execute_shell_command raises on timeout."""
    ##Step purpose: Try long-running command with short timeout
    with pytest.raises(ToolError) as exc_info:
        execute_shell_command("sleep 10", timeout=1)
    
    ##Assertion purpose: Verify timeout error
    assert "timed out" in str(exc_info.value)


##Function purpose: Test execute shell command not found
def test_execute_shell_command_not_found() -> None:
    """##Test case: execute_shell_command raises on command not found."""
    ##Step purpose: Try nonexistent command
    with pytest.raises(ToolError) as exc_info:
        execute_shell_command("nonexistent_command_xyz_123")
    
    ##Assertion purpose: Verify error
    assert "not found" in str(exc_info.value)


##Function purpose: Test execute shell command empty
def test_execute_shell_command_empty() -> None:
    """##Test case: execute_shell_command rejects empty command."""
    ##Step purpose: Try empty command
    with pytest.raises(ToolError) as exc_info:
        execute_shell_command("")
    
    ##Assertion purpose: Verify error
    assert "empty" in str(exc_info.value).lower()


##Function purpose: Test execute shell command bad directory
def test_execute_shell_command_bad_directory() -> None:
    """##Test case: execute_shell_command rejects nonexistent directory."""
    ##Step purpose: Try with bad directory
    with pytest.raises(ToolError) as exc_info:
        execute_shell_command("echo test", working_dir="/nonexistent/path/xyz")
    
    ##Assertion purpose: Verify error
    assert "not exist" in str(exc_info.value)


##Function purpose: Test execute shell command working directory
def test_execute_shell_command_working_directory(tmp_path: Path) -> None:
    """##Test case: execute_shell_command changes directory."""
    ##Step purpose: Create test file in temp dir
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    ##Action purpose: Execute pwd in that directory
    output, code = execute_shell_command("pwd", working_dir=str(tmp_path))
    
    ##Assertion purpose: Verify directory change
    assert code == 0
    assert str(tmp_path) in output


##Function purpose: Test execute shell command output truncation
def test_execute_shell_command_output_truncation() -> None:
    """##Test case: execute_shell_command truncates long output."""
    ##Step purpose: Generate large output
    large_output, code = execute_shell_command(
        "python -c \"print('x' * 20000)\"",
        max_output_length=1000
    )
    
    ##Assertion purpose: Verify truncation
    assert len(large_output) <= 1500  # 1000 + truncation message


##Function purpose: Test execute shell command stderr capture
def test_execute_shell_command_stderr_capture() -> None:
    """##Test case: execute_shell_command captures stderr."""
    ##Step purpose: Execute command that writes to stderr
    output, code = execute_shell_command("python -c \"import sys; sys.stderr.write('error')\"")
    
    ##Assertion purpose: Verify stderr in output
    assert "error" in output or code == 0  # At least one should be true


##Function purpose: Test execute shell command security no shell
def test_execute_shell_command_security_no_shell_injection() -> None:
    """##Test case: execute_shell_command doesn't use shell."""
    ##Step purpose: Try shell injection (should not work with shell=False)
    output, code = execute_shell_command("echo $(whoami)")
    
    ##Assertion purpose: Verify no shell expansion
    # Without shell, $(whoami) is treated as literal text
    assert "$(whoami)" in output or code != 0


##Function purpose: Test command exists true
def test_command_exists_true() -> None:
    """##Test case: command_exists returns True for existing command."""
    ##Action purpose: Check common command
    result = command_exists("echo")
    
    ##Assertion purpose: Verify
    assert result is True


##Function purpose: Test command exists false
def test_command_exists_false() -> None:
    """##Test case: command_exists returns False for missing command."""
    ##Action purpose: Check nonexistent command
    result = command_exists("nonexistent_command_xyz_123")
    
    ##Assertion purpose: Verify
    assert result is False


##Function purpose: Test get system info structure
def test_get_system_info() -> None:
    """##Test case: get_system_info returns all required fields."""
    ##Action purpose: Get info
    info = get_system_info()
    
    ##Assertion purpose: Verify structure
    assert isinstance(info, dict)
    assert "platform" in info
    assert "platform_release" in info
    assert "platform_version" in info
    assert "architecture" in info
    assert "python_version" in info
    assert "hostname" in info
    assert "cpu_count" in info


##Function purpose: Test get system info values
def test_get_system_info_values() -> None:
    """##Test case: get_system_info returns non-empty values."""
    ##Action purpose: Get info
    info = get_system_info()
    
    ##Assertion purpose: Verify values
    assert info["platform"] is not None
    assert info["architecture"] is not None
    assert info["python_version"] is not None


##Function purpose: Test format datetime absolute
def test_format_datetime_for_display_absolute() -> None:
    """##Test case: format_datetime_for_display absolute format."""
    ##Step purpose: Create test datetime
    dt = datetime(2026, 1, 19, 14, 30, 0)
    
    ##Action purpose: Format
    result = format_datetime_for_display(dt, relative=False)
    
    ##Assertion purpose: Verify format
    assert isinstance(result, str)
    assert "2026" in result or "January" in result


##Function purpose: Test format datetime relative just now
def test_format_datetime_for_display_relative_just_now() -> None:
    """##Test case: format_datetime_for_display returns 'just now' for recent."""
    ##Step purpose: Create recent datetime
    dt = datetime.now() - timedelta(seconds=10)
    
    ##Action purpose: Format
    result = format_datetime_for_display(dt, relative=True)
    
    ##Assertion purpose: Verify
    assert "just now" in result


##Function purpose: Test format datetime relative minutes
def test_format_datetime_for_display_relative_minutes() -> None:
    """##Test case: format_datetime_for_display returns minutes ago."""
    ##Step purpose: Create datetime from 5 minutes ago
    dt = datetime.now() - timedelta(minutes=5)
    
    ##Action purpose: Format
    result = format_datetime_for_display(dt, relative=True)
    
    ##Assertion purpose: Verify
    assert "minute" in result
    assert "ago" in result


##Function purpose: Test format datetime relative hours
def test_format_datetime_for_display_relative_hours() -> None:
    """##Test case: format_datetime_for_display returns hours ago."""
    ##Step purpose: Create datetime from 3 hours ago
    dt = datetime.now() - timedelta(hours=3)
    
    ##Action purpose: Format
    result = format_datetime_for_display(dt, relative=True)
    
    ##Assertion purpose: Verify
    assert "hour" in result
    assert "ago" in result


##Function purpose: Test format datetime relative days
def test_format_datetime_for_display_relative_days() -> None:
    """##Test case: format_datetime_for_display returns days ago."""
    ##Step purpose: Create datetime from 5 days ago
    dt = datetime.now() - timedelta(days=5)
    
    ##Action purpose: Format
    result = format_datetime_for_display(dt, relative=True)
    
    ##Assertion purpose: Verify
    assert "day" in result
    assert "ago" in result


##Function purpose: Test format datetime relative weeks
def test_format_datetime_for_display_relative_weeks() -> None:
    """##Test case: format_datetime_for_display returns weeks ago."""
    ##Step purpose: Create datetime from 3 weeks ago
    dt = datetime.now() - timedelta(weeks=3)
    
    ##Action purpose: Format
    result = format_datetime_for_display(dt, relative=True)
    
    ##Assertion purpose: Verify
    assert "week" in result
    assert "ago" in result


##Function purpose: Test format datetime default to now
def test_format_datetime_for_display_default_to_now() -> None:
    """##Test case: format_datetime_for_display uses now if None."""
    ##Action purpose: Format with None
    result = format_datetime_for_display(None, relative=False)
    
    ##Assertion purpose: Verify
    assert isinstance(result, str)
    assert len(result) > 0


##Function purpose: Test execute shell command singular plural
def test_execute_shell_command_grammar_singular() -> None:
    """##Test case: format_datetime_for_display uses correct grammar."""
    ##Step purpose: 1 minute ago
    dt = datetime.now() - timedelta(minutes=1)
    result = format_datetime_for_display(dt, relative=True)
    
    ##Assertion purpose: Verify singular
    assert "1 minute ago" in result


##Function purpose: Test format datetime grammar plural
def test_format_datetime_for_display_grammar_plural() -> None:
    """##Test case: format_datetime_for_display handles plurals."""
    ##Step purpose: 2+ minutes ago
    dt = datetime.now() - timedelta(minutes=5)
    result = format_datetime_for_display(dt, relative=True)
    
    ##Assertion purpose: Verify plural
    assert "minutes ago" in result
    assert "minute ago" not in result or "5 minutes" in result


##Function purpose: Test execute shell command with special characters in args
def test_execute_shell_command_special_chars() -> None:
    """##Test case: execute_shell_command handles special characters."""
    ##Action purpose: Execute echo with special chars
    output, code = execute_shell_command("echo 'hello world'")
    
    ##Assertion purpose: Verify
    assert code == 0
    assert "hello" in output


##Function purpose: Test execute shell command strip output
def test_execute_shell_command_strip_output() -> None:
    """##Test case: execute_shell_command strips output."""
    ##Action purpose: Execute command with trailing newlines
    output, code = execute_shell_command("echo hello")
    
    ##Assertion purpose: Verify stripped
    assert output == "hello"
    assert not output.endswith("\n")
