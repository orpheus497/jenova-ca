##Script function and purpose: Utility tools for shell commands and datetime operations
"""
Tools Module - Utility tools for JENOVA operations.

This module provides utility functions for common operations like
datetime handling and shell command execution.

Reference: .devdocs/resources/src/jenova/tools.py
"""

from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess

import structlog

from jenova.exceptions import ToolError

##Class purpose: Define logger for tool operations
logger = structlog.get_logger(__name__)

##Step purpose: Configuration constants
SHELL_TIMEOUT_DEFAULT = 30
SHELL_MAX_OUTPUT_LENGTH = 10000


##Function purpose: Get current datetime in standard format
def get_current_datetime(
    include_timezone: bool = True,
    format_string: str | None = None,
) -> str:
    """Get the current datetime as a formatted string.
    
    Args:
        include_timezone: Include timezone in output (default True)
        format_string: Custom format string (default ISO format)
        
    Returns:
        Formatted datetime string
        
    Example:
        >>> get_current_datetime()
        '2026-01-19T14:30:00+00:00'
    """
    ##Step purpose: Get current time with or without timezone
    if include_timezone:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()
    
    ##Condition purpose: Use custom format or ISO
    if format_string:
        return now.strftime(format_string)
    else:
        return now.isoformat()


##Function purpose: Get current date only
def get_current_date(format_string: str = "%Y-%m-%d") -> str:
    """Get the current date as a formatted string.
    
    Args:
        format_string: Date format string (default YYYY-MM-DD)
        
    Returns:
        Formatted date string
        
    Example:
        >>> get_current_date()
        '2026-01-19'
    """
    return datetime.now().strftime(format_string)


##Function purpose: Get current time only
def get_current_time(
    include_seconds: bool = True,
    format_24h: bool = True,
) -> str:
    """Get the current time as a formatted string.
    
    Args:
        include_seconds: Include seconds in output
        format_24h: Use 24-hour format (default True)
        
    Returns:
        Formatted time string
        
    Example:
        >>> get_current_time()
        '14:30:00'
    """
    ##Step purpose: Build format string
    if format_24h:
        fmt = "%H:%M"
    else:
        fmt = "%I:%M %p"
    
    ##Condition purpose: Add seconds if requested
    if include_seconds:
        if format_24h:
            fmt = "%H:%M:%S"
        else:
            fmt = "%I:%M:%S %p"
    
    return datetime.now().strftime(fmt)


##Function purpose: Execute a shell command safely
def execute_shell_command(
    command: str,
    timeout: int = SHELL_TIMEOUT_DEFAULT,
    working_dir: Path | str | None = None,
    capture_stderr: bool = True,
    max_output_length: int = SHELL_MAX_OUTPUT_LENGTH,
) -> tuple[str, int]:
    """Execute a shell command safely with timeout and output limits.
    
    Args:
        command: The shell command to execute
        timeout: Timeout in seconds (default 30)
        working_dir: Working directory for command (optional)
        capture_stderr: Capture stderr in output (default True)
        max_output_length: Maximum output length to return
        
    Returns:
        Tuple of (output, return_code)
        
    Raises:
        ToolError: If command execution fails
        
    Example:
        >>> output, code = execute_shell_command("echo hello")
        >>> print(output)
        'hello'
        
    Security:
        Commands are NOT executed through a shell to prevent injection.
        Arguments are properly parsed with shlex.
    """
    ##Step purpose: Parse command into arguments safely
    try:
        args = shlex.split(command)
    except ValueError as e:
        raise ToolError(f"Invalid command syntax: {e}") from e
    
    ##Condition purpose: Validate command is not empty
    if not args:
        raise ToolError("Empty command")
    
    ##Step purpose: Prepare working directory
    cwd: Path | None = None
    if working_dir:
        cwd = Path(working_dir)
        ##Condition purpose: Validate directory exists
        if not cwd.is_dir():
            raise ToolError(f"Working directory does not exist: {cwd}")
    
    ##Step purpose: Configure stderr handling
    stderr = subprocess.STDOUT if capture_stderr else subprocess.DEVNULL
    
    ##Error purpose: Execute command with error handling
    try:
        logger.debug(
            "shell_execute",
            command=args[0],
            arg_count=len(args) - 1,
            working_dir=str(cwd) if cwd else None,
        )
        
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            shell=False,  # Security: never use shell=True
        )
        
        ##Step purpose: Combine output
        output = result.stdout
        if capture_stderr and result.stderr:
            output = output + "\n" + result.stderr if output else result.stderr
        
        ##Step purpose: Truncate if too long
        if len(output) > max_output_length:
            output = output[:max_output_length] + "\n... (output truncated)"
        
        logger.debug(
            "shell_complete",
            command=args[0],
            return_code=result.returncode,
            output_length=len(output),
        )
        
        return (output.strip(), result.returncode)
        
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout} seconds")
    except FileNotFoundError:
        raise ToolError(f"Command not found: {args[0]}")
    except PermissionError:
        raise ToolError(f"Permission denied: {args[0]}")
    except OSError as e:
        raise ToolError(f"Command execution failed: {e}") from e


##Function purpose: Check if a command exists
def command_exists(command_name: str) -> bool:
    """Check if a command exists in PATH.
    
    Args:
        command_name: Name of the command to check
        
    Returns:
        True if command exists, False otherwise
        
    Example:
        >>> command_exists("git")
        True
    """
    try:
        result = subprocess.run(
            ["which", command_name],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


##Function purpose: Get system information
def get_system_info() -> dict[str, str]:
    """Get basic system information.
    
    Returns:
        Dictionary with system information
        
    Example:
        >>> info = get_system_info()
        >>> print(info["platform"])
        'freebsd'
    """
    import platform
    import os
    
    return {
        "platform": platform.system().lower(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu_count": str(os.cpu_count() or "unknown"),
    }


##Function purpose: Format a datetime for user display
def format_datetime_for_display(
    dt: datetime | None = None,
    relative: bool = False,
) -> str:
    """Format a datetime for user-friendly display.
    
    Args:
        dt: Datetime to format (default now)
        relative: Use relative time (e.g., "2 hours ago")
        
    Returns:
        Formatted datetime string
        
    Example:
        >>> format_datetime_for_display(relative=True)
        'just now'
    """
    ##Step purpose: Use current time if not provided
    if dt is None:
        dt = datetime.now()
    
    ##Condition purpose: Format relative time
    if relative:
        now = datetime.now()
        diff = now - dt
        
        ##Step purpose: Calculate relative time
        seconds = diff.total_seconds()
        
        ##Condition purpose: Handle different time ranges
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            weeks = int(seconds / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    else:
        ##Step purpose: Use absolute format
        return dt.strftime("%B %d, %Y at %I:%M %p")
