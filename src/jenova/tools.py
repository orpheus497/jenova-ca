##Script function and purpose: Tool Functions - Utility functions for JENOVA including datetime and shell commands
##Dependency purpose: Provides utility functions that JENOVA can use including datetime and shell command execution
"""Tool Functions for JENOVA.

This module provides utility functions that JENOVA can use including:
- Current date and time retrieval
- Safe shell command execution
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime

import structlog

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Result from shell command execution
@dataclass
class ShellCommandResult:
    """Result from shell command execution.
    
    Attributes:
        stdout: Standard output from command.
        stderr: Standard error from command.
        returncode: Return code from command.
        error: Whether command had an error (returncode != 0).
    """
    
    stdout: str
    stderr: str
    returncode: int
    error: bool


##Function purpose: Return the current date and time in ISO 8601 format
def get_current_datetime() -> str:
    """Return the current date and time in ISO 8601 format.
    
    Returns:
        ISO 8601 formatted datetime string.
    """
    return datetime.now().isoformat()


##Function purpose: Execute a shell command and return the result safely
def execute_shell_command(command: str, description: str = "") -> ShellCommandResult:
    """Execute a shell command and return the result safely.
    
    Uses shlex.split to safely parse the command string and prevents
    shell injection attacks.
    
    Args:
        command: Shell command to execute.
        description: Optional description of the command for logging.
        
    Returns:
        ShellCommandResult with stdout, stderr, returncode, and error flag.
    """
    ##Error purpose: Handle command execution errors
    try:
        ##Step purpose: Use shlex.split to safely parse command
        command_args = shlex.split(command)
        
        ##Step purpose: Execute command
        result = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,  # 30 second timeout
        )
        
        ##Action purpose: Log command execution
        if description:
            logger.debug(
                "shell_command_executed",
                description=description,
                returncode=result.returncode,
            )
        
        return ShellCommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            error=result.returncode != 0,
        )
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"Command timed out: {command}"
        logger.error("shell_command_timeout", command=command[:50], error=str(e))
        return ShellCommandResult(
            stdout="",
            stderr=error_msg,
            returncode=-1,
            error=True,
        )
    except Exception as e:
        error_msg = f"Command execution failed: {e}"
        logger.error("shell_command_failed", command=command[:50], error=str(e))
        return ShellCommandResult(
            stdout="",
            stderr=error_msg,
            returncode=-1,
            error=True,
        )
