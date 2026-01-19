##Script function and purpose: Error message sanitization utilities
"""
Error Message Sanitization Utilities

Provides functions to sanitize error messages and file paths
to prevent information disclosure.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


##Function purpose: Sanitize file path for error messages
def sanitize_path_for_error(path: Path | str) -> str:
    """Sanitize file path for error messages.
    
    Removes sensitive path information, showing only the last
    few path components to prevent information disclosure.
    
    Args:
        path: Path to sanitize
        
    Returns:
        Sanitized path string safe for error messages
    """
    path_str = str(path)
    
    ##Step purpose: Convert to Path object
    path_obj = Path(path_str)
    
    ##Step purpose: Get path parts
    parts = path_obj.parts
    
    ##Condition purpose: Show only last 2 components if path is long
    if len(parts) > 2:
        return str(Path(*parts[-2:]))
    
    return path_str


##Function purpose: Remove sensitive information from error messages
def sanitize_error_message(msg: str) -> str:
    """Remove sensitive information from error messages.
    
    Removes absolute paths and home directory references
    from error messages.
    
    Args:
        msg: Error message to sanitize
        
    Returns:
        Sanitized error message
    """
    ##Step purpose: Get home directory
    home = os.path.expanduser('~')
    
    ##Step purpose: Replace home directory with ~
    sanitized = msg.replace(home, '~')
    
    ##Step purpose: Remove absolute paths (keep relative) - POSIX only
    ##Condition purpose: Only process if message contains absolute paths
    if '/' in sanitized:
        ##Step purpose: Replace absolute paths with filename only
        def replace_path(match: re.Match[str]) -> str:
            path_str = match.group(0)
            ##Condition purpose: Check if absolute path (POSIX: starts with /)
            if path_str.startswith('/'):
                return Path(path_str).name
            return path_str
        
        ##Action purpose: Replace absolute paths (POSIX forward slash only)
        sanitized = re.sub(r'/[^\s]+', replace_path, sanitized)
    
    return sanitized


##Function purpose: Create safe error message with path
def safe_error_with_path(message: str, path: Path | str) -> str:
    """Create safe error message including path.
    
    Combines error message with sanitized path.
    
    Args:
        message: Base error message
        path: Path to include (will be sanitized)
        
    Returns:
        Safe error message with sanitized path
    """
    safe_path = sanitize_path_for_error(path)
    return f"{message}: {safe_path}"
