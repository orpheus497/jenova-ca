##Script function and purpose: Centralized file I/O utilities for The JENOVA Cognitive Architecture
##This module provides JSON file load/save operations to avoid code duplication across modules

import json
import os
from typing import Any, Optional

##Function purpose: Load JSON data from file with error handling
def load_json_file(
    filepath: str, 
    default: Any = None, 
    file_logger: Optional[Any] = None
) -> Any:
    """
    Load JSON data from a file with error handling.
    
    Args:
        filepath: Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid
        file_logger: Optional FileLogger instance for error logging
        
    Returns:
        Parsed JSON data, or default value on failure
    """
    ##Block purpose: Return default if file doesn't exist
    if not os.path.exists(filepath):
        return default if default is not None else {}
    
    ##Block purpose: Attempt to load and parse JSON file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        if file_logger:
            file_logger.log_error(f"JSON decode error in {filepath}: {e}")
        return default if default is not None else {}
    except OSError as e:
        if file_logger:
            file_logger.log_error(f"Error reading file {filepath}: {e}")
        return default if default is not None else {}


##Function purpose: Save data to JSON file with error handling
def save_json_file(
    filepath: str, 
    data: Any, 
    indent: int = 4,
    file_logger: Optional[Any] = None
) -> bool:
    """
    Save data to a JSON file with error handling.
    
    Args:
        filepath: Path to the JSON file
        data: Data to serialize and save
        indent: JSON indentation level (default 4)
        file_logger: Optional FileLogger instance for error logging
        
    Returns:
        True if save was successful, False otherwise
    """
    ##Block purpose: Ensure parent directory exists
    parent_dir = os.path.dirname(filepath)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except OSError as e:
            if file_logger:
                file_logger.log_error(f"Error creating directory {parent_dir}: {e}")
            return False
    
    ##Block purpose: Attempt to write JSON file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except (OSError, TypeError) as e:
        if file_logger:
            file_logger.log_error(f"Error saving file {filepath}: {e}")
        return False


##Function purpose: Ensure a directory exists, creating it if necessary
def ensure_directory(path: str, file_logger: Optional[Any] = None) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        file_logger: Optional FileLogger instance for error logging
        
    Returns:
        True if directory exists or was created, False on failure
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        if file_logger:
            file_logger.log_error(f"Error creating directory {path}: {e}")
        return False
