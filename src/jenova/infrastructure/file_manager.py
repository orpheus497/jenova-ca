# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Atomic file operations manager.

Provides safe, atomic file operations to prevent data corruption.
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import filelock


class FileManager:
    """Manager for atomic file operations."""

    def __init__(self, ui_logger=None, file_logger=None):
        """
        Initialize file manager.

        Args:
            ui_logger: Optional UI logger
            file_logger: Optional file logger
        """
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.lock_timeout = 30.0  # seconds

    @contextmanager
    def atomic_write(self, filepath: str, mode: str = "w"):
        """
        Context manager for atomic file writes.

        Writes to a temporary file first, then atomically replaces the target.
        This prevents corruption if write is interrupted.

        Args:
            filepath: Target file path
            mode: File open mode ('w' for text, 'wb' for binary)

        Yields:
            File object to write to

        Example:
            with file_manager.atomic_write('data.json') as f:
                json.dump(data, f)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file in same directory as target
        # This ensures atomic rename works (same filesystem)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent, prefix=f".{filepath.name}.", suffix=".tmp"
        )

        try:
            # Close the file descriptor, we'll use the path
            os.close(temp_fd)

            # Open temp file with requested mode
            with open(temp_path, mode) as f:
                yield f

            # Atomically replace target file
            # os.replace is atomic on POSIX systems
            os.replace(temp_path, filepath)

            if self.file_logger:
                self.file_logger.log_info(f"Atomically wrote: {filepath}")

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise e

    @contextmanager
    def file_lock(self, filepath: str, timeout: Optional[float] = None):
        """
        Context manager for file locking.

        Prevents concurrent access to the same file.

        Args:
            filepath: File to lock
            timeout: Lock timeout in seconds (None = use default)

        Yields:
            None

        Raises:
            filelock.Timeout: If lock cannot be acquired within timeout

        Example:
            with file_manager.file_lock('data.json'):
                # Safe to read/write file here
                pass
        """
        lock_path = f"{filepath}.lock"
        lock = filelock.FileLock(lock_path, timeout=timeout or self.lock_timeout)

        try:
            with lock:
                yield
        finally:
            # Clean up lock file if it exists
            try:
                lock_file = Path(lock_path)
                if lock_file.exists():
                    lock_file.unlink()
            except Exception:
                pass

    def write_json_atomic(self, filepath: str, data: Dict[str, Any], indent: int = 2):
        """
        Atomically write JSON data to file.

        Args:
            filepath: Target file path
            data: Dictionary to write
            indent: JSON indentation (None for compact)

        Raises:
            ValueError: If data is not serializable
            IOError: If write fails
        """
        with self.atomic_write(filepath, "w") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    def read_json_safe(
        self, filepath: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Safely read JSON file with fallback.

        Args:
            filepath: File to read
            default: Default value if file doesn't exist or is invalid

        Returns:
            Parsed JSON data or default value
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return default if default is not None else {}

        try:
            with self.file_lock(str(filepath)):
                with open(filepath, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            if self.file_logger:
                self.file_logger.log_warning(f"Failed to read {filepath}: {e}")
            return default if default is not None else {}

    def backup_file(
        self, filepath: str, backup_suffix: str = ".backup"
    ) -> Optional[str]:
        """
        Create a backup of a file.

        Args:
            filepath: File to backup
            backup_suffix: Suffix for backup file

        Returns:
            Path to backup file or None if source doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return None

        backup_path = Path(str(filepath) + backup_suffix)

        try:
            shutil.copy2(filepath, backup_path)
            if self.file_logger:
                self.file_logger.log_info(f"Backed up {filepath} to {backup_path}")
            return str(backup_path)
        except IOError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to backup {filepath}: {e}")
            return None

    def safe_delete(self, filepath: str, backup: bool = True) -> bool:
        """
        Safely delete a file, optionally creating a backup first.

        Args:
            filepath: File to delete
            backup: Whether to create backup before deletion

        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return True

        try:
            # Create backup if requested
            if backup:
                self.backup_file(str(filepath))

            # Delete file
            filepath.unlink()

            if self.file_logger:
                self.file_logger.log_info(f"Deleted: {filepath}")

            return True

        except IOError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to delete {filepath}: {e}")
            return False

    def ensure_directory(self, dirpath: str) -> bool:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            dirpath: Directory path

        Returns:
            True if directory exists or was created successfully
        """
        dirpath = Path(dirpath)

        try:
            dirpath.mkdir(parents=True, exist_ok=True)
            return True
        except IOError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to create directory {dirpath}: {e}")
            return False

    def move_atomic(self, src: str, dst: str, overwrite: bool = False) -> bool:
        """
        Atomically move a file.

        Args:
            src: Source file path
            dst: Destination file path
            overwrite: Whether to overwrite existing destination

        Returns:
            True if successful, False otherwise
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            if self.file_logger:
                self.file_logger.log_error(f"Source file does not exist: {src}")
            return False

        if dst_path.exists() and not overwrite:
            if self.file_logger:
                self.file_logger.log_error(f"Destination already exists: {dst}")
            return False

        try:
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic move/rename
            os.replace(src_path, dst_path)

            if self.file_logger:
                self.file_logger.log_info(f"Moved {src} to {dst}")

            return True

        except IOError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to move {src} to {dst}: {e}")
            return False

    def append_to_file(self, filepath: str, content: str, create: bool = True) -> bool:
        """
        Safely append content to a file.

        Args:
            filepath: File to append to
            content: Content to append
            create: Whether to create file if it doesn't exist

        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)

        if not filepath.exists() and not create:
            if self.file_logger:
                self.file_logger.log_error(f"File does not exist: {filepath}")
            return False

        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Use file lock to prevent concurrent appends
            with self.file_lock(str(filepath)):
                with open(filepath, "a") as f:
                    f.write(content)

            return True

        except IOError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to append to {filepath}: {e}")
            return False

    def read_lines(self, filepath: str, encoding: str = "utf-8") -> Optional[list[str]]:
        """
        Safely read all lines from a file.

        Args:
            filepath: File to read
            encoding: File encoding

        Returns:
            List of lines or None if file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return None

        try:
            with self.file_lock(str(filepath)):
                with open(filepath, "r", encoding=encoding) as f:
                    return f.readlines()
        except IOError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to read {filepath}: {e}")
            return None

    def get_file_size(self, filepath: str) -> Optional[int]:
        """
        Get file size in bytes.

        Args:
            filepath: File path

        Returns:
            File size in bytes or None if file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return None

        try:
            return filepath.stat().st_size
        except IOError:
            return None

    def list_files(
        self, directory: str, pattern: str = "*", recursive: bool = False
    ) -> list[Path]:
        """
        List files in a directory.

        Args:
            directory: Directory to search
            pattern: Glob pattern for matching files
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        directory = Path(directory)

        if not directory.exists():
            return []

        try:
            if recursive:
                return list(directory.rglob(pattern))
            else:
                return list(directory.glob(pattern))
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to list files in {directory}: {e}")
            return []
