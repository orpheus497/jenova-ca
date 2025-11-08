# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - File Tools

"""
File system operations tool module.

This module provides secure file system operations for the JENOVA cognitive
architecture with sandboxing, path traversal protection, and size limits.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from jenova.tools.base import BaseTool, ToolResult


class FileTools(BaseTool):
    """
    File system operations tool with security controls.

    Provides safe file reading, writing, and directory operations with
    sandboxing, path traversal protection, and configurable size limits.

    Security Features:
        - Path traversal protection via resolve() and validation
        - Configurable file size limits (default: 100 MB)
        - Sandbox directory enforcement (optional)
        - Safe path handling using pathlib.Path

    Methods:
        execute: Main entry point for file operations
        read_file: Read file contents with size validation
        write_file: Write file contents safely
        list_directory: List directory contents
        file_exists: Check file existence
        get_file_info: Get file metadata
        create_directory: Create directory safely
        delete_file: Delete file with confirmation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ui_logger: Any,
        file_logger: Any,
        sandbox_root: Optional[Union[str, Path]] = None,
        max_file_size_mb: Optional[int] = None
    ):
        """
        Initialize file tools.

        Args:
            config: Configuration dictionary
            ui_logger: UI logger instance
            file_logger: File logger instance
            sandbox_root: Optional root directory for sandboxing (all operations
                         must be within this directory)
            max_file_size_mb: Maximum file size in MB (default: 100 from config)
        """
        super().__init__(
            name="file_tools",
            description="Provides secure file system operations",
            config=config,
            ui_logger=ui_logger,
            file_logger=file_logger
        )

        # Configure sandbox root
        if sandbox_root:
            self.sandbox_root = Path(sandbox_root).resolve()
        else:
            self.sandbox_root = None

        # Configure file size limit
        self.max_file_size_mb = max_file_size_mb or config.get(
            'file_tools', {}
        ).get('max_file_size_mb', 100)

    def execute(
        self,
        operation: str,
        path: Union[str, Path],
        **kwargs
    ) -> ToolResult:
        """
        Execute file operation.

        Args:
            operation: Operation type ('read', 'write', 'list', 'exists',
                      'info', 'mkdir', 'delete')
            path: File or directory path
            **kwargs: Operation-specific arguments

        Returns:
            ToolResult with operation results

        Example:
            >>> tools = FileTools(config, ui_logger, file_logger)
            >>> result = tools.execute('read', '/tmp/test.txt')
            >>> print(result.data)  # File contents
        """
        try:
            # Validate and resolve path
            resolved_path = self._validate_path(path)
            if not resolved_path:
                return self._create_error_result(
                    f"Path validation failed: {path}"
                )

            # Route to appropriate operation
            if operation == 'read':
                return self.read_file(resolved_path)
            elif operation == 'write':
                content = kwargs.get('content', '')
                return self.write_file(resolved_path, content)
            elif operation == 'list':
                return self.list_directory(resolved_path)
            elif operation == 'exists':
                return self.file_exists(resolved_path)
            elif operation == 'info':
                return self.get_file_info(resolved_path)
            elif operation == 'mkdir':
                return self.create_directory(resolved_path)
            elif operation == 'delete':
                confirm = kwargs.get('confirm', False)
                return self.delete_file(resolved_path, confirm)
            else:
                return self._create_error_result(
                    f"Unknown operation: {operation}"
                )

        except Exception as e:
            error_msg = f"File operation '{operation}' failed: {str(e)}"
            result = self._create_error_result(error_msg)
            self._log_execution({'operation': operation, 'path': str(path)}, result)
            return result

    def read_file(self, path: Path) -> ToolResult:
        """
        Read file contents with size validation.

        Args:
            path: Resolved path to file

        Returns:
            ToolResult with file contents as string

        Example:
            >>> result = tools.read_file(Path('/tmp/test.txt'))
            >>> print(result.data)  # "Hello, world!"
        """
        try:
            if not path.exists():
                return self._create_error_result(f"File not found: {path}")

            if not path.is_file():
                return self._create_error_result(f"Not a file: {path}")

            # Check file size
            file_size = path.stat().st_size
            max_bytes = self.max_file_size_mb * 1024 * 1024

            if file_size > max_bytes:
                return self._create_error_result(
                    f"File too large: {file_size / 1024 / 1024:.2f} MB "
                    f"(max: {self.max_file_size_mb} MB)"
                )

            # Read file contents
            content = path.read_text(encoding='utf-8')

            result = self._create_success_result(
                data=content,
                metadata={
                    'path': str(path),
                    'size_bytes': file_size,
                    'encoding': 'utf-8'
                }
            )

            self._log_execution({'operation': 'read', 'path': str(path)}, result)
            return result

        except UnicodeDecodeError:
            return self._create_error_result(
                f"File is not valid UTF-8 text: {path}"
            )
        except Exception as e:
            return self._create_error_result(f"Failed to read file: {str(e)}")

    def write_file(self, path: Path, content: str) -> ToolResult:
        """
        Write file contents safely.

        Args:
            path: Resolved path to file
            content: Content to write

        Returns:
            ToolResult indicating success

        Example:
            >>> result = tools.write_file(Path('/tmp/test.txt'), "Hello!")
            >>> print(result.success)  # True
        """
        try:
            # Validate content size
            content_bytes = len(content.encode('utf-8'))
            max_bytes = self.max_file_size_mb * 1024 * 1024

            if content_bytes > max_bytes:
                return self._create_error_result(
                    f"Content too large: {content_bytes / 1024 / 1024:.2f} MB "
                    f"(max: {self.max_file_size_mb} MB)"
                )

            # Create parent directory if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            path.write_text(content, encoding='utf-8')

            result = self._create_success_result(
                data=f"File written: {path}",
                metadata={
                    'path': str(path),
                    'size_bytes': content_bytes
                }
            )

            self._log_execution(
                {'operation': 'write', 'path': str(path)},
                result
            )
            return result

        except Exception as e:
            return self._create_error_result(f"Failed to write file: {str(e)}")

    def list_directory(self, path: Path) -> ToolResult:
        """
        List directory contents.

        Args:
            path: Resolved path to directory

        Returns:
            ToolResult with list of file/directory names

        Example:
            >>> result = tools.list_directory(Path('/tmp'))
            >>> print(result.data)  # ['file1.txt', 'dir1', 'file2.py']
        """
        try:
            if not path.exists():
                return self._create_error_result(f"Directory not found: {path}")

            if not path.is_dir():
                return self._create_error_result(f"Not a directory: {path}")

            # List directory contents
            entries = sorted([entry.name for entry in path.iterdir()])

            result = self._create_success_result(
                data=entries,
                metadata={
                    'path': str(path),
                    'count': len(entries)
                }
            )

            self._log_execution({'operation': 'list', 'path': str(path)}, result)
            return result

        except Exception as e:
            return self._create_error_result(f"Failed to list directory: {str(e)}")

    def file_exists(self, path: Path) -> ToolResult:
        """
        Check if file or directory exists.

        Args:
            path: Resolved path to check

        Returns:
            ToolResult with boolean indicating existence

        Example:
            >>> result = tools.file_exists(Path('/tmp/test.txt'))
            >>> print(result.data)  # True or False
        """
        try:
            exists = path.exists()

            result = self._create_success_result(
                data=exists,
                metadata={
                    'path': str(path),
                    'is_file': path.is_file() if exists else None,
                    'is_dir': path.is_dir() if exists else None
                }
            )

            self._log_execution({'operation': 'exists', 'path': str(path)}, result)
            return result

        except Exception as e:
            return self._create_error_result(f"Failed to check existence: {str(e)}")

    def get_file_info(self, path: Path) -> ToolResult:
        """
        Get file metadata.

        Args:
            path: Resolved path to file

        Returns:
            ToolResult with file metadata dictionary

        Example:
            >>> result = tools.get_file_info(Path('/tmp/test.txt'))
            >>> print(result.data['size'])  # 1024
        """
        try:
            if not path.exists():
                return self._create_error_result(f"Path not found: {path}")

            stat = path.stat()

            info = {
                'path': str(path),
                'name': path.name,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / 1024 / 1024,
                'is_file': path.is_file(),
                'is_dir': path.is_dir(),
                'modified_time': stat.st_mtime,
                'created_time': stat.st_ctime,
                'permissions': oct(stat.st_mode)[-3:]
            }

            result = self._create_success_result(data=info)
            self._log_execution({'operation': 'info', 'path': str(path)}, result)
            return result

        except Exception as e:
            return self._create_error_result(f"Failed to get file info: {str(e)}")

    def create_directory(self, path: Path) -> ToolResult:
        """
        Create directory safely.

        Args:
            path: Resolved path to directory

        Returns:
            ToolResult indicating success

        Example:
            >>> result = tools.create_directory(Path('/tmp/newdir'))
            >>> print(result.success)  # True
        """
        try:
            # Create directory (with parents if needed)
            path.mkdir(parents=True, exist_ok=True)

            result = self._create_success_result(
                data=f"Directory created: {path}",
                metadata={'path': str(path)}
            )

            self._log_execution({'operation': 'mkdir', 'path': str(path)}, result)
            return result

        except Exception as e:
            return self._create_error_result(f"Failed to create directory: {str(e)}")

    def delete_file(self, path: Path, confirm: bool = False) -> ToolResult:
        """
        Delete file with confirmation.

        Args:
            path: Resolved path to file
            confirm: Confirmation flag (must be True to delete)

        Returns:
            ToolResult indicating success

        Example:
            >>> result = tools.delete_file(Path('/tmp/test.txt'), confirm=True)
            >>> print(result.success)  # True
        """
        try:
            if not confirm:
                return self._create_error_result(
                    "Delete operation requires confirm=True"
                )

            if not path.exists():
                return self._create_error_result(f"Path not found: {path}")

            # Delete file or directory
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                # Only delete empty directories for safety
                path.rmdir()
            else:
                return self._create_error_result(f"Cannot delete: {path}")

            result = self._create_success_result(
                data=f"Deleted: {path}",
                metadata={'path': str(path)}
            )

            self._log_execution({'operation': 'delete', 'path': str(path)}, result)
            return result

        except OSError as e:
            if "Directory not empty" in str(e):
                return self._create_error_result(
                    f"Cannot delete non-empty directory: {path}"
                )
            return self._create_error_result(f"Failed to delete: {str(e)}")
        except Exception as e:
            return self._create_error_result(f"Failed to delete: {str(e)}")

    def _validate_path(self, path: Union[str, Path]) -> Optional[Path]:
        """
        Validate path for security.

        Performs:
        - Path resolution to absolute path
        - Sandbox validation if sandbox_root is configured
        - Path traversal attack prevention

        Args:
            path: Path to validate

        Returns:
            Resolved Path object if valid, None otherwise
        """
        try:
            # Convert to Path and resolve to absolute
            resolved = Path(path).resolve()

            # Check sandbox if configured
            if self.sandbox_root:
                # Ensure path is within sandbox
                try:
                    resolved.relative_to(self.sandbox_root)
                except ValueError:
                    if self.file_logger:
                        self.file_logger.log_warning(
                            f"Path outside sandbox: {path} "
                            f"(sandbox: {self.sandbox_root})"
                        )
                    return None

            return resolved

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Path validation failed: {e}")
            return None
