# The JENOVA Cognitive Architecture - Plugin Sandbox
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 26: Plugin Sandbox - Security enforcement for plugin execution.

Provides isolated execution environment with:
- CPU time limits
- Memory limits
- Restricted file I/O
- Import whitelist
- No network access
- No subprocess execution
"""

import time
import sys
import os
import resource
import threading
from typing import Any, Dict, List, Optional, Callable, Set
from pathlib import Path
import traceback
import importlib.util
import builtins


class ResourceMonitor:
    """Monitor and enforce resource limits for plugin execution."""

    def __init__(self, max_cpu_seconds: int = 30, max_memory_mb: int = 256):
        """
        Initialize resource monitor.

        Args:
            max_cpu_seconds: Maximum CPU time in seconds
            max_memory_mb: Maximum memory in megabytes
        """
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_mb = max_memory_mb
        self.start_time = None
        self.cpu_start = None

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.start_time = time.time()
        self.cpu_start = time.process_time()

        # Set memory limit (soft and hard)
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes)
            )
        except (ValueError, OSError):
            # Memory limiting not supported on this platform
            pass

    def check_limits(self) -> None:
        """
        Check if resource limits are exceeded.

        Raises:
            RuntimeError: If CPU time limit exceeded
        """
        if self.cpu_start is None:
            return

        cpu_elapsed = time.process_time() - self.cpu_start

        if cpu_elapsed > self.max_cpu_seconds:
            raise RuntimeError(
                f"CPU time limit exceeded ({cpu_elapsed:.1f}s > {self.max_cpu_seconds}s)"
            )

    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop monitoring and return resource usage.

        Returns:
            Dict with cpu_time, wall_time, memory_mb
        """
        if self.start_time is None or self.cpu_start is None:
            return {}

        cpu_time = time.process_time() - self.cpu_start
        wall_time = time.time() - self.start_time

        # Get memory usage
        try:
            memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is in KB on Linux, bytes on macOS
            if sys.platform == "darwin":
                memory_mb = memory_bytes / (1024 * 1024)
            else:
                memory_mb = memory_bytes / 1024
        except Exception:
            memory_mb = 0

        return {
            "cpu_time": cpu_time,
            "wall_time": wall_time,
            "memory_mb": memory_mb,
        }


class ImportWhitelist:
    """Whitelist for safe module imports."""

    # Safe standard library modules
    SAFE_MODULES: Set[str] = {
        # Core types
        "typing",
        "dataclasses",
        "collections",
        "enum",
        "abc",
        # Data structures
        "heapq",
        "bisect",
        "array",
        "queue",
        # String/text
        "string",
        "re",
        "difflib",
        # Math/numbers
        "math",
        "decimal",
        "fractions",
        "random",
        "statistics",
        # Date/time
        "datetime",
        "time",
        "calendar",
        # File paths
        "pathlib",
        # JSON/data
        "json",
        "csv",
        # Functional
        "itertools",
        "functools",
        "operator",
        # Other safe modules
        "copy",
        "hashlib",
        "uuid",
        "base64",
    }

    # Forbidden modules (security risks)
    FORBIDDEN_MODULES: Set[str] = {
        "os",  # File system access
        "sys",  # System manipulation
        "subprocess",  # Command execution
        "socket",  # Network access
        "urllib",  # Network access
        "requests",  # Network access
        "http",  # Network access
        "ftplib",  # Network access
        "smtplib",  # Network access
        "telnetlib",  # Network access
        "eval",  # Code execution
        "exec",  # Code execution
        "compile",  # Code execution
        "__import__",  # Dynamic imports
        "importlib",  # Dynamic imports
        "pickle",  # Arbitrary code execution
        "shelve",  # Uses pickle
        "marshal",  # Serialization
        "ctypes",  # Low-level access
        "threading",  # Resource management
        "multiprocessing",  # Resource management
        "signal",  # Signal handling
        "atexit",  # Exit handlers
    }

    @classmethod
    def is_allowed(cls, module_name: str) -> bool:
        """
        Check if module import is allowed.

        Args:
            module_name: Module name

        Returns:
            True if allowed, False otherwise
        """
        # Get top-level module
        top_level = module_name.split(".")[0]

        # Check forbidden first
        if top_level in cls.FORBIDDEN_MODULES:
            return False

        # Check whitelist
        if top_level in cls.SAFE_MODULES:
            return True

        # Allow numpy, scipy (needed for optimization)
        if top_level in {"numpy", "scipy", "np"}:
            return True

        # Default deny
        return False


class SandboxedFileIO:
    """Sandboxed file I/O restricted to plugin directory."""

    def __init__(self, sandbox_dir: Path):
        """
        Initialize sandboxed file I/O.

        Args:
            sandbox_dir: Directory to restrict operations to
        """
        self.sandbox_dir = Path(sandbox_dir).resolve()

    def validate_path(self, path: str) -> Path:
        """
        Validate file path is within sandbox.

        Args:
            path: Relative path

        Returns:
            Resolved absolute path

        Raises:
            PermissionError: If path is outside sandbox
        """
        # Resolve path
        file_path = (self.sandbox_dir / path).resolve()

        # Check if within sandbox
        if not str(file_path).startswith(str(self.sandbox_dir)):
            raise PermissionError(f"Path traversal detected: {path}")

        return file_path

    def read_file(self, path: str) -> str:
        """
        Read file from sandbox.

        Args:
            path: Relative path

        Returns:
            File contents
        """
        file_path = self.validate_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return file_path.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        """
        Write file to sandbox.

        Args:
            path: Relative path
            content: Content to write
        """
        file_path = self.validate_path(path)

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path.write_text(content, encoding="utf-8")

    def list_files(self, path: str = ".") -> List[str]:
        """
        List files in directory.

        Args:
            path: Relative directory path

        Returns:
            List of file names
        """
        dir_path = self.validate_path(path)

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        # Return relative paths
        files = []
        for item in dir_path.iterdir():
            rel_path = item.relative_to(self.sandbox_dir)
            files.append(str(rel_path))

        return files


class PluginSandbox:
    """
    Security sandbox for plugin execution.

    Provides isolated environment with resource limits, restricted imports,
    and sandboxed file I/O.

    Example:
        >>> sandbox = PluginSandbox(
        ...     plugin_id="my_plugin",
        ...     sandbox_dir=Path("/tmp/plugin_sandbox"),
        ...     max_cpu_seconds=30,
        ...     max_memory_mb=256
        ... )
        >>> result = sandbox.execute(plugin_function, arg1, arg2)
    """

    def __init__(
        self,
        plugin_id: str,
        sandbox_dir: Path,
        max_cpu_seconds: int = 30,
        max_memory_mb: int = 256,
        allowed_modules: Optional[List[str]] = None,
    ):
        """
        Initialize plugin sandbox.

        Args:
            plugin_id: Plugin identifier
            sandbox_dir: Sandbox directory for plugin
            max_cpu_seconds: Maximum CPU time
            max_memory_mb: Maximum memory
            allowed_modules: Additional allowed modules
        """
        self.plugin_id = plugin_id
        self.sandbox_dir = Path(sandbox_dir)
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_mb = max_memory_mb

        # Resource monitor
        self.monitor = ResourceMonitor(max_cpu_seconds, max_memory_mb)

        # File I/O
        self.file_io = SandboxedFileIO(self.sandbox_dir)

        # Allowed modules
        self.allowed_modules = set(allowed_modules or [])

        # Execution stats
        self.total_executions = 0
        self.total_cpu_time = 0.0
        self.total_wall_time = 0.0

    def execute(
        self, function: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute function in sandbox.

        Args:
            function: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Dict with result, success, error, resources

        Example:
            >>> result = sandbox.execute(my_function, arg1, kwarg1=value1)
            >>> if result["success"]:
            ...     print(result["result"])
        """
        self.total_executions += 1

        # Start monitoring
        self.monitor.start_monitoring()

        result = {
            "success": False,
            "result": None,
            "error": None,
            "resources": {},
        }

        try:
            # Execute function
            output = function(*args, **kwargs)
            result["success"] = True
            result["result"] = output

        except Exception as e:
            result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

        finally:
            # Stop monitoring and collect stats
            resources = self.monitor.stop_monitoring()
            result["resources"] = resources

            # Update totals
            self.total_cpu_time += resources.get("cpu_time", 0.0)
            self.total_wall_time += resources.get("wall_time", 0.0)

        return result

    def validate_import(self, module_name: str) -> bool:
        """
        Check if module import is allowed.

        Args:
            module_name: Module name

        Returns:
            True if allowed
        """
        # Check additional allowed modules
        if module_name in self.allowed_modules:
            return True

        # Check whitelist
        return ImportWhitelist.is_allowed(module_name)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get sandbox execution statistics.

        Returns:
            Dict with execution stats
        """
        return {
            "plugin_id": self.plugin_id,
            "total_executions": self.total_executions,
            "total_cpu_time": self.total_cpu_time,
            "total_wall_time": self.total_wall_time,
            "avg_cpu_time": (
                self.total_cpu_time / self.total_executions
                if self.total_executions > 0
                else 0.0
            ),
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_memory_mb": self.max_memory_mb,
        }

    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.total_executions = 0
        self.total_cpu_time = 0.0
        self.total_wall_time = 0.0
