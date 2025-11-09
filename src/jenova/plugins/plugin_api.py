# The JENOVA Cognitive Architecture - Plugin API
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 26: Plugin API - Safe interface for plugin interactions.

Provides controlled access to JENOVA functionality with:
- Tool registration
- Command registration
- Memory access (read-only by default)
- LLM inference (rate-limited)
- File I/O (sandboxed)
"""

import time
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from collections import defaultdict
import threading


class RateLimiter:
    """Simple rate limiter for plugin API calls."""

    def __init__(self, max_calls: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def check_rate_limit(self, key: str) -> bool:
        """
        Check if key is within rate limit.

        Args:
            key: Rate limit key (e.g., plugin_id)

        Returns:
            True if within limit, False if exceeded
        """
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds

            # Remove old calls outside window
            self.calls[key] = [
                t for t in self.calls[key]
                if t > window_start
            ]

            # Check if limit exceeded
            if len(self.calls[key]) >= self.max_calls:
                return False

            # Record this call
            self.calls[key].append(now)
            return True


class PluginAPI:
    """
    Safe API surface for plugins.

    Provides controlled access to system functionality with:
    - Permission checking
    - Rate limiting
    - Resource tracking
    - Sandboxed operations
    """

    def __init__(
        self,
        plugin_id: str,
        permissions: List[str],
        sandbox_dir: Path,
        cognitive_engine: Any,
        file_logger: Any,
    ):
        """
        Initialize plugin API.

        Args:
            plugin_id: Plugin identifier
            permissions: List of granted permissions
            sandbox_dir: Sandboxed directory for file operations
            cognitive_engine: Reference to cognitive engine
            file_logger: Logger instance
        """
        self.plugin_id = plugin_id
        self.permissions = set(permissions)
        self.sandbox_dir = Path(sandbox_dir)
        self.cognitive_engine = cognitive_engine
        self.file_logger = file_logger

        # Rate limiters
        self.llm_rate_limiter = RateLimiter(max_calls=10, window_seconds=60)
        self.memory_rate_limiter = RateLimiter(max_calls=30, window_seconds=60)

        # Registered tools and commands
        self.registered_tools: Dict[str, Callable] = {}
        self.registered_commands: Dict[str, Callable] = {}

        # Resource tracking
        self.api_calls_count = 0

    def _check_permission(self, permission: str) -> bool:
        """Check if plugin has permission."""
        return permission in self.permissions

    def _log_api_call(self, method: str, **kwargs):
        """Log API call for auditing."""
        self.api_calls_count += 1
        self.file_logger.log_info(
            f"Plugin {self.plugin_id} called {method} (call #{self.api_calls_count})"
        )

    # Tool Management

    def register_tool(self, tool_name: str, tool_function: Callable) -> bool:
        """
        Register custom tool.

        Args:
            tool_name: Tool name (must be unique)
            tool_function: Tool implementation

        Returns:
            True if registered successfully

        Example:
            >>> def my_tool(args):
            ...     return {"result": "success"}
            >>> api.register_tool("my_tool", my_tool)
        """
        if not self._check_permission("tools:register"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: tools:register"
            )

        self._log_api_call("register_tool", tool_name=tool_name)

        if tool_name in self.registered_tools:
            return False  # Already registered

        self.registered_tools[tool_name] = tool_function
        return True

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister tool.

        Args:
            tool_name: Tool to unregister

        Returns:
            True if unregistered successfully
        """
        if not self._check_permission("tools:register"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: tools:register"
            )

        self._log_api_call("unregister_tool", tool_name=tool_name)

        if tool_name in self.registered_tools:
            del self.registered_tools[tool_name]
            return True
        return False

    # Command Management

    def register_command(self, command_name: str, handler: Callable) -> bool:
        """
        Register custom command.

        Args:
            command_name: Command name (without leading /)
            handler: Command handler function

        Returns:
            True if registered successfully

        Example:
            >>> def my_command(args):
            ...     return "Command executed"
            >>> api.register_command("mycommand", my_command)
        """
        if not self._check_permission("commands:register"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: commands:register"
            )

        self._log_api_call("register_command", command_name=command_name)

        if command_name in self.registered_commands:
            return False

        self.registered_commands[command_name] = handler
        return True

    def unregister_command(self, command_name: str) -> bool:
        """
        Unregister command.

        Args:
            command_name: Command to unregister

        Returns:
            True if unregistered successfully
        """
        if not self._check_permission("commands:register"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: commands:register"
            )

        self._log_api_call("unregister_command", command_name=command_name)

        if command_name in self.registered_commands:
            del self.registered_commands[command_name]
            return True
        return False

    # Memory Access

    def query_memory(
        self,
        query: str,
        memory_type: str = "semantic",
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query memory system (read-only).

        Args:
            query: Search query
            memory_type: Type (semantic, episodic, procedural)
            n_results: Number of results

        Returns:
            List of memory results

        Example:
            >>> results = api.query_memory("Python programming", "semantic")
        """
        if not self._check_permission("memory:read"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: memory:read"
            )

        # Rate limiting
        if not self.memory_rate_limiter.check_rate_limit(self.plugin_id):
            raise RuntimeError(f"Plugin {self.plugin_id} exceeded memory query rate limit")

        self._log_api_call("query_memory", query=query, memory_type=memory_type)

        # Access memory system (simplified)
        if hasattr(self.cognitive_engine, "memory_manager"):
            memory = getattr(self.cognitive_engine.memory_manager, memory_type, None)
            if memory:
                results = memory.search(query, n_results=n_results)
                return results

        return []

    def add_insight(self, content: str, concern: Optional[str] = None) -> bool:
        """
        Add insight to memory (controlled write).

        Args:
            content: Insight content
            concern: Optional concern/topic

        Returns:
            True if added successfully
        """
        if not self._check_permission("memory:write"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: memory:write"
            )

        self._log_api_call("add_insight", content=content[:50])

        # Add to insight manager
        if hasattr(self.cognitive_engine, "insight_manager"):
            self.cognitive_engine.insight_manager.save_insight(content, concern)
            return True

        return False

    # LLM Access

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using LLM (rate-limited).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text

        Example:
            >>> text = api.generate_text("Explain quantum physics", max_tokens=256)
        """
        if not self._check_permission("llm:inference"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: llm:inference"
            )

        # Rate limiting
        if not self.llm_rate_limiter.check_rate_limit(self.plugin_id):
            raise RuntimeError(f"Plugin {self.plugin_id} exceeded LLM rate limit")

        # Token limit enforcement
        max_tokens = min(max_tokens, 512)  # Cap at 512

        self._log_api_call("generate_text", prompt_length=len(prompt))

        # Generate using LLM
        if hasattr(self.cognitive_engine, "llm_interface"):
            result = self.cognitive_engine.llm_interface.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return result

        return ""

    # File I/O (Sandboxed)

    def read_file(self, path: str) -> str:
        """
        Read file from sandbox (sandboxed).

        Args:
            path: Relative path within sandbox

        Returns:
            File contents

        Example:
            >>> content = api.read_file("data/input.txt")
        """
        if not self._check_permission("file:read"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: file:read"
            )

        self._log_api_call("read_file", path=path)

        # Resolve path within sandbox
        file_path = (self.sandbox_dir / path).resolve()

        # Security: ensure path is within sandbox
        if not str(file_path).startswith(str(self.sandbox_dir.resolve())):
            raise PermissionError("Path traversal detected")

        # Read file
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"File not found: {path}")

    def write_file(self, path: str, content: str) -> bool:
        """
        Write file to sandbox (sandboxed).

        Args:
            path: Relative path within sandbox
            content: Content to write

        Returns:
            True if written successfully

        Example:
            >>> api.write_file("output/results.txt", "Results: 42")
        """
        if not self._check_permission("file:write"):
            raise PermissionError(
                f"Plugin {self.plugin_id} lacks permission: file:write"
            )

        self._log_api_call("write_file", path=path, size=len(content))

        # Resolve path within sandbox
        file_path = (self.sandbox_dir / path).resolve()

        # Security: ensure path is within sandbox
        if not str(file_path).startswith(str(self.sandbox_dir.resolve())):
            raise PermissionError("Path traversal detected")

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path.write_text(content, encoding="utf-8")
        return True

    # Configuration

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get plugin-specific configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        self._log_api_call("get_config", key=key)

        # Access plugin config (simplified)
        plugin_config = {}  # Would load from config
        return plugin_config.get(key, default)

    # Logging

    def log(self, level: str, message: str) -> None:
        """
        Log message (integrated with system logger).

        Args:
            level: Log level (info, warning, error)
            message: Log message

        Example:
            >>> api.log("info", "Plugin initialized successfully")
        """
        self._log_api_call("log", level=level)

        log_message = f"[Plugin {self.plugin_id}] {message}"

        if level == "info":
            self.file_logger.log_info(log_message)
        elif level == "warning":
            self.file_logger.log_warning(log_message)
        elif level == "error":
            self.file_logger.log_error(log_message)
        else:
            self.file_logger.log_info(log_message)

    # Stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Get plugin API usage statistics.

        Returns:
            Dict with usage stats
        """
        return {
            "plugin_id": self.plugin_id,
            "api_calls_total": self.api_calls_count,
            "tools_registered": len(self.registered_tools),
            "commands_registered": len(self.registered_commands),
            "permissions": list(self.permissions),
        }
