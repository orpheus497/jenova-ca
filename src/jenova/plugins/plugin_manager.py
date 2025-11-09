# The JENOVA Cognitive Architecture - Plugin Manager
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 26: Plugin Manager - Plugin lifecycle orchestration.

Manages plugin discovery, loading, initialization, activation,
deactivation, and cleanup with dependency resolution.
"""

import importlib.util
import yaml
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from enum import Enum
import threading

from jenova.plugins.plugin_schema import PluginManifest, PluginPermission, validate_version_compatibility
from jenova.plugins.plugin_api import PluginAPI
from jenova.plugins.plugin_sandbox import PluginSandbox


class PluginState(str, Enum):
    """Plugin lifecycle states."""

    UNLOADED = "unloaded"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    FAILED = "failed"


class PluginInfo:
    """Information about loaded plugin."""

    def __init__(
        self,
        manifest: PluginManifest,
        plugin_dir: Path,
        module: Any = None,
        state: PluginState = PluginState.UNLOADED,
    ):
        """
        Initialize plugin info.

        Args:
            manifest: Plugin manifest
            plugin_dir: Plugin directory
            module: Loaded module
            state: Current state
        """
        self.manifest = manifest
        self.plugin_dir = plugin_dir
        self.module = module
        self.state = state
        self.api: Optional[PluginAPI] = None
        self.sandbox: Optional[PluginSandbox] = None
        self.error: Optional[str] = None


class PluginManager:
    """
    Plugin lifecycle manager.

    Handles plugin discovery, loading, dependency resolution,
    and lifecycle management (init, activate, deactivate, cleanup).

    Example:
        >>> manager = PluginManager(
        ...     plugins_dir=Path("~/.jenova-ai/plugins"),
        ...     sandbox_dir=Path("~/.jenova-ai/plugin_sandboxes"),
        ...     cognitive_engine=engine,
        ...     file_logger=logger
        ... )
        >>> manager.discover_plugins()
        >>> manager.load_plugin("example_plugin")
        >>> manager.activate_plugin("example_plugin")
    """

    def __init__(
        self,
        plugins_dir: Path,
        sandbox_dir: Path,
        cognitive_engine: Any,
        file_logger: Any,
        jenova_version: str = "0.1.0",
    ):
        """
        Initialize plugin manager.

        Args:
            plugins_dir: Directory containing plugins
            sandbox_dir: Directory for plugin sandboxes
            cognitive_engine: Reference to cognitive engine
            file_logger: Logger instance
            jenova_version: Current JENOVA version
        """
        self.plugins_dir = Path(plugins_dir)
        self.sandbox_dir = Path(sandbox_dir)
        self.cognitive_engine = cognitive_engine
        self.file_logger = file_logger
        self.jenova_version = jenova_version

        # Loaded plugins
        self.plugins: Dict[str, PluginInfo] = {}

        # Thread safety
        self.lock = threading.RLock()

        # Create directories
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins.

        Returns:
            List of plugin IDs

        Example:
            >>> plugin_ids = manager.discover_plugins()
            >>> print(f"Found {len(plugin_ids)} plugins")
        """
        discovered = []

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            manifest_path = plugin_dir / "plugin.yaml"
            if not manifest_path.exists():
                continue

            try:
                # Load manifest
                with open(manifest_path) as f:
                    manifest_data = yaml.safe_load(f)

                manifest = PluginManifest(**manifest_data)

                # Check version compatibility
                if not validate_version_compatibility(
                    self.jenova_version,
                    manifest.jenova_min_version,
                    manifest.jenova_max_version,
                ):
                    self.file_logger.log_warning(
                        f"Plugin {manifest.id} incompatible with JENOVA {self.jenova_version}"
                    )
                    continue

                discovered.append(manifest.id)

                # Store plugin info if not already loaded
                with self.lock:
                    if manifest.id not in self.plugins:
                        self.plugins[manifest.id] = PluginInfo(
                            manifest=manifest,
                            plugin_dir=plugin_dir,
                            state=PluginState.UNLOADED,
                        )

            except Exception as e:
                self.file_logger.log_error(
                    f"Error discovering plugin in {plugin_dir}: {e}"
                )

        return discovered

    def load_plugin(self, plugin_id: str) -> bool:
        """
        Load plugin module.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if loaded successfully

        Example:
            >>> if manager.load_plugin("my_plugin"):
            ...     print("Plugin loaded")
        """
        with self.lock:
            if plugin_id not in self.plugins:
                self.file_logger.log_error(f"Plugin {plugin_id} not found")
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.state != PluginState.UNLOADED:
                self.file_logger.log_warning(
                    f"Plugin {plugin_id} already loaded (state: {plugin_info.state})"
                )
                return False

            try:
                # Resolve dependencies
                if not self._resolve_dependencies(plugin_id):
                    plugin_info.error = "Dependency resolution failed"
                    plugin_info.state = PluginState.FAILED
                    return False

                # Load module
                entry_point = plugin_info.plugin_dir / plugin_info.manifest.entry_point
                spec = importlib.util.spec_from_file_location(
                    plugin_info.manifest.id, entry_point
                )

                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load module from {entry_point}")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                plugin_info.module = module
                plugin_info.state = PluginState.LOADED

                self.file_logger.log_info(f"Plugin {plugin_id} loaded successfully")
                return True

            except Exception as e:
                plugin_info.error = str(e)
                plugin_info.state = PluginState.FAILED
                self.file_logger.log_error(f"Error loading plugin {plugin_id}: {e}")
                return False

    def _resolve_dependencies(self, plugin_id: str) -> bool:
        """
        Resolve and load plugin dependencies.

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if all dependencies resolved
        """
        plugin_info = self.plugins[plugin_id]
        dependencies = plugin_info.manifest.dependencies

        for dep in dependencies:
            # Check if dependency plugin exists
            if dep.plugin_id not in self.plugins:
                self.file_logger.log_error(
                    f"Dependency {dep.plugin_id} not found for {plugin_id}"
                )
                return False

            dep_info = self.plugins[dep.plugin_id]

            # Check version compatibility
            if not validate_version_compatibility(
                dep_info.manifest.version, dep.min_version, dep.max_version
            ):
                self.file_logger.log_error(
                    f"Dependency {dep.plugin_id} version incompatible for {plugin_id}"
                )
                return False

            # Load dependency if not loaded
            if dep_info.state == PluginState.UNLOADED:
                if not self.load_plugin(dep.plugin_id):
                    return False

        return True

    def initialize_plugin(self, plugin_id: str) -> bool:
        """
        Initialize plugin (create API and sandbox).

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if initialized successfully

        Example:
            >>> manager.initialize_plugin("my_plugin")
        """
        with self.lock:
            if plugin_id not in self.plugins:
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.state != PluginState.LOADED:
                self.file_logger.log_error(
                    f"Plugin {plugin_id} must be loaded first (state: {plugin_info.state})"
                )
                return False

            try:
                # Create sandbox directory
                plugin_sandbox_dir = self.sandbox_dir / plugin_id
                plugin_sandbox_dir.mkdir(parents=True, exist_ok=True)

                # Create sandbox
                plugin_info.sandbox = PluginSandbox(
                    plugin_id=plugin_id,
                    sandbox_dir=plugin_sandbox_dir,
                    max_cpu_seconds=plugin_info.manifest.resources.max_cpu_seconds,
                    max_memory_mb=plugin_info.manifest.resources.max_memory_mb,
                )

                # Create API
                plugin_info.api = PluginAPI(
                    plugin_id=plugin_id,
                    permissions=[p.value for p in plugin_info.manifest.permissions],
                    sandbox_dir=plugin_sandbox_dir,
                    cognitive_engine=self.cognitive_engine,
                    file_logger=self.file_logger,
                )

                # Call plugin initialize() if exists
                if hasattr(plugin_info.module, "initialize"):
                    result = plugin_info.sandbox.execute(
                        plugin_info.module.initialize, plugin_info.api
                    )

                    if not result["success"]:
                        error = result["error"]
                        raise RuntimeError(
                            f"Initialize failed: {error['type']}: {error['message']}"
                        )

                plugin_info.state = PluginState.INITIALIZED

                self.file_logger.log_info(
                    f"Plugin {plugin_id} initialized successfully"
                )
                return True

            except Exception as e:
                plugin_info.error = str(e)
                plugin_info.state = PluginState.FAILED
                self.file_logger.log_error(
                    f"Error initializing plugin {plugin_id}: {e}"
                )
                return False

    def activate_plugin(self, plugin_id: str) -> bool:
        """
        Activate plugin (start operation).

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if activated successfully

        Example:
            >>> manager.activate_plugin("my_plugin")
        """
        with self.lock:
            if plugin_id not in self.plugins:
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.state != PluginState.INITIALIZED:
                self.file_logger.log_error(
                    f"Plugin {plugin_id} must be initialized first"
                )
                return False

            try:
                # Call plugin activate() if exists
                if hasattr(plugin_info.module, "activate"):
                    result = plugin_info.sandbox.execute(
                        plugin_info.module.activate, plugin_info.api
                    )

                    if not result["success"]:
                        error = result["error"]
                        raise RuntimeError(
                            f"Activate failed: {error['type']}: {error['message']}"
                        )

                plugin_info.state = PluginState.ACTIVE

                self.file_logger.log_info(f"Plugin {plugin_id} activated")
                return True

            except Exception as e:
                plugin_info.error = str(e)
                plugin_info.state = PluginState.FAILED
                self.file_logger.log_error(
                    f"Error activating plugin {plugin_id}: {e}"
                )
                return False

    def deactivate_plugin(self, plugin_id: str) -> bool:
        """
        Deactivate plugin (stop operation).

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if deactivated successfully

        Example:
            >>> manager.deactivate_plugin("my_plugin")
        """
        with self.lock:
            if plugin_id not in self.plugins:
                return False

            plugin_info = self.plugins[plugin_id]

            if plugin_info.state != PluginState.ACTIVE:
                return True  # Already deactivated

            try:
                # Call plugin deactivate() if exists
                if hasattr(plugin_info.module, "deactivate"):
                    result = plugin_info.sandbox.execute(
                        plugin_info.module.deactivate, plugin_info.api
                    )

                    if not result["success"]:
                        self.file_logger.log_warning(
                            f"Plugin {plugin_id} deactivate() returned error"
                        )

                plugin_info.state = PluginState.INITIALIZED

                self.file_logger.log_info(f"Plugin {plugin_id} deactivated")
                return True

            except Exception as e:
                self.file_logger.log_error(
                    f"Error deactivating plugin {plugin_id}: {e}"
                )
                return False

    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload plugin (cleanup and remove).

        Args:
            plugin_id: Plugin identifier

        Returns:
            True if unloaded successfully

        Example:
            >>> manager.unload_plugin("my_plugin")
        """
        with self.lock:
            if plugin_id not in self.plugins:
                return False

            plugin_info = self.plugins[plugin_id]

            # Deactivate if active
            if plugin_info.state == PluginState.ACTIVE:
                self.deactivate_plugin(plugin_id)

            try:
                # Call plugin cleanup() if exists
                if hasattr(plugin_info.module, "cleanup"):
                    result = plugin_info.sandbox.execute(
                        plugin_info.module.cleanup, plugin_info.api
                    )

                    if not result["success"]:
                        self.file_logger.log_warning(
                            f"Plugin {plugin_id} cleanup() returned error"
                        )

                # Clear references
                plugin_info.module = None
                plugin_info.api = None
                plugin_info.sandbox = None
                plugin_info.state = PluginState.UNLOADED

                self.file_logger.log_info(f"Plugin {plugin_id} unloaded")
                return True

            except Exception as e:
                self.file_logger.log_error(f"Error unloading plugin {plugin_id}: {e}")
                return False

    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """
        Get plugin information.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Dict with plugin info or None
        """
        with self.lock:
            if plugin_id not in self.plugins:
                return None

            plugin_info = self.plugins[plugin_id]

            return {
                "id": plugin_info.manifest.id,
                "name": plugin_info.manifest.name,
                "version": plugin_info.manifest.version,
                "author": plugin_info.manifest.author,
                "description": plugin_info.manifest.description,
                "state": plugin_info.state.value,
                "permissions": [p.value for p in plugin_info.manifest.permissions],
                "dependencies": [
                    {
                        "plugin_id": dep.plugin_id,
                        "min_version": dep.min_version,
                        "max_version": dep.max_version,
                    }
                    for dep in plugin_info.manifest.dependencies
                ],
                "resources": {
                    "max_cpu_seconds": plugin_info.manifest.resources.max_cpu_seconds,
                    "max_memory_mb": plugin_info.manifest.resources.max_memory_mb,
                    "max_file_size_mb": plugin_info.manifest.resources.max_file_size_mb,
                },
                "error": plugin_info.error,
            }

    def list_plugins(self, state_filter: Optional[PluginState] = None) -> List[str]:
        """
        List all plugins.

        Args:
            state_filter: Filter by state (optional)

        Returns:
            List of plugin IDs

        Example:
            >>> active_plugins = manager.list_plugins(PluginState.ACTIVE)
        """
        with self.lock:
            if state_filter is None:
                return list(self.plugins.keys())

            return [
                plugin_id
                for plugin_id, info in self.plugins.items()
                if info.state == state_filter
            ]

    def get_all_registered_tools(self) -> Dict[str, Callable]:
        """
        Get all registered tools from active plugins.

        Returns:
            Dict mapping tool names to functions
        """
        tools = {}

        with self.lock:
            for plugin_id, plugin_info in self.plugins.items():
                if plugin_info.state == PluginState.ACTIVE and plugin_info.api:
                    # Add plugin's registered tools
                    for tool_name, tool_func in plugin_info.api.registered_tools.items():
                        # Prefix with plugin ID to avoid conflicts
                        full_name = f"{plugin_id}:{tool_name}"
                        tools[full_name] = tool_func

        return tools

    def get_all_registered_commands(self) -> Dict[str, Callable]:
        """
        Get all registered commands from active plugins.

        Returns:
            Dict mapping command names to handlers
        """
        commands = {}

        with self.lock:
            for plugin_id, plugin_info in self.plugins.items():
                if plugin_info.state == PluginState.ACTIVE and plugin_info.api:
                    # Add plugin's registered commands
                    for cmd_name, cmd_handler in plugin_info.api.registered_commands.items():
                        # Prefix with plugin ID
                        full_name = f"{plugin_id}:{cmd_name}"
                        commands[full_name] = cmd_handler

        return commands

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get plugin system statistics.

        Returns:
            Dict with statistics
        """
        with self.lock:
            stats = {
                "total_plugins": len(self.plugins),
                "by_state": {},
                "active_plugins": [],
                "failed_plugins": [],
            }

            # Count by state
            for state in PluginState:
                count = sum(
                    1 for p in self.plugins.values() if p.state == state
                )
                stats["by_state"][state.value] = count

            # Active plugins
            stats["active_plugins"] = [
                {
                    "id": p.manifest.id,
                    "name": p.manifest.name,
                    "version": p.manifest.version,
                }
                for p in self.plugins.values()
                if p.state == PluginState.ACTIVE
            ]

            # Failed plugins
            stats["failed_plugins"] = [
                {
                    "id": p.manifest.id,
                    "name": p.manifest.name,
                    "error": p.error,
                }
                for p in self.plugins.values()
                if p.state == PluginState.FAILED
            ]

            return stats
