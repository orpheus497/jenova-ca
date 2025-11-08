# The JENOVA Cognitive Architecture - Plugins Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 26: Plugin Architecture - Extensible plugin system.

Provides safe, sandboxed plugin execution with permission-based
security, resource limits, and lifecycle management.

Components:
    - PluginSchema: Pydantic models for manifest validation
    - PluginAPI: Safe API interface for plugins
    - PluginSandbox: Security sandbox for execution
    - PluginManager: Plugin lifecycle orchestration

Example:
    >>> from jenova.plugins import PluginManager
    >>> from pathlib import Path
    >>>
    >>> manager = PluginManager(
    ...     plugins_dir=Path("~/.jenova-ai/plugins"),
    ...     sandbox_dir=Path("~/.jenova-ai/plugin_sandboxes"),
    ...     cognitive_engine=engine,
    ...     file_logger=logger
    ... )
    >>>
    >>> # Discover and load plugins
    >>> plugins = manager.discover_plugins()
    >>> for plugin_id in plugins:
    ...     manager.load_plugin(plugin_id)
    ...     manager.initialize_plugin(plugin_id)
    ...     manager.activate_plugin(plugin_id)
    >>>
    >>> # Get registered tools
    >>> tools = manager.get_all_registered_tools()
    >>> commands = manager.get_all_registered_commands()
"""

from jenova.plugins.plugin_schema import (
    PluginPermission,
    ResourceLimits,
    PluginDependency,
    PluginManifest,
    validate_version_compatibility,
)
from jenova.plugins.plugin_api import PluginAPI, RateLimiter
from jenova.plugins.plugin_sandbox import (
    PluginSandbox,
    ResourceMonitor,
    ImportWhitelist,
    SandboxedFileIO,
)
from jenova.plugins.plugin_manager import PluginManager, PluginState, PluginInfo

__all__ = [
    # Schema
    "PluginPermission",
    "ResourceLimits",
    "PluginDependency",
    "PluginManifest",
    "validate_version_compatibility",
    # API
    "PluginAPI",
    "RateLimiter",
    # Sandbox
    "PluginSandbox",
    "ResourceMonitor",
    "ImportWhitelist",
    "SandboxedFileIO",
    # Manager
    "PluginManager",
    "PluginState",
    "PluginInfo",
]
