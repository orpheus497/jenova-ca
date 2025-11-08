# The JENOVA Cognitive Architecture - Plugin Schema
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 26: Plugin manifest schema and validation.

Defines the structure and validation for plugin.yaml manifests using
Pydantic for type-safe configuration.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class PluginPermission(str, Enum):
    """Available plugin permissions."""

    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    TOOLS_REGISTER = "tools:register"
    COMMANDS_REGISTER = "commands:register"
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    LLM_INFERENCE = "llm:inference"
    NETWORK_ACCESS = "network:access"


class ResourceLimits(BaseModel):
    """Resource limits for plugin execution."""

    max_cpu_seconds: int = Field(default=30, ge=1, le=300)
    max_memory_mb: int = Field(default=256, ge=64, le=2048)
    max_file_size_mb: int = Field(default=10, ge=1, le=100)

    class Config:
        """Pydantic config."""
        extra = "forbid"


class PluginDependency(BaseModel):
    """Plugin dependency specification."""

    plugin: str = Field(..., min_length=1)
    version: str = Field(default=">=1.0.0")

    class Config:
        """Pydantic config."""
        extra = "forbid"


class PluginManifest(BaseModel):
    """
    Plugin manifest (plugin.yaml) specification.

    Example:
        >>> manifest = PluginManifest(
        ...     id="example_plugin",
        ...     name="Example Plugin",
        ...     version="1.0.0",
        ...     author="developer",
        ...     description="Example plugin",
        ...     entry_point="example_plugin.ExamplePlugin",
        ...     jenova_min_version="5.3.0"
        ... )
    """

    # Required fields
    id: str = Field(..., min_length=1, max_length=64, regex=r"^[a-z0-9_]+$")
    name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., regex=r"^\d+\.\d+\.\d+$")
    author: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1, max_length=512)
    entry_point: str = Field(..., min_length=1)

    # Version compatibility
    jenova_min_version: str = Field(..., regex=r"^\d+\.\d+\.\d+$")
    jenova_max_version: Optional[str] = Field(None, regex=r"^\d+\.\d+\.\d+$")

    # Optional fields
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: str = Field(default="MIT")

    # Dependencies
    dependencies: List[PluginDependency] = Field(default_factory=list)
    python_dependencies: List[str] = Field(default_factory=list)

    # Permissions
    permissions: List[PluginPermission] = Field(default_factory=list)

    # Resource limits
    resources: ResourceLimits = Field(default_factory=ResourceLimits)

    # Additional metadata
    tags: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""
        extra = "forbid"
        use_enum_values = True

    @validator("id")
    def validate_id(cls, v):
        """Validate plugin ID format."""
        if v.startswith("_") or v.endswith("_"):
            raise ValueError("Plugin ID cannot start or end with underscore")
        return v

    @validator("permissions")
    def validate_permissions(cls, v):
        """Validate permissions are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Permissions must be unique")
        return v

    @validator("entry_point")
    def validate_entry_point(cls, v):
        """Validate entry point format."""
        if "." not in v:
            raise ValueError("Entry point must be in format 'module.ClassName'")
        return v


def validate_version_compatibility(
    plugin_version: str,
    jenova_version: str,
    min_version: str,
    max_version: Optional[str] = None
) -> bool:
    """
    Check if plugin is compatible with JENOVA version.

    Args:
        plugin_version: Plugin version string
        jenova_version: Current JENOVA version
        min_version: Minimum required JENOVA version
        max_version: Maximum supported JENOVA version (optional)

    Returns:
        True if compatible, False otherwise

    Example:
        >>> compatible = validate_version_compatibility(
        ...     "1.0.0",
        ...     "5.3.0",
        ...     "5.2.0",
        ...     "6.0.0"
        ... )
    """
    def parse_version(v: str) -> tuple:
        """Parse version string to tuple."""
        return tuple(int(x) for x in v.split("."))

    current = parse_version(jenova_version)
    min_ver = parse_version(min_version)

    # Check minimum version
    if current < min_ver:
        return False

    # Check maximum version if specified
    if max_version:
        max_ver = parse_version(max_version)
        if current >= max_ver:
            return False

    return True
