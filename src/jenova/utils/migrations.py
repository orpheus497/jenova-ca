##Script function and purpose: Provides schema-versioned data loading and migration utilities
"""
Data Migration Utilities

Handles loading and saving JSON data with automatic schema versioning
and migration. Ensures data files are never corrupted during saves
(atomic write) and old data is automatically upgraded.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from jenova.exceptions import MigrationError, MigrationFailedError, SchemaVersionError
from jenova.utils.json_safe import safe_json_loads


##Step purpose: Define current schema version constant
SCHEMA_VERSION: int = 1

##Step purpose: Type variable for generic data loading
T = TypeVar("T", bound=dict)


##Function purpose: Load JSON with automatic schema migration
def load_json_with_migration(
    path: Path,
    default_factory: Callable[[], dict[str, object]],
    migrations: dict[int, Callable[[dict[str, object]], dict[str, object]]] | None = None,
) -> dict[str, object]:
    """
    Load JSON with automatic schema migration.
    
    Args:
        path: Path to JSON file
        default_factory: Factory function to create default data if file missing
        migrations: Dict mapping version -> migration function
        
    Returns:
        Loaded and migrated data dict
        
    Raises:
        SchemaVersionError: If data version is newer than supported
        MigrationFailedError: If migration fails
    """
    ##Condition purpose: Handle missing file by creating with defaults
    if not path.exists():
        ##Step purpose: Create default data with schema version
        data = default_factory()
        data["schema_version"] = SCHEMA_VERSION
        save_json_atomic(path, data)
        return data
    
    ##Step purpose: Load existing file
    with open(path, encoding="utf-8") as f:
        ##Sec: Use safe_json_loads for size/depth limits (P2-001)
        data = safe_json_loads(f.read())
    
    ##Step purpose: Get current version, defaulting to 0 for legacy data
    version = data.get("schema_version", 0)
    
    ##Condition purpose: Check if data is from future version
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(found=version, supported=SCHEMA_VERSION)
    
    ##Condition purpose: Migrate if version is old
    if version < SCHEMA_VERSION:
        data = migrate(data, from_version=version, migrations=migrations or {})
        save_json_atomic(path, data)
    
    return data


##Function purpose: Apply sequential migrations to upgrade data
def migrate(
    data: dict[str, object],
    from_version: int,
    migrations: dict[int, Callable[[dict[str, object]], dict[str, object]]],
) -> dict[str, object]:
    """
    Apply sequential migrations from from_version to SCHEMA_VERSION.
    
    Args:
        data: Data to migrate
        from_version: Starting version
        migrations: Dict mapping version -> migration function
        
    Returns:
        Migrated data with updated schema_version
        
    Raises:
        MigrationFailedError: If any migration fails
    """
    ##Loop purpose: Apply each migration in sequence
    for version in range(from_version, SCHEMA_VERSION):
        migration_fn = migrations.get(version)
        
        ##Condition purpose: Skip if no migration defined for this version
        if migration_fn is None:
            ##Step purpose: Just bump version if no explicit migration
            data["schema_version"] = version + 1
            continue
        
        ##Error purpose: Catch and wrap migration errors
        try:
            data = migration_fn(data)
            data["schema_version"] = version + 1
        except Exception as e:
            raise MigrationFailedError(
                from_version=version,
                to_version=version + 1,
                error=str(e),
            ) from e
    
    return data


##Function purpose: Save JSON atomically to prevent corruption
def save_json_atomic(path: Path, data: dict[str, object]) -> None:
    """
    Save JSON atomically to prevent corruption.
    
    Uses a temp file + rename strategy which is atomic on POSIX systems.
    The data is fully written to a temp file before being moved to the
    target path, ensuring the target is never left in a partial state.
    
    Args:
        path: Target path for JSON file
        data: Data to save
    """
    ##Step purpose: Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ##Step purpose: Write to temp file in same directory for atomic rename
    temp_fd, temp_path_str = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.stem,
        dir=path.parent,
    )
    temp_path = Path(temp_path_str)
    
    ##Error purpose: Clean up temp file on any error
    try:
        ##Action purpose: Write JSON to temp file
        with open(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        ##Action purpose: Set secure file permissions (0o600 = owner read/write only)
        os.chmod(temp_path, 0o600)
        
        ##Action purpose: Atomic rename to target path
        temp_path.replace(path)
        
        ##Action purpose: Ensure permissions are preserved after rename
        os.chmod(path, 0o600)
    except (OSError, IOError, PermissionError, json.JSONEncodeError) as e:
        ##Fix: Catch specific exceptions instead of bare Exception - preserves error context and security visibility
        ##Action purpose: Remove temp file if write failed
        temp_path.unlink(missing_ok=True)
        raise MigrationError(f"Failed to save migrated data to {path}: {e}") from e


##Function purpose: Load JSON without migration (for read-only access)
def load_json(path: Path) -> dict[str, object]:
    """
    Load JSON file without migration.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data dict
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(path, encoding="utf-8") as f:
        ##Sec: Use safe_json_loads for size/depth limits (P2-001)
        return safe_json_loads(f.read())


##Function purpose: Get schema version from file without loading full data
def get_schema_version(path: Path) -> int:
    """
    Get schema version from a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Schema version, or 0 if not present
    """
    ##Condition purpose: Return 0 if file doesn't exist
    if not path.exists():
        return 0
    
    ##Step purpose: Load and extract version
    data = load_json(path)
    return data.get("schema_version", 0)
