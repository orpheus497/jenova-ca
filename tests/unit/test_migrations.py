##Script function and purpose: Unit tests for migrations module
"""
Migrations Unit Tests

Tests for schema-versioned data loading and migration utilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jenova.utils.migrations import (
    SCHEMA_VERSION,
    load_json_with_migration,
    save_json_atomic,
    migrate,
    get_schema_version,
)
from jenova.exceptions import SchemaVersionError, MigrationFailedError


##Class purpose: Test atomic save operations
class TestSaveJsonAtomic:
    """Tests for save_json_atomic."""
    
    ##Method purpose: Test that save creates file
    def test_saves_json_file(self, tmp_path: Path) -> None:
        """save_json_atomic creates a valid JSON file."""
        path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        
        ##Action purpose: Save and verify
        save_json_atomic(path, data)
        
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data
    
    ##Method purpose: Test that save creates parent directories
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """save_json_atomic creates parent directories if missing."""
        path = tmp_path / "nested" / "deep" / "test.json"
        data = {"key": "value"}
        
        ##Action purpose: Save to nested path
        save_json_atomic(path, data)
        
        assert path.exists()
    
    ##Method purpose: Test that save overwrites existing file
    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """save_json_atomic overwrites existing content."""
        path = tmp_path / "test.json"
        
        ##Step purpose: Write initial content
        save_json_atomic(path, {"old": "data"})
        
        ##Step purpose: Overwrite
        save_json_atomic(path, {"new": "data"})
        
        loaded = json.loads(path.read_text())
        assert loaded == {"new": "data"}


##Class purpose: Test load with migration
class TestLoadJsonWithMigration:
    """Tests for load_json_with_migration."""
    
    ##Method purpose: Test that load creates file if missing
    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        """Creates file with defaults if missing."""
        path = tmp_path / "test.json"
        
        ##Action purpose: Load non-existent file
        data = load_json_with_migration(
            path,
            default_factory=lambda: {"nodes": {}},
        )
        
        assert path.exists()
        assert data["nodes"] == {}
        assert data["schema_version"] == SCHEMA_VERSION
    
    ##Method purpose: Test that load preserves existing data
    def test_loads_existing_data(self, tmp_path: Path) -> None:
        """Loads existing data without modification if current version."""
        path = tmp_path / "test.json"
        existing = {"key": "value", "schema_version": SCHEMA_VERSION}
        path.write_text(json.dumps(existing))
        
        ##Action purpose: Load existing file
        data = load_json_with_migration(
            path,
            default_factory=lambda: {},
        )
        
        assert data["key"] == "value"
    
    ##Method purpose: Test that load raises error for future version
    def test_raises_error_for_future_version(self, tmp_path: Path) -> None:
        """Raises SchemaVersionError for unsupported future version."""
        path = tmp_path / "test.json"
        future = {"key": "value", "schema_version": SCHEMA_VERSION + 10}
        path.write_text(json.dumps(future))
        
        ##Action purpose: Attempt to load future version
        with pytest.raises(SchemaVersionError) as exc_info:
            load_json_with_migration(path, default_factory=lambda: {})
        
        assert exc_info.value.found == SCHEMA_VERSION + 10
        assert exc_info.value.supported == SCHEMA_VERSION


##Class purpose: Test migration function
class TestMigrate:
    """Tests for migrate function."""
    
    ##Method purpose: Test migration applies functions in order
    def test_applies_migrations_in_order(self) -> None:
        """Migrations are applied sequentially."""
        ##Step purpose: Define test migrations
        migrations = {
            0: lambda d: {**d, "v1_field": True, "schema_version": 1},
        }
        
        ##Action purpose: Migrate from v0
        data = {"original": "data", "schema_version": 0}
        result = migrate(data, from_version=0, migrations=migrations)
        
        assert result["original"] == "data"
        assert result["v1_field"] is True
    
    ##Method purpose: Test migration failure is wrapped
    def test_migration_failure_raises_migration_failed_error(self) -> None:
        """Migration errors are wrapped in MigrationFailedError."""
        ##Step purpose: Define failing migration
        def failing_migration(data: dict) -> dict:
            raise ValueError("Migration failed!")
        
        migrations = {0: failing_migration}
        
        ##Action purpose: Attempt migration
        with pytest.raises(MigrationFailedError) as exc_info:
            migrate({"schema_version": 0}, from_version=0, migrations=migrations)
        
        assert exc_info.value.from_version == 0
        assert exc_info.value.to_version == 1
