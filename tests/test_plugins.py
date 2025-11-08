# The JENOVA Cognitive Architecture - Plugin Tests
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 26: Tests for Plugin Architecture.

Tests plugin schema validation, API functionality, sandbox security,
and manager lifecycle with comprehensive coverage.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock

from jenova.plugins import (
    PluginPermission,
    ResourceLimits,
    PluginDependency,
    PluginManifest,
    validate_version_compatibility,
    PluginAPI,
    RateLimiter,
    PluginSandbox,
    ResourceMonitor,
    ImportWhitelist,
    SandboxedFileIO,
    PluginManager,
    PluginState,
)


class TestPluginSchema:
    """Test suite for plugin schema validation."""

    def test_plugin_permission_enum(self):
        """Test PluginPermission enum values."""
        assert PluginPermission.MEMORY_READ.value == "memory:read"
        assert PluginPermission.TOOLS_REGISTER.value == "tools:register"
        assert PluginPermission.LLM_INFERENCE.value == "llm:inference"

    def test_resource_limits_defaults(self):
        """Test ResourceLimits default values."""
        limits = ResourceLimits()
        assert limits.max_cpu_seconds == 30
        assert limits.max_memory_mb == 256
        assert limits.max_file_size_mb == 10

    def test_resource_limits_validation(self):
        """Test ResourceLimits range validation."""
        # Valid values
        limits = ResourceLimits(max_cpu_seconds=60, max_memory_mb=512)
        assert limits.max_cpu_seconds == 60

        # Invalid values should raise
        with pytest.raises(ValueError):
            ResourceLimits(max_cpu_seconds=0)  # Too low

        with pytest.raises(ValueError):
            ResourceLimits(max_memory_mb=3000)  # Too high

    def test_plugin_dependency_model(self):
        """Test PluginDependency model."""
        dep = PluginDependency(
            plugin_id="other_plugin", min_version="1.0.0", max_version="2.0.0"
        )
        assert dep.plugin_id == "other_plugin"
        assert dep.min_version == "1.0.0"

    def test_plugin_manifest_validation(self):
        """Test PluginManifest validation."""
        manifest = PluginManifest(
            id="test_plugin",
            name="Test Plugin",
            version="1.0.0",
            author="Test Author",
            description="Test description",
            entry_point="__init__.py",
            jenova_min_version="0.1.0",
            dependencies=[],
            permissions=[PluginPermission.MEMORY_READ],
            resources=ResourceLimits(),
        )

        assert manifest.id == "test_plugin"
        assert len(manifest.permissions) == 1

    def test_plugin_id_validation(self):
        """Test plugin ID regex validation."""
        # Valid IDs
        valid_ids = ["test", "test_plugin", "my_plugin_123"]
        for plugin_id in valid_ids:
            manifest = PluginManifest(
                id=plugin_id,
                name="Test",
                version="1.0.0",
                author="Test",
                description="Test",
                entry_point="__init__.py",
                jenova_min_version="0.1.0",
            )
            assert manifest.id == plugin_id

        # Invalid IDs
        with pytest.raises(ValueError):
            PluginManifest(
                id="Invalid-Name",  # Hyphens not allowed
                name="Test",
                version="1.0.0",
                author="Test",
                description="Test",
                entry_point="__init__.py",
                jenova_min_version="0.1.0",
            )

    def test_version_compatibility(self):
        """Test version compatibility checking."""
        # Compatible versions
        assert validate_version_compatibility("1.0.0", "1.0.0", "2.0.0")
        assert validate_version_compatibility("1.5.0", "1.0.0", "2.0.0")

        # Incompatible versions
        assert not validate_version_compatibility("0.9.0", "1.0.0", "2.0.0")
        assert not validate_version_compatibility("2.1.0", "1.0.0", "2.0.0")

        # No max version
        assert validate_version_compatibility("5.0.0", "1.0.0", None)


class TestRateLimiter:
    """Test suite for RateLimiter."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_calls=10, window_seconds=60)
        assert limiter.max_calls == 10
        assert limiter.window_seconds == 60

    def test_rate_limiting(self):
        """Test rate limit enforcement."""
        limiter = RateLimiter(max_calls=3, window_seconds=60)

        # First 3 calls should succeed
        assert limiter.check_rate_limit("test_key")
        assert limiter.check_rate_limit("test_key")
        assert limiter.check_rate_limit("test_key")

        # 4th call should fail
        assert not limiter.check_rate_limit("test_key")

    def test_separate_keys(self):
        """Test rate limiting with separate keys."""
        limiter = RateLimiter(max_calls=2, window_seconds=60)

        # Different keys have separate limits
        assert limiter.check_rate_limit("key1")
        assert limiter.check_rate_limit("key1")
        assert not limiter.check_rate_limit("key1")  # Exceeded

        assert limiter.check_rate_limit("key2")  # Still within limit
        assert limiter.check_rate_limit("key2")


class TestPluginAPI:
    """Test suite for PluginAPI."""

    @pytest.fixture
    def temp_sandbox(self):
        """Fixture providing temporary sandbox directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_engine(self):
        """Fixture providing mock cognitive engine."""
        engine = Mock()
        engine.memory_manager = Mock()
        engine.llm_interface = Mock()
        return engine

    @pytest.fixture
    def mock_logger(self):
        """Fixture providing mock logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        return logger

    @pytest.fixture
    def plugin_api(self, temp_sandbox, mock_engine, mock_logger):
        """Fixture providing PluginAPI instance."""
        return PluginAPI(
            plugin_id="test_plugin",
            permissions=["tools:register", "memory:read", "file:write"],
            sandbox_dir=temp_sandbox,
            cognitive_engine=mock_engine,
            file_logger=mock_logger,
        )

    def test_initialization(self, plugin_api):
        """Test API initialization."""
        assert plugin_api.plugin_id == "test_plugin"
        assert "tools:register" in plugin_api.permissions
        assert plugin_api.api_calls_count == 0

    def test_tool_registration(self, plugin_api):
        """Test tool registration."""

        def my_tool(args):
            return {"result": "success"}

        # Register tool
        assert plugin_api.register_tool("my_tool", my_tool)

        # Tool should be registered
        assert "my_tool" in plugin_api.registered_tools

        # Cannot register duplicate
        assert not plugin_api.register_tool("my_tool", my_tool)

    def test_tool_registration_permission(self, temp_sandbox, mock_engine, mock_logger):
        """Test tool registration permission check."""
        api_no_perm = PluginAPI(
            plugin_id="test",
            permissions=[],  # No permissions
            sandbox_dir=temp_sandbox,
            cognitive_engine=mock_engine,
            file_logger=mock_logger,
        )

        with pytest.raises(PermissionError):
            api_no_perm.register_tool("tool", lambda x: x)

    def test_file_operations(self, plugin_api):
        """Test sandboxed file operations."""
        # Write file
        assert plugin_api.write_file("test.txt", "Hello world")

        # Read file
        content = plugin_api.read_file("test.txt")
        assert content == "Hello world"

    def test_file_path_traversal_prevention(self, plugin_api):
        """Test path traversal prevention."""
        # Should raise PermissionError for path traversal
        with pytest.raises(PermissionError):
            plugin_api.read_file("../../etc/passwd")

        with pytest.raises(PermissionError):
            plugin_api.write_file("../../../tmp/evil.txt", "data")

    def test_get_stats(self, plugin_api):
        """Test statistics retrieval."""
        stats = plugin_api.get_stats()
        assert stats["plugin_id"] == "test_plugin"
        assert stats["tools_registered"] == 0
        assert "permissions" in stats


class TestResourceMonitor:
    """Test suite for ResourceMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ResourceMonitor(max_cpu_seconds=10, max_memory_mb=128)
        assert monitor.max_cpu_seconds == 10
        assert monitor.max_memory_mb == 128

    def test_monitoring_cycle(self):
        """Test start/stop monitoring cycle."""
        monitor = ResourceMonitor()
        monitor.start_monitoring()

        # Do some work
        sum([i**2 for i in range(1000)])

        resources = monitor.stop_monitoring()
        assert "cpu_time" in resources
        assert "wall_time" in resources
        assert resources["cpu_time"] >= 0


class TestImportWhitelist:
    """Test suite for ImportWhitelist."""

    def test_safe_modules_allowed(self):
        """Test safe modules are allowed."""
        safe_modules = ["json", "math", "datetime", "pathlib", "typing"]
        for module in safe_modules:
            assert ImportWhitelist.is_allowed(module)

    def test_forbidden_modules_blocked(self):
        """Test forbidden modules are blocked."""
        forbidden = ["os", "sys", "subprocess", "socket", "eval"]
        for module in forbidden:
            assert not ImportWhitelist.is_allowed(module)

    def test_numpy_scipy_allowed(self):
        """Test numpy/scipy allowed for optimization."""
        assert ImportWhitelist.is_allowed("numpy")
        assert ImportWhitelist.is_allowed("scipy")


class TestSandboxedFileIO:
    """Test suite for SandboxedFileIO."""

    @pytest.fixture
    def temp_sandbox(self):
        """Fixture providing temporary sandbox."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def file_io(self, temp_sandbox):
        """Fixture providing SandboxedFileIO instance."""
        return SandboxedFileIO(temp_sandbox)

    def test_write_and_read(self, file_io):
        """Test file write and read."""
        file_io.write_file("test.txt", "Hello world")
        content = file_io.read_file("test.txt")
        assert content == "Hello world"

    def test_path_traversal_prevention(self, file_io):
        """Test path traversal prevention."""
        with pytest.raises(PermissionError):
            file_io.read_file("../../etc/passwd")

        with pytest.raises(PermissionError):
            file_io.write_file("../../../tmp/evil.txt", "data")

    def test_list_files(self, file_io):
        """Test directory listing."""
        # Create some files
        file_io.write_file("file1.txt", "content1")
        file_io.write_file("file2.txt", "content2")

        files = file_io.list_files()
        assert len(files) == 2


class TestPluginSandbox:
    """Test suite for PluginSandbox."""

    @pytest.fixture
    def temp_sandbox(self):
        """Fixture providing temporary sandbox."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sandbox(self, temp_sandbox):
        """Fixture providing PluginSandbox instance."""
        return PluginSandbox(
            plugin_id="test_plugin",
            sandbox_dir=temp_sandbox,
            max_cpu_seconds=5,
            max_memory_mb=256,
        )

    def test_initialization(self, sandbox):
        """Test sandbox initialization."""
        assert sandbox.plugin_id == "test_plugin"
        assert sandbox.total_executions == 0

    def test_execute_success(self, sandbox):
        """Test successful function execution."""

        def test_func(x, y):
            return x + y

        result = sandbox.execute(test_func, 3, 5)

        assert result["success"]
        assert result["result"] == 8
        assert "resources" in result

    def test_execute_error(self, sandbox):
        """Test error handling in execution."""

        def failing_func():
            raise ValueError("Test error")

        result = sandbox.execute(failing_func)

        assert not result["success"]
        assert result["error"]["type"] == "ValueError"
        assert "Test error" in result["error"]["message"]

    def test_execution_stats(self, sandbox):
        """Test execution statistics tracking."""

        def simple_func():
            return 42

        # Execute multiple times
        for _ in range(3):
            sandbox.execute(simple_func)

        stats = sandbox.get_stats()
        assert stats["total_executions"] == 3
        assert stats["plugin_id"] == "test_plugin"

    def test_import_validation(self, sandbox):
        """Test import validation."""
        # Safe imports allowed
        assert sandbox.validate_import("json")
        assert sandbox.validate_import("math")

        # Forbidden imports blocked
        assert not sandbox.validate_import("os")
        assert not sandbox.validate_import("subprocess")


class TestPluginManager:
    """Test suite for PluginManager."""

    @pytest.fixture
    def temp_dirs(self):
        """Fixture providing temporary directories."""
        with tempfile.TemporaryDirectory() as plugins_dir:
            with tempfile.TemporaryDirectory() as sandbox_dir:
                yield Path(plugins_dir), Path(sandbox_dir)

    @pytest.fixture
    def mock_engine(self):
        """Fixture providing mock cognitive engine."""
        return Mock()

    @pytest.fixture
    def mock_logger(self):
        """Fixture providing mock logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        return logger

    @pytest.fixture
    def manager(self, temp_dirs, mock_engine, mock_logger):
        """Fixture providing PluginManager instance."""
        plugins_dir, sandbox_dir = temp_dirs
        return PluginManager(
            plugins_dir=plugins_dir,
            sandbox_dir=sandbox_dir,
            cognitive_engine=mock_engine,
            file_logger=mock_logger,
            jenova_version="0.1.0",
        )

    @pytest.fixture
    def test_plugin(self, temp_dirs):
        """Fixture creating a test plugin."""
        plugins_dir, _ = temp_dirs
        plugin_dir = plugins_dir / "test_plugin"
        plugin_dir.mkdir()

        # Create manifest
        manifest = {
            "id": "test_plugin",
            "name": "Test Plugin",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "Test plugin",
            "entry_point": "__init__.py",
            "jenova_min_version": "0.1.0",
            "dependencies": [],
            "permissions": ["memory:read"],
            "resources": {
                "max_cpu_seconds": 30,
                "max_memory_mb": 256,
                "max_file_size_mb": 10,
            },
        }

        with open(plugin_dir / "plugin.yaml", "w") as f:
            yaml.dump(manifest, f)

        # Create entry point
        with open(plugin_dir / "__init__.py", "w") as f:
            f.write(
                """
def initialize(api):
    api.log("info", "Test plugin initialized")
    return True

def activate(api):
    return True

def deactivate(api):
    return True

def cleanup(api):
    return True
"""
            )

        return "test_plugin"

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.jenova_version == "0.1.0"
        assert len(manager.plugins) == 0

    def test_discover_plugins(self, manager, test_plugin):
        """Test plugin discovery."""
        discovered = manager.discover_plugins()
        assert "test_plugin" in discovered
        assert "test_plugin" in manager.plugins

    def test_plugin_lifecycle(self, manager, test_plugin):
        """Test complete plugin lifecycle."""
        # Discover
        manager.discover_plugins()
        assert "test_plugin" in manager.plugins

        # Load
        assert manager.load_plugin("test_plugin")
        plugin_info = manager.plugins["test_plugin"]
        assert plugin_info.state == PluginState.LOADED

        # Initialize
        assert manager.initialize_plugin("test_plugin")
        assert plugin_info.state == PluginState.INITIALIZED
        assert plugin_info.api is not None
        assert plugin_info.sandbox is not None

        # Activate
        assert manager.activate_plugin("test_plugin")
        assert plugin_info.state == PluginState.ACTIVE

        # Deactivate
        assert manager.deactivate_plugin("test_plugin")
        assert plugin_info.state == PluginState.INITIALIZED

        # Unload
        assert manager.unload_plugin("test_plugin")
        assert plugin_info.state == PluginState.UNLOADED

    def test_list_plugins(self, manager, test_plugin):
        """Test listing plugins with state filter."""
        manager.discover_plugins()
        manager.load_plugin("test_plugin")

        # List all plugins
        all_plugins = manager.list_plugins()
        assert "test_plugin" in all_plugins

        # List loaded plugins
        loaded = manager.list_plugins(PluginState.LOADED)
        assert "test_plugin" in loaded

        # List active plugins (should be empty)
        active = manager.list_plugins(PluginState.ACTIVE)
        assert len(active) == 0

    def test_get_plugin_info(self, manager, test_plugin):
        """Test getting plugin information."""
        manager.discover_plugins()
        manager.load_plugin("test_plugin")

        info = manager.get_plugin_info("test_plugin")
        assert info is not None
        assert info["id"] == "test_plugin"
        assert info["name"] == "Test Plugin"
        assert info["version"] == "1.0.0"
        assert info["state"] == "loaded"

    def test_get_statistics(self, manager, test_plugin):
        """Test getting system statistics."""
        manager.discover_plugins()
        manager.load_plugin("test_plugin")

        stats = manager.get_statistics()
        assert stats["total_plugins"] == 1
        assert stats["by_state"]["loaded"] == 1
        assert stats["by_state"]["active"] == 0


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def full_setup(self):
        """Fixture providing complete plugin system setup."""
        with tempfile.TemporaryDirectory() as plugins_dir:
            with tempfile.TemporaryDirectory() as sandbox_dir:
                plugins_path = Path(plugins_dir)
                sandbox_path = Path(sandbox_dir)

                # Create test plugin
                plugin_dir = plugins_path / "integration_test"
                plugin_dir.mkdir()

                manifest = {
                    "id": "integration_test",
                    "name": "Integration Test Plugin",
                    "version": "1.0.0",
                    "author": "Test",
                    "description": "Integration test",
                    "entry_point": "__init__.py",
                    "jenova_min_version": "0.1.0",
                    "dependencies": [],
                    "permissions": ["tools:register", "file:write"],
                    "resources": {
                        "max_cpu_seconds": 30,
                        "max_memory_mb": 256,
                        "max_file_size_mb": 10,
                    },
                }

                with open(plugin_dir / "plugin.yaml", "w") as f:
                    yaml.dump(manifest, f)

                with open(plugin_dir / "__init__.py", "w") as f:
                    f.write(
                        """
def test_tool(args):
    return {"result": "success", "args": args}

def initialize(api):
    api.register_tool("test_tool", test_tool)
    api.write_file("init.txt", "initialized")

def activate(api):
    pass

def deactivate(api):
    pass

def cleanup(api):
    api.write_file("cleanup.txt", "cleaned up")
"""
                    )

                mock_engine = Mock()
                mock_logger = Mock()
                mock_logger.log_info = Mock()
                mock_logger.log_warning = Mock()
                mock_logger.log_error = Mock()

                manager = PluginManager(
                    plugins_dir=plugins_path,
                    sandbox_dir=sandbox_path,
                    cognitive_engine=mock_engine,
                    file_logger=mock_logger,
                    jenova_version="0.1.0",
                )

                yield manager, "integration_test"

    def test_complete_workflow(self, full_setup):
        """Test complete plugin workflow end-to-end."""
        manager, plugin_id = full_setup

        # Discover
        plugins = manager.discover_plugins()
        assert plugin_id in plugins

        # Load
        assert manager.load_plugin(plugin_id)

        # Initialize
        assert manager.initialize_plugin(plugin_id)
        plugin_info = manager.plugins[plugin_id]

        # Verify tool registered
        assert "test_tool" in plugin_info.api.registered_tools

        # Verify file created
        content = plugin_info.api.read_file("init.txt")
        assert content == "initialized"

        # Activate
        assert manager.activate_plugin(plugin_id)

        # Get registered tools
        tools = manager.get_all_registered_tools()
        assert f"{plugin_id}:test_tool" in tools

        # Unload (cleanup)
        assert manager.unload_plugin(plugin_id)
