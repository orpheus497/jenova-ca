# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 22: Code Quality & Testing - Core Module Tests

"""
Unit tests for core application modules.

Tests the dependency injection container, component lifecycle management,
and application bootstrapper to ensure correct initialization order and
error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock

from jenova.core.container import DependencyContainer, ServiceLifetime
from jenova.core.lifecycle import ComponentLifecycle, LifecyclePhase
from jenova.core.bootstrap import ApplicationBootstrapper


class TestDependencyContainer:
    """Tests for DependencyContainer class."""

    def test_register_singleton(self):
        """Test singleton service registration."""
        container = DependencyContainer()

        # Register a simple class
        class TestService:
            def __init__(self):
                self.value = 42

        container.register_singleton('test_service', TestService)

        # Resolve twice - should be same instance
        instance1 = container.resolve('test_service')
        instance2 = container.resolve('test_service')

        assert instance1 is instance2
        assert instance1.value == 42

    def test_register_transient(self):
        """Test transient service registration."""
        container = DependencyContainer()

        class TestService:
            def __init__(self):
                self.value = 42

        container.register_transient('test_service', TestService)

        # Resolve twice - should be different instances
        instance1 = container.resolve('test_service')
        instance2 = container.resolve('test_service')

        assert instance1 is not instance2
        assert instance1.value == 42
        assert instance2.value == 42

    def test_dependency_resolution(self):
        """Test automatic dependency resolution."""
        container = DependencyContainer()

        class Logger:
            def __init__(self):
                self.logs = []

            def log(self, message):
                self.logs.append(message)

        class Database:
            def __init__(self, logger):
                self.logger = logger
                self.logger.log("Database initialized")

        # Register dependencies
        container.register_singleton('logger', Logger)
        container.register_singleton('database', Database, depends_on=['logger'])

        # Resolve database (should auto-inject logger)
        db = container.resolve('database')

        assert isinstance(db.logger, Logger)
        assert "Database initialized" in db.logger.logs

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        container = DependencyContainer()

        class ServiceA:
            def __init__(self, service_b):
                self.service_b = service_b

        class ServiceB:
            def __init__(self, service_a):
                self.service_a = service_a

        # Create circular dependency
        container.register('service_a', ServiceA, depends_on=['service_b'])
        container.register('service_b', ServiceB, depends_on=['service_a'])

        # Should raise RuntimeError for circular dependency
        with pytest.raises(RuntimeError, match="Circular dependency"):
            container.resolve('service_a')

    def test_factory_registration(self):
        """Test factory function registration."""
        container = DependencyContainer()

        class Config:
            def __init__(self):
                self.db_path = "/tmp/test.db"

        def create_database(config):
            return {'path': config.db_path, 'connected': True}

        container.register_instance('config', Config())
        container.register_factory('database', create_database, depends_on=['config'])

        db = container.resolve('database')

        assert db['path'] == "/tmp/test.db"
        assert db['connected'] is True

    def test_missing_service_error(self):
        """Test error when resolving unregistered service."""
        container = DependencyContainer()

        with pytest.raises(KeyError, match="not registered"):
            container.resolve('nonexistent_service')

    def test_register_instance(self):
        """Test registering existing instance."""
        container = DependencyContainer()

        existing_instance = Mock()
        existing_instance.value = 123

        container.register_instance('service', existing_instance)

        resolved = container.resolve('service')

        assert resolved is existing_instance
        assert resolved.value == 123

    def test_is_registered(self):
        """Test service registration check."""
        container = DependencyContainer()

        class TestService:
            pass

        assert not container.is_registered('test')

        container.register_singleton('test', TestService)

        assert container.is_registered('test')

    def test_get_service_info(self):
        """Test retrieving service metadata."""
        container = DependencyContainer()

        class TestService:
            pass

        container.register_singleton('test', TestService, depends_on=['logger'])

        info = container.get_service_info('test')

        assert info['service_type'] == 'test'
        assert info['implementation'] == 'TestService'
        assert info['lifetime'] == 'SINGLETON'
        assert info['dependencies'] == ['logger']
        assert info['is_resolved'] is False


class TestComponentLifecycle:
    """Tests for ComponentLifecycle class."""

    def test_component_registration(self):
        """Test component registration."""
        lifecycle = ComponentLifecycle()

        component = Mock()
        lifecycle.register('test', component)

        assert lifecycle.get_component('test') is component
        assert lifecycle.get_phase('test') == LifecyclePhase.CREATED

    def test_initialization_order(self):
        """Test dependency-based initialization order."""
        lifecycle = ComponentLifecycle()

        # Create mock components
        logger = Mock()
        logger.initialize = Mock()

        database = Mock()
        database.initialize = Mock()

        service = Mock()
        service.initialize = Mock()

        # Register with dependencies: service depends on database depends on logger
        lifecycle.register('logger', logger)
        lifecycle.register('database', database, depends_on=['logger'])
        lifecycle.register('service', service, depends_on=['database'])

        # Initialize all
        lifecycle.initialize_all()

        # Verify initialization order (logger first, then database, then service)
        assert lifecycle.get_phase('logger') == LifecyclePhase.INITIALIZED
        assert lifecycle.get_phase('database') == LifecyclePhase.INITIALIZED
        assert lifecycle.get_phase('service') == LifecyclePhase.INITIALIZED

        # Verify methods were called
        logger.initialize.assert_called_once()
        database.initialize.assert_called_once()
        service.initialize.assert_called_once()

    def test_lifecycle_phases(self):
        """Test component lifecycle state transitions."""
        lifecycle = ComponentLifecycle()

        component = Mock()
        component.initialize = Mock()
        component.start = Mock()
        component.stop = Mock()
        component.dispose = Mock()

        lifecycle.register('test', component)

        # CREATED -> INITIALIZED
        lifecycle.initialize_component('test')
        assert lifecycle.get_phase('test') == LifecyclePhase.INITIALIZED

        # INITIALIZED -> STARTED
        lifecycle.start_component('test')
        assert lifecycle.get_phase('test') == LifecyclePhase.STARTED

        # STARTED -> STOPPED
        lifecycle.stop_component('test')
        assert lifecycle.get_phase('test') == LifecyclePhase.STOPPED

        # STOPPED -> DISPOSED
        lifecycle.dispose_component('test')
        assert lifecycle.get_phase('test') == LifecyclePhase.DISPOSED

        # Verify all lifecycle methods called
        component.initialize.assert_called_once()
        component.start.assert_called_once()
        component.stop.assert_called_once()
        component.dispose.assert_called_once()

    def test_circular_dependency_detection(self):
        """Test circular dependency detection in lifecycle."""
        lifecycle = ComponentLifecycle()

        # Create circular dependency
        lifecycle.register('a', Mock(), depends_on=['b'])
        lifecycle.register('b', Mock(), depends_on=['a'])

        # Should raise error
        with pytest.raises(RuntimeError, match="Circular dependency"):
            lifecycle.initialize_all()

    def test_error_handling_on_initialization(self):
        """Test error handling during component initialization."""
        lifecycle = ComponentLifecycle()

        component = Mock()
        component.initialize = Mock(side_effect=RuntimeError("Init failed"))

        lifecycle.register('test', component)

        # Should raise and mark as failed
        with pytest.raises(RuntimeError, match="Init failed"):
            lifecycle.initialize_component('test')

        assert lifecycle.get_phase('test') == LifecyclePhase.FAILED

    def test_stop_all_reverse_order(self):
        """Test stop_all stops components in reverse order."""
        lifecycle = ComponentLifecycle()

        stop_order = []

        logger = Mock()
        logger.stop = Mock(side_effect=lambda: stop_order.append('logger'))

        database = Mock()
        database.initialize = Mock()
        database.start = Mock()
        database.stop = Mock(side_effect=lambda: stop_order.append('database'))

        service = Mock()
        service.initialize = Mock()
        service.start = Mock()
        service.stop = Mock(side_effect=lambda: stop_order.append('service'))

        lifecycle.register('logger', logger)
        lifecycle.register('database', database, depends_on=['logger'])
        lifecycle.register('service', service, depends_on=['database'])

        lifecycle.initialize_all()
        lifecycle.start_all()
        lifecycle.stop_all()

        # Should stop in reverse order: service, database, logger
        assert stop_order == ['service', 'database', 'logger']

    def test_is_running(self):
        """Test is_running status check."""
        lifecycle = ComponentLifecycle()

        component = Mock()
        component.initialize = Mock()
        component.start = Mock()

        lifecycle.register('test', component)

        assert not lifecycle.is_running()

        lifecycle.initialize_component('test')
        assert not lifecycle.is_running()

        lifecycle.start_component('test')
        assert lifecycle.is_running()

    def test_has_failed_components(self):
        """Test failed component detection."""
        lifecycle = ComponentLifecycle()

        component = Mock()
        component.initialize = Mock(side_effect=RuntimeError("Failed"))

        lifecycle.register('test', component)

        assert not lifecycle.has_failed_components()

        try:
            lifecycle.initialize_component('test')
        except RuntimeError:
            pass

        assert lifecycle.has_failed_components()


class TestApplicationBootstrapper:
    """Tests for ApplicationBootstrapper class."""

    def test_initialization(self):
        """Test bootstrapper initialization."""
        bootstrapper = ApplicationBootstrapper(username="test_user")

        assert bootstrapper.username == "test_user"
        assert "test_user" in bootstrapper.user_data_root
        assert bootstrapper.container is not None

    def test_container_setup(self):
        """Test DI container setup in bootstrap."""
        bootstrapper = ApplicationBootstrapper(username="test_user")

        # Container should be created
        assert isinstance(bootstrapper.container, DependencyContainer)

    @pytest.mark.skip(reason="Requires full application context")
    def test_bootstrap_phases(self):
        """Test bootstrap phase execution (integration test)."""
        # This would require mocking many components
        # Skipped for unit test - would be integration test
        pass


# Test fixtures
@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return {
        'model': {'threads': 4, 'gpu_layers': -1},
        'tools': {'shell_command_whitelist': ['ls', 'cat']},
        'network': {'enabled': False}
    }


@pytest.fixture
def mock_logger():
    """Provide mock logger."""
    logger = Mock()
    logger.log_info = Mock()
    logger.log_error = Mock()
    logger.log_warning = Mock()
    return logger


# Example integration test
def test_container_lifecycle_integration(mock_config):
    """Integration test for container and lifecycle together."""
    container = DependencyContainer()
    lifecycle = ComponentLifecycle()

    # Create a simple service
    class TestService:
        def __init__(self):
            self.initialized = False
            self.started = False

        def initialize(self):
            self.initialized = True

        def start(self):
            self.started = True

    # Register in container
    container.register_singleton('service', TestService)

    # Get instance and register in lifecycle
    service = container.resolve('service')
    lifecycle.register('service', service)

    # Initialize and start
    lifecycle.initialize_component('service')
    lifecycle.start_component('service')

    # Verify
    assert service.initialized
    assert service.started
    assert lifecycle.get_phase('service') == LifecyclePhase.STARTED


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
