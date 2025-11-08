# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 21: Full Project Remediation & Modernization
# Dependency Injection Container

"""
Dependency injection container for The JENOVA Cognitive Architecture.

This module provides a lightweight dependency injection (DI) container that
manages component creation, dependency resolution, and lifecycle management.
It eliminates the need for manual parameter passing and enables better testing.

The container supports:
    - Singleton and transient service lifetimes
    - Lazy initialization
    - Dependency resolution
    - Factory functions
    - Interface-based registration

Example:
    >>> container = DependencyContainer()
    >>> container.register_singleton('logger', FileLogger, user_data_root="/path")
    >>> container.register('database', Database, depends_on=['logger'])
    >>> logger = container.resolve('logger')
    >>> db = container.resolve('database')  # Automatically injects logger
"""

from typing import Any, Callable, Dict, List, Optional, Type
from enum import Enum, auto


class ServiceLifetime(Enum):
    """
    Service lifetime options for dependency injection.

    Lifetimes:
        SINGLETON: Single instance shared across application
        TRANSIENT: New instance created each time
        SCOPED: Single instance within a scope (future feature)
    """
    SINGLETON = auto()
    TRANSIENT = auto()
    SCOPED = auto()


class ServiceDescriptor:
    """
    Describes how to create and manage a service.

    Attributes:
        service_type: Type or interface identifier
        implementation: Concrete type or factory function
        lifetime: Service lifetime (singleton/transient)
        instance: Cached instance for singletons
        dependencies: List of dependency identifiers
        factory_kwargs: Keyword arguments for factory/constructor
    """

    def __init__(
        self,
        service_type: str,
        implementation: Any,
        lifetime: ServiceLifetime,
        dependencies: Optional[List[str]] = None,
        **factory_kwargs
    ):
        """
        Initialize service descriptor.

        Args:
            service_type: Service identifier
            implementation: Class type or factory function
            lifetime: Service lifetime
            dependencies: List of dependency identifiers
            **factory_kwargs: Additional constructor/factory arguments
        """
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.instance: Optional[Any] = None
        self.dependencies = dependencies or []
        self.factory_kwargs = factory_kwargs


class DependencyContainer:
    """
    Lightweight dependency injection container.

    This container manages service registration, dependency resolution, and
    instance lifecycle. It supports both singleton and transient lifetimes,
    automatic dependency injection, and factory functions.

    Attributes:
        services: Registry of service descriptors
        _resolution_stack: Stack for circular dependency detection

    Example:
        >>> container = DependencyContainer()
        >>>
        >>> # Register singleton logger
        >>> container.register_singleton('logger', FileLogger, user_data_root="/path")
        >>>
        >>> # Register database with logger dependency
        >>> container.register('database', Database, depends_on=['logger'])
        >>>
        >>> # Resolve services (dependencies auto-injected)
        >>> logger = container.resolve('logger')
        >>> db = container.resolve('database')
        >>>
        >>> # Register factory function
        >>> def create_cache(logger):
        >>>     return Cache(logger=logger, size=1000)
        >>> container.register_factory('cache', create_cache, depends_on=['logger'])
    """

    def __init__(self):
        """Initialize dependency container."""
        self.services: Dict[str, ServiceDescriptor] = {}
        self._resolution_stack: List[str] = []

    def register(
        self,
        service_type: str,
        implementation: Type,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        depends_on: Optional[List[str]] = None,
        **factory_kwargs
    ) -> None:
        """
        Register a service with the container.

        Args:
            service_type: Unique service identifier
            implementation: Class to instantiate
            lifetime: Service lifetime (singleton/transient)
            depends_on: List of dependency identifiers
            **factory_kwargs: Additional constructor arguments

        Raises:
            ValueError: If service already registered
        """
        if service_type in self.services:
            raise ValueError(f"Service '{service_type}' already registered")

        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifetime=lifetime,
            dependencies=depends_on,
            **factory_kwargs
        )
        self.services[service_type] = descriptor

    def register_singleton(
        self,
        service_type: str,
        implementation: Type,
        depends_on: Optional[List[str]] = None,
        **factory_kwargs
    ) -> None:
        """
        Register a singleton service.

        Singleton services are created once and shared across the application.

        Args:
            service_type: Unique service identifier
            implementation: Class to instantiate
            depends_on: List of dependency identifiers
            **factory_kwargs: Additional constructor arguments
        """
        self.register(
            service_type=service_type,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            depends_on=depends_on,
            **factory_kwargs
        )

    def register_transient(
        self,
        service_type: str,
        implementation: Type,
        depends_on: Optional[List[str]] = None,
        **factory_kwargs
    ) -> None:
        """
        Register a transient service.

        Transient services are created anew each time they are resolved.

        Args:
            service_type: Unique service identifier
            implementation: Class to instantiate
            depends_on: List of dependency identifiers
            **factory_kwargs: Additional constructor arguments
        """
        self.register(
            service_type=service_type,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            depends_on=depends_on,
            **factory_kwargs
        )

    def register_factory(
        self,
        service_type: str,
        factory: Callable,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        depends_on: Optional[List[str]] = None
    ) -> None:
        """
        Register a factory function for creating services.

        The factory function will be called with resolved dependencies as arguments.

        Args:
            service_type: Unique service identifier
            factory: Factory function that creates the service
            lifetime: Service lifetime
            depends_on: List of dependency identifiers passed to factory

        Example:
            >>> def create_logger(config):
            >>>     return FileLogger(config['log_path'])
            >>> container.register_factory('logger', create_logger, depends_on=['config'])
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=factory,
            lifetime=lifetime,
            dependencies=depends_on
        )
        self.services[service_type] = descriptor

    def register_instance(self, service_type: str, instance: Any) -> None:
        """
        Register an existing instance as a singleton.

        This is useful for registering pre-configured objects or external services.

        Args:
            service_type: Unique service identifier
            instance: Existing instance to register

        Example:
            >>> config = load_config()
            >>> container.register_instance('config', config)
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON
        )
        descriptor.instance = instance
        self.services[service_type] = descriptor

    def resolve(self, service_type: str) -> Any:
        """
        Resolve a service and its dependencies.

        This method retrieves or creates a service instance, automatically
        resolving and injecting all dependencies.

        Args:
            service_type: Service identifier to resolve

        Returns:
            Service instance with all dependencies injected

        Raises:
            KeyError: If service not registered
            RuntimeError: If circular dependency detected
        """
        if service_type not in self.services:
            raise KeyError(f"Service '{service_type}' not registered")

        # Check for circular dependencies
        if service_type in self._resolution_stack:
            cycle = ' -> '.join(self._resolution_stack + [service_type])
            raise RuntimeError(f"Circular dependency detected: {cycle}")

        descriptor = self.services[service_type]

        # Return cached singleton instance if exists
        if descriptor.lifetime == ServiceLifetime.SINGLETON and descriptor.instance is not None:
            return descriptor.instance

        # Add to resolution stack
        self._resolution_stack.append(service_type)

        try:
            # Resolve dependencies
            resolved_deps = {}
            for dep in descriptor.dependencies:
                resolved_deps[dep] = self.resolve(dep)

            # Create instance
            if callable(descriptor.implementation):
                # Check if it's a factory function or class
                if hasattr(descriptor.implementation, '__self__'):
                    # It's a bound method (factory)
                    instance = descriptor.implementation(**resolved_deps)
                elif descriptor.dependencies:
                    # It's a factory function with dependencies
                    instance = descriptor.implementation(**resolved_deps)
                else:
                    # It's a class constructor
                    instance = descriptor.implementation(
                        **resolved_deps,
                        **descriptor.factory_kwargs
                    )
            else:
                raise ValueError(
                    f"Invalid implementation for service '{service_type}': "
                    f"must be callable (class or factory function)"
                )

            # Cache singleton instance
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                descriptor.instance = instance

            return instance

        finally:
            # Remove from resolution stack
            self._resolution_stack.pop()

    def resolve_all(self) -> Dict[str, Any]:
        """
        Resolve all registered services.

        Returns:
            Dictionary mapping service identifiers to instances

        Example:
            >>> all_services = container.resolve_all()
            >>> logger = all_services['logger']
        """
        resolved = {}
        for service_type in self.services:
            try:
                resolved[service_type] = self.resolve(service_type)
            except Exception as e:
                # Skip services that fail to resolve
                resolved[service_type] = f"<Resolution failed: {e}>"
        return resolved

    def is_registered(self, service_type: str) -> bool:
        """
        Check if a service is registered.

        Args:
            service_type: Service identifier

        Returns:
            True if service is registered
        """
        return service_type in self.services

    def unregister(self, service_type: str) -> None:
        """
        Unregister a service from the container.

        Args:
            service_type: Service identifier to unregister

        Raises:
            KeyError: If service not registered
        """
        if service_type not in self.services:
            raise KeyError(f"Service '{service_type}' not registered")
        del self.services[service_type]

    def clear(self) -> None:
        """Clear all registered services."""
        self.services.clear()

    def get_registered_services(self) -> List[str]:
        """
        Get list of all registered service identifiers.

        Returns:
            List of service identifiers
        """
        return list(self.services.keys())

    def get_service_info(self, service_type: str) -> Dict[str, Any]:
        """
        Get information about a registered service.

        Args:
            service_type: Service identifier

        Returns:
            Dictionary containing service information

        Raises:
            KeyError: If service not registered
        """
        if service_type not in self.services:
            raise KeyError(f"Service '{service_type}' not registered")

        descriptor = self.services[service_type]
        return {
            'service_type': descriptor.service_type,
            'implementation': descriptor.implementation.__name__,
            'lifetime': descriptor.lifetime.name,
            'dependencies': descriptor.dependencies,
            'is_resolved': descriptor.instance is not None
        }
