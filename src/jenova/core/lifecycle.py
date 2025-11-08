# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 21: Full Project Remediation & Modernization
# Component Lifecycle Management

"""
Component lifecycle management for The JENOVA Cognitive Architecture.

This module provides structured lifecycle management for application components,
ensuring proper initialization, startup, shutdown, and cleanup phases.

The lifecycle follows these phases:
    1. CREATED: Component instance created but not initialized
    2. INITIALIZED: Component initialized with configuration
    3. STARTED: Component actively running
    4. STOPPED: Component gracefully stopped
    5. DISPOSED: Component resources released
"""

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol
from jenova.infrastructure import ErrorHandler, ErrorSeverity


class LifecyclePhase(Enum):
    """
    Lifecycle phases for application components.

    Phases:
        CREATED: Component instance created
        INITIALIZED: Component initialized with configuration
        STARTED: Component actively running
        STOPPED: Component gracefully stopped
        DISPOSED: Component resources released
        FAILED: Component encountered fatal error
    """
    CREATED = auto()
    INITIALIZED = auto()
    STARTED = auto()
    STOPPED = auto()
    DISPOSED = auto()
    FAILED = auto()


class LifecycleAware(Protocol):
    """
    Protocol for components that implement lifecycle methods.

    Components implementing this protocol can participate in the
    application lifecycle management system.
    """

    def initialize(self) -> None:
        """Initialize component with configuration."""
        ...

    def start(self) -> None:
        """Start component operation."""
        ...

    def stop(self) -> None:
        """Stop component operation gracefully."""
        ...

    def dispose(self) -> None:
        """Release component resources."""
        ...


class ComponentLifecycle:
    """
    Manages lifecycle of application components.

    This class tracks component states and orchestrates lifecycle transitions,
    ensuring components are initialized, started, stopped, and disposed in
    the correct order with proper error handling.

    Attributes:
        components: Dictionary mapping component names to instances
        phases: Dictionary tracking current phase of each component
        dependencies: Dictionary mapping components to their dependencies
        error_handler: Optional error handler for lifecycle errors

    Example:
        >>> lifecycle = ComponentLifecycle()
        >>> lifecycle.register("logger", logger_instance)
        >>> lifecycle.register("database", db_instance, depends_on=["logger"])
        >>> lifecycle.initialize_all()
        >>> lifecycle.start_all()
        >>> # ... application runs ...
        >>> lifecycle.stop_all()
        >>> lifecycle.dispose_all()
    """

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """
        Initialize component lifecycle manager.

        Args:
            error_handler: Optional error handler for lifecycle errors
        """
        self.components: Dict[str, Any] = {}
        self.phases: Dict[str, LifecyclePhase] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.error_handler = error_handler
        self._initialization_order: List[str] = []

    def register(
        self,
        name: str,
        component: Any,
        depends_on: Optional[List[str]] = None
    ) -> None:
        """
        Register a component for lifecycle management.

        Args:
            name: Unique component identifier
            component: Component instance to manage
            depends_on: List of component names this component depends on

        Raises:
            ValueError: If component name already registered
        """
        if name in self.components:
            raise ValueError(f"Component '{name}' already registered")

        self.components[name] = component
        self.phases[name] = LifecyclePhase.CREATED
        self.dependencies[name] = depends_on or []

    def unregister(self, name: str) -> None:
        """
        Unregister a component from lifecycle management.

        Args:
            name: Component identifier to unregister

        Raises:
            KeyError: If component not found
        """
        if name not in self.components:
            raise KeyError(f"Component '{name}' not registered")

        # Ensure component is disposed before unregistering
        if self.phases[name] not in (LifecyclePhase.DISPOSED, LifecyclePhase.FAILED):
            self.dispose_component(name)

        del self.components[name]
        del self.phases[name]
        del self.dependencies[name]

    def get_component(self, name: str) -> Any:
        """
        Get registered component by name.

        Args:
            name: Component identifier

        Returns:
            Component instance

        Raises:
            KeyError: If component not found
        """
        if name not in self.components:
            raise KeyError(f"Component '{name}' not registered")
        return self.components[name]

    def get_phase(self, name: str) -> LifecyclePhase:
        """
        Get current lifecycle phase of component.

        Args:
            name: Component identifier

        Returns:
            Current lifecycle phase

        Raises:
            KeyError: If component not found
        """
        if name not in self.phases:
            raise KeyError(f"Component '{name}' not registered")
        return self.phases[name]

    def _resolve_initialization_order(self) -> List[str]:
        """
        Resolve component initialization order based on dependencies.

        Uses topological sort to ensure dependencies are initialized first.

        Returns:
            List of component names in initialization order

        Raises:
            ValueError: If circular dependencies detected
        """
        # Topological sort using DFS
        visited = set()
        temp_visited = set()
        order = []

        def visit(name: str) -> None:
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{name}'")
            if name in visited:
                return

            temp_visited.add(name)
            for dep in self.dependencies.get(name, []):
                visit(dep)
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)

        for component_name in self.components:
            if component_name not in visited:
                visit(component_name)

        return order

    def initialize_all(self) -> None:
        """
        Initialize all registered components in dependency order.

        Components are initialized according to their dependency graph,
        ensuring dependencies are initialized before dependent components.

        Raises:
            RuntimeError: If initialization fails for any component
        """
        # Resolve initialization order
        try:
            self._initialization_order = self._resolve_initialization_order()
        except ValueError as e:
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    context="lifecycle_initialization_order",
                    severity=ErrorSeverity.CRITICAL
                )
            raise RuntimeError(f"Failed to resolve component dependencies: {e}")

        # Initialize components in order
        for name in self._initialization_order:
            self.initialize_component(name)

    def initialize_component(self, name: str) -> None:
        """
        Initialize a specific component.

        Args:
            name: Component identifier

        Raises:
            RuntimeError: If initialization fails
        """
        if name not in self.components:
            raise KeyError(f"Component '{name}' not registered")

        component = self.components[name]
        current_phase = self.phases[name]

        # Check if already initialized
        if current_phase != LifecyclePhase.CREATED:
            return

        # Check dependencies are initialized
        for dep in self.dependencies[name]:
            dep_phase = self.phases.get(dep, LifecyclePhase.CREATED)
            if dep_phase == LifecyclePhase.CREATED:
                raise RuntimeError(
                    f"Cannot initialize '{name}': dependency '{dep}' not initialized"
                )

        # Initialize component
        try:
            if hasattr(component, 'initialize'):
                component.initialize()
            self.phases[name] = LifecyclePhase.INITIALIZED
        except Exception as e:
            self.phases[name] = LifecyclePhase.FAILED
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    context=f"lifecycle_initialize_{name}",
                    severity=ErrorSeverity.CRITICAL
                )
            raise RuntimeError(f"Failed to initialize component '{name}': {e}") from e

    def start_all(self) -> None:
        """
        Start all initialized components in dependency order.

        Raises:
            RuntimeError: If starting fails for any component
        """
        for name in self._initialization_order:
            self.start_component(name)

    def start_component(self, name: str) -> None:
        """
        Start a specific component.

        Args:
            name: Component identifier

        Raises:
            RuntimeError: If starting fails or component not initialized
        """
        if name not in self.components:
            raise KeyError(f"Component '{name}' not registered")

        component = self.components[name]
        current_phase = self.phases[name]

        # Check if initialized
        if current_phase != LifecyclePhase.INITIALIZED:
            if current_phase == LifecyclePhase.STARTED:
                return  # Already started
            raise RuntimeError(
                f"Cannot start '{name}': component not initialized (phase: {current_phase.name})"
            )

        # Start component
        try:
            if hasattr(component, 'start'):
                component.start()
            self.phases[name] = LifecyclePhase.STARTED
        except Exception as e:
            self.phases[name] = LifecyclePhase.FAILED
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    context=f"lifecycle_start_{name}",
                    severity=ErrorSeverity.CRITICAL
                )
            raise RuntimeError(f"Failed to start component '{name}': {e}") from e

    def stop_all(self) -> None:
        """
        Stop all running components in reverse dependency order.

        Components are stopped in reverse order to ensure dependent components
        are stopped before their dependencies.
        """
        # Stop in reverse order
        for name in reversed(self._initialization_order):
            try:
                self.stop_component(name)
            except Exception as e:
                # Log error but continue stopping other components
                if self.error_handler:
                    self.error_handler.handle_error(
                        error=e,
                        context=f"lifecycle_stop_{name}",
                        severity=ErrorSeverity.HIGH
                    )

    def stop_component(self, name: str) -> None:
        """
        Stop a specific component.

        Args:
            name: Component identifier
        """
        if name not in self.components:
            return  # Component not registered, nothing to stop

        component = self.components[name]
        current_phase = self.phases[name]

        # Only stop if started
        if current_phase != LifecyclePhase.STARTED:
            return

        # Stop component
        try:
            if hasattr(component, 'stop'):
                component.stop()
            self.phases[name] = LifecyclePhase.STOPPED
        except Exception as e:
            self.phases[name] = LifecyclePhase.FAILED
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    context=f"lifecycle_stop_{name}",
                    severity=ErrorSeverity.HIGH
                )
            # Don't re-raise, allow other components to stop

    def dispose_all(self) -> None:
        """
        Dispose all components in reverse dependency order.

        This releases all resources held by components. Components are disposed
        in reverse order to ensure dependent components are disposed before
        their dependencies.
        """
        # Dispose in reverse order
        for name in reversed(self._initialization_order):
            try:
                self.dispose_component(name)
            except Exception as e:
                # Log error but continue disposing other components
                if self.error_handler:
                    self.error_handler.handle_error(
                        error=e,
                        context=f"lifecycle_dispose_{name}",
                        severity=ErrorSeverity.MEDIUM
                    )

    def dispose_component(self, name: str) -> None:
        """
        Dispose a specific component.

        Args:
            name: Component identifier
        """
        if name not in self.components:
            return  # Component not registered, nothing to dispose

        component = self.components[name]
        current_phase = self.phases[name]

        # Stop component if running
        if current_phase == LifecyclePhase.STARTED:
            self.stop_component(name)

        # Dispose component
        try:
            if hasattr(component, 'dispose'):
                component.dispose()
            self.phases[name] = LifecyclePhase.DISPOSED
        except Exception as e:
            self.phases[name] = LifecyclePhase.FAILED
            if self.error_handler:
                self.error_handler.handle_error(
                    error=e,
                    context=f"lifecycle_dispose_{name}",
                    severity=ErrorSeverity.MEDIUM
                )
            # Don't re-raise, allow other components to dispose

    def get_status(self) -> Dict[str, str]:
        """
        Get status of all components.

        Returns:
            Dictionary mapping component names to their current phases
        """
        return {name: phase.name for name, phase in self.phases.items()}

    def is_running(self) -> bool:
        """
        Check if application is fully running.

        Returns:
            True if all components are in STARTED phase
        """
        return all(phase == LifecyclePhase.STARTED for phase in self.phases.values())

    def has_failed_components(self) -> bool:
        """
        Check if any components have failed.

        Returns:
            True if any component is in FAILED phase
        """
        return any(phase == LifecyclePhase.FAILED for phase in self.phases.values())
