# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Circuit Breaker Pattern Implementation for JENOVA Cognitive Architecture.

This module implements the circuit breaker pattern to prevent cascading failures
in LLM operations, network calls, and other external dependencies.

The circuit breaker protects the system by:
- Detecting failures and opening the circuit after threshold is exceeded
- Preventing requests to failing services (fail-fast)
- Automatically attempting recovery after cooldown period
- Providing metrics and observability

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed

Phase 20 Feature #1: Circuit Breaker Pattern for Resilience
- Prevents cascading failures when LLM or network operations fail repeatedly
- Configurable failure thresholds and recovery timeouts
- Comprehensive metrics tracking
- Thread-safe implementation
"""

import time
import threading
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before attempting recovery
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 30.0  # Operation timeout in seconds
    expected_exceptions: tuple = (Exception,)  # Exceptions that count as failures


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_transitions: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_failure_count: int = 0
    current_success_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "failure_rate": self.failed_requests / max(1, self.total_requests),
            "rejection_rate": self.rejected_requests / max(1, self.total_requests),
            "state_transitions": self.state_transitions,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "current_failure_count": self.current_failure_count,
            "current_success_count": self.current_success_count,
        }


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Protects operations by monitoring failures and opening the circuit
    when failure threshold is exceeded. Automatically attempts recovery
    after timeout period.

    Thread-safe implementation suitable for concurrent operations.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit (for logging and metrics)
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._lock = threading.RLock()
        self._opened_at: Optional[float] = None

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        with self._lock:
            return self._metrics

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function execution

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        with self._lock:
            self._metrics.total_requests += 1

            # Check if circuit should be closed
            self._check_state()

            # If circuit is open, reject immediately
            if self._state == CircuitState.OPEN:
                self._metrics.rejected_requests += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service is unavailable. Will retry after recovery timeout."
                )

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exceptions as e:
            self._on_failure()
            raise

    def _check_state(self) -> None:
        """Check if circuit state should transition."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self._opened_at and (time.time() - self._opened_at >= self.config.recovery_timeout):
                self._transition_to_half_open()

    def _on_success(self) -> None:
        """Handle successful operation."""
        with self._lock:
            self._metrics.successful_requests += 1
            self._metrics.last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._metrics.current_success_count += 1

                # If enough successes, close circuit
                if self._metrics.current_success_count >= self.config.success_threshold:
                    self._transition_to_closed()

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._metrics.current_failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed operation."""
        with self._lock:
            self._metrics.failed_requests += 1
            self._metrics.last_failure_time = datetime.now()
            self._metrics.current_failure_count += 1

            # If in half-open state, any failure reopens circuit
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()

            # If in closed state, check if threshold exceeded
            elif self._state == CircuitState.CLOSED:
                if self._metrics.current_failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        self._metrics.state_transitions += 1
        self._metrics.current_success_count = 0

        logger.warning(
            f"Circuit breaker '{self.name}' transitioned to OPEN. "
            f"Failure threshold ({self.config.failure_threshold}) exceeded. "
            f"Will attempt recovery in {self.config.recovery_timeout}s"
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        self._state = CircuitState.HALF_OPEN
        self._metrics.state_transitions += 1
        self._metrics.current_failure_count = 0
        self._metrics.current_success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned to HALF_OPEN. "
            f"Testing service recovery..."
        )

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._metrics.state_transitions += 1
        self._metrics.current_failure_count = 0
        self._metrics.current_success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned to CLOSED. "
            f"Service recovered successfully."
        )

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state.

        This should be used carefully, typically only for testing or
        administrative purposes.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._metrics.current_failure_count = 0
            self._metrics.current_success_count = 0

            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of circuit breaker.

        Returns:
            Dictionary with state and metrics
        """
        with self._lock:
            status = {
                "name": self.name,
                "state": self._state.value,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                },
                "metrics": self._metrics.to_dict(),
            }

            # Add time until recovery if circuit is open
            if self._state == CircuitState.OPEN and self._opened_at:
                time_since_open = time.time() - self._opened_at
                time_until_recovery = max(0, self.config.recovery_timeout - time_since_open)
                status["recovery_in_seconds"] = time_until_recovery

            return status


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> Callable:
    """
    Decorator to protect functions with circuit breaker.

    Args:
        name: Name for the circuit breaker
        config: Circuit breaker configuration

    Returns:
        Decorated function

    Example:
        @circuit_breaker("llm_generation")
        def generate_text(prompt: str) -> str:
            return llm.generate(prompt)
    """
    breaker = CircuitBreaker(name, config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return breaker.call(func, *args, **kwargs)

        # Attach circuit breaker instance to function for access
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator


class CircuitBreakerRegistry:
    """
    Global registry for circuit breakers.

    Manages all circuit breakers in the application, providing
    centralized access to metrics and state.
    """

    def __init__(self):
        """Initialize circuit breaker registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def register(self, breaker: CircuitBreaker) -> None:
        """
        Register a circuit breaker.

        Args:
            breaker: Circuit breaker to register
        """
        with self._lock:
            if breaker.name in self._breakers:
                logger.warning(
                    f"Circuit breaker '{breaker.name}' already registered. Replacing."
                )
            self._breakers[breaker.name] = breaker
            logger.debug(f"Registered circuit breaker: {breaker.name}")

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            CircuitBreaker if found, None otherwise
        """
        with self._lock:
            return self._breakers.get(name)

    def get_all(self) -> Dict[str, CircuitBreaker]:
        """
        Get all registered circuit breakers.

        Returns:
            Dictionary of circuit breakers by name
        """
        with self._lock:
            return self._breakers.copy()

    def get_status_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all circuit breakers.

        Returns:
            Dictionary with status for each circuit breaker
        """
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> int:
        """
        Reset all circuit breakers to CLOSED state.

        Returns:
            Number of circuit breakers reset
        """
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            return len(self._breakers)


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_registry() -> CircuitBreakerRegistry:
    """
    Get global circuit breaker registry.

    Returns:
        Global CircuitBreakerRegistry instance
    """
    return _registry
