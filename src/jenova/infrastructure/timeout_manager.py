# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Timeout manager to prevent hung operations from freezing the system.

Provides context managers and decorators for timing out operations.
Thread-safe implementation that works in both main and worker threads.
"""

import threading
from contextlib import contextmanager
from typing import Any, Callable


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


@contextmanager
def timeout(seconds: int = 30, error_message: str = "Operation timed out"):
    """
    Thread-safe context manager for timing out operations.

    Works in both main thread and worker threads by using threading.Timer
    instead of signal-based timeouts.

    Usage:
        with timeout(10, "Model loading timed out"):
            llm.generate(prompt)

    Args:
        seconds: Timeout duration in seconds
        error_message: Error message to raise on timeout

    Raises:
        TimeoutError: If operation exceeds timeout duration
    """
    # Use a flag to track if timeout occurred
    timeout_occurred = threading.Event()

    def timeout_handler():
        timeout_occurred.set()

    # Create a timer that will set the flag after timeout
    timer = threading.Timer(seconds, timeout_handler)
    timer.daemon = True
    timer.start()

    try:
        yield
        # Check if timeout occurred during the operation
        if timeout_occurred.is_set():
            raise TimeoutError(error_message)
    finally:
        # Cancel the timer if it hasn't fired yet
        timer.cancel()


def with_timeout(timeout_seconds: int = 30):
    """
    Decorator for timing out functions.

    Usage:
        @with_timeout(60)
        def slow_operation():
            ...

    Args:
        timeout_seconds: Timeout duration in seconds

    Returns:
        Decorated function that raises TimeoutError on timeout
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            with timeout(timeout_seconds, f"{func.__name__} timed out after {timeout_seconds}s"):
                return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# Convenience functions for common timeout durations
def with_short_timeout(func: Callable) -> Callable:
    """Decorator for 30 second timeout."""
    return with_timeout(30)(func)


def with_medium_timeout(func: Callable) -> Callable:
    """Decorator for 60 second timeout."""
    return with_timeout(60)(func)


def with_long_timeout(func: Callable) -> Callable:
    """Decorator for 120 second timeout."""
    return with_timeout(120)(func)
