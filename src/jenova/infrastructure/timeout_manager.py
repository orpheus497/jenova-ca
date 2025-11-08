# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Timeout manager to prevent hung operations from freezing the system.

Provides context managers and decorators for timing out operations with multiple strategies:
- Signal-based timeout (Unix/Linux only): True interruption of blocking operations
- Thread-based soft timeout (Cross-platform): Checks timeout after operation completes
- Auto strategy: Automatically selects the best strategy for the platform

Thread-safe implementation that works in both main and worker threads.
"""

import os
import platform
import signal
import threading
from contextlib import contextmanager
from typing import Any, Callable, Literal

# Type alias for timeout strategies
TimeoutStrategy = Literal["signal", "thread", "auto"]


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


# Determine if we're on a Unix-like system that supports signals
_IS_UNIX = platform.system() in ('Linux', 'Darwin', 'FreeBSD', 'OpenBSD')


@contextmanager
def timeout(
    seconds: int = 30,
    error_message: str = "Operation timed out",
    strategy: TimeoutStrategy = "auto"
):
    """
    Context manager for timing out operations with multiple strategies.

    Strategies:
    -----------
    1. "signal" (Unix/Linux only):
       - Uses SIGALRM to truly interrupt blocking operations
       - Works only in main thread
       - Provides hard timeout - operation is interrupted mid-execution
       - Fastest and most reliable on Unix systems
       - Raises TimeoutError immediately when timeout expires

    2. "thread" (Cross-platform):
       - Uses threading.Timer to set a flag after timeout
       - Works in both main and worker threads
       - Provides soft timeout - only checks after operation completes
       - Operation may run longer than timeout before check occurs
       - Compatible with Windows and all platforms

    3. "auto" (Recommended):
       - Automatically selects "signal" on Unix/Linux (in main thread)
       - Falls back to "thread" on Windows or in worker threads
       - Provides best performance while maintaining compatibility

    Usage Examples:
    --------------
    ```python
    # Auto strategy (recommended)
    with timeout(10, "Model loading timed out"):
        llm.generate(prompt)

    # Force signal-based timeout (Unix only, main thread)
    with timeout(30, "Generation timed out", strategy="signal"):
        result = llm.generate(prompt)

    # Force thread-based soft timeout (cross-platform, any thread)
    with timeout(60, "Search timed out", strategy="thread"):
        results = memory.search(query)
    ```

    Args:
        seconds: Timeout duration in seconds
        error_message: Error message to raise on timeout
        strategy: Timeout strategy - "signal", "thread", or "auto"

    Raises:
        TimeoutError: If operation exceeds timeout duration
        ValueError: If invalid strategy specified or signal used inappropriately

    Notes:
    ------
    - Signal-based timeout only works in the main thread on Unix systems
    - Thread-based timeout is a "soft" timeout that checks after completion
    - For long-running operations, consider using async/await patterns instead
    """
    # Validate strategy
    if strategy not in ("signal", "thread", "auto"):
        raise ValueError(f"Invalid timeout strategy: {strategy}. Must be 'signal', 'thread', or 'auto'")

    # Determine actual strategy to use
    actual_strategy = _resolve_strategy(strategy)

    if actual_strategy == "signal":
        # Use signal-based timeout (Unix only, main thread)
        return _signal_timeout(seconds, error_message)
    else:
        # Use thread-based soft timeout (cross-platform)
        return _thread_timeout(seconds, error_message)


def _resolve_strategy(strategy: TimeoutStrategy) -> Literal["signal", "thread"]:
    """
    Resolve timeout strategy based on platform and thread context.

    Args:
        strategy: Requested strategy ("signal", "thread", or "auto")

    Returns:
        Actual strategy to use ("signal" or "thread")

    Raises:
        ValueError: If signal strategy requested but not available
    """
    if strategy == "thread":
        return "thread"

    if strategy == "signal":
        # Validate that signal strategy is available
        if not _IS_UNIX:
            raise ValueError("Signal-based timeout is only available on Unix/Linux systems")
        if threading.current_thread() != threading.main_thread():
            raise ValueError("Signal-based timeout only works in the main thread")
        return "signal"

    # Auto strategy: use signal if available, otherwise thread
    if _IS_UNIX and threading.current_thread() == threading.main_thread():
        return "signal"
    return "thread"


@contextmanager
def _signal_timeout(seconds: int, error_message: str):
    """
    Signal-based timeout using SIGALRM (Unix/Linux only).

    This provides a true hard timeout that interrupts the operation
    mid-execution. Only works in the main thread on Unix systems.

    Args:
        seconds: Timeout duration in seconds
        error_message: Error message to raise on timeout

    Raises:
        TimeoutError: If operation exceeds timeout duration
    """
    def _timeout_handler(signum, frame):
        raise TimeoutError(error_message)

    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def _thread_timeout(seconds: int, error_message: str):
    """
    Thread-based soft timeout using threading.Timer (cross-platform).

    This provides a "soft" timeout that sets a flag after the specified
    duration but only checks the flag after the operation completes.
    The operation may run longer than the timeout before the check occurs.

    Works in both main thread and worker threads on all platforms.

    Args:
        seconds: Timeout duration in seconds
        error_message: Error message to raise on timeout

    Raises:
        TimeoutError: If timeout flag is set after operation completes
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
        # Note: This is a "soft" timeout - the operation may have run longer
        # than the specified duration before this check occurs
        if timeout_occurred.is_set():
            raise TimeoutError(f"{error_message} (soft timeout - operation completed but exceeded {seconds}s)")
    finally:
        # Cancel the timer if it hasn't fired yet
        timer.cancel()


def with_timeout(timeout_seconds: int = 30, strategy: TimeoutStrategy = "auto"):
    """
    Decorator for timing out functions.

    Usage:
    ------
    ```python
    @with_timeout(60)
    def slow_operation():
        # Long-running operation
        ...

    @with_timeout(30, strategy="thread")
    def threaded_operation():
        # Operation that runs in worker thread
        ...
    ```

    Args:
        timeout_seconds: Timeout duration in seconds
        strategy: Timeout strategy - "signal", "thread", or "auto"

    Returns:
        Decorated function that raises TimeoutError on timeout
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            with timeout(
                timeout_seconds,
                f"{func.__name__} timed out after {timeout_seconds}s",
                strategy=strategy
            ):
                return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# Convenience functions for common timeout durations
def with_short_timeout(func: Callable, strategy: TimeoutStrategy = "auto") -> Callable:
    """Decorator for 30 second timeout."""
    return with_timeout(30, strategy=strategy)(func)


def with_medium_timeout(func: Callable, strategy: TimeoutStrategy = "auto") -> Callable:
    """Decorator for 60 second timeout."""
    return with_timeout(60, strategy=strategy)(func)


def with_long_timeout(func: Callable, strategy: TimeoutStrategy = "auto") -> Callable:
    """Decorator for 120 second timeout."""
    return with_timeout(120, strategy=strategy)(func)


# Utility function to check timeout capabilities
def get_timeout_info() -> dict[str, Any]:
    """
    Get information about timeout capabilities on this platform.

    Returns:
        Dictionary with timeout capability information:
        - platform: Operating system name
        - signal_available: Whether signal-based timeout is available
        - main_thread: Whether currently in main thread
        - recommended_strategy: Recommended timeout strategy
    """
    is_main_thread = threading.current_thread() == threading.main_thread()

    return {
        "platform": platform.system(),
        "signal_available": _IS_UNIX and is_main_thread,
        "main_thread": is_main_thread,
        "recommended_strategy": "signal" if (_IS_UNIX and is_main_thread) else "thread"
    }
