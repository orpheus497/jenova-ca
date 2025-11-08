"""
JENOVA Cognitive Architecture - Hooks System Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides an event-driven hooks system for automation.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


logger = logging.getLogger(__name__)


class HookTiming(Enum):
    """When a hook should be executed."""
    PRE = "pre"
    POST = "post"
    ON_ERROR = "on_error"


@dataclass
class HookResult:
    """Result of hook execution."""

    hook_id: str
    event: str
    timing: HookTiming
    success: bool
    result: Optional[Any] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hook_id": self.hook_id,
            "event": self.event,
            "timing": self.timing.value,
            "success": self.success,
            "result": str(self.result) if self.result is not None else None,
            "error": str(self.error) if self.error else None,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Hook:
    """Represents a registered hook."""

    hook_id: str
    event: str
    timing: HookTiming
    callback: Callable
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare by priority (higher priority first)."""
        return self.priority > other.priority


class HooksSystem:
    """
    Event-driven hooks system for automation.

    Features:
    - Pre/post/error hooks
    - Priority-based execution
    - Hook enabling/disabling
    - Execution history
    - Error handling and recovery
    """

    def __init__(self, enable_history: bool = True, history_limit: int = 100):
        """
        Initialize the hooks system.

        Args:
            enable_history: Whether to keep execution history
            history_limit: Maximum number of history entries to keep
        """
        self.hooks: Dict[str, Dict[HookTiming, List[Hook]]] = defaultdict(
            lambda: {
                HookTiming.PRE: [],
                HookTiming.POST: [],
                HookTiming.ON_ERROR: []
            }
        )
        self.enable_history = enable_history
        self.history_limit = history_limit
        self.execution_history: List[HookResult] = []
        self._next_hook_id = 0

    def register(
        self,
        event: str,
        callback: Callable,
        timing: HookTiming = HookTiming.POST,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a hook for an event.

        Args:
            event: Event name to hook into
            callback: Function to call when event occurs
            timing: When to execute (pre/post/on_error)
            priority: Execution priority (higher = earlier)
            metadata: Optional metadata for the hook

        Returns:
            Hook ID
        """
        hook_id = f"hook_{self._next_hook_id}"
        self._next_hook_id += 1

        hook = Hook(
            hook_id=hook_id,
            event=event,
            timing=timing,
            callback=callback,
            priority=priority,
            metadata=metadata or {}
        )

        # Add to appropriate list and sort by priority
        self.hooks[event][timing].append(hook)
        self.hooks[event][timing].sort()

        logger.info(f"Registered {timing.value} hook for event '{event}': {hook_id}")
        return hook_id

    def unregister(self, hook_id: str) -> bool:
        """
        Unregister a hook.

        Args:
            hook_id: Hook ID to unregister

        Returns:
            True if unregistered, False if not found
        """
        for event, timings in self.hooks.items():
            for timing, hooks in timings.items():
                for i, hook in enumerate(hooks):
                    if hook.hook_id == hook_id:
                        hooks.pop(i)
                        logger.info(f"Unregistered hook: {hook_id}")
                        return True
        return False

    def trigger(
        self,
        event: str,
        timing: HookTiming = HookTiming.POST,
        context: Optional[Dict[str, Any]] = None,
        stop_on_error: bool = False
    ) -> List[HookResult]:
        """
        Trigger hooks for an event.

        Args:
            event: Event name
            timing: Hook timing to trigger
            context: Optional context to pass to hooks
            stop_on_error: Whether to stop execution on first error

        Returns:
            List of HookResult objects
        """
        context = context or {}
        results = []

        # Get hooks for this event and timing
        hooks = self.hooks.get(event, {}).get(timing, [])

        if not hooks:
            logger.debug(f"No {timing.value} hooks registered for event '{event}'")
            return results

        logger.info(f"Triggering {len(hooks)} {timing.value} hooks for event '{event}'")

        for hook in hooks:
            if not hook.enabled:
                logger.debug(f"Skipping disabled hook: {hook.hook_id}")
                continue

            result = self._execute_hook(hook, context)
            results.append(result)

            if not result.success and stop_on_error:
                logger.warning(f"Stopping hook execution due to error in {hook.hook_id}")
                break

        # Add to history
        if self.enable_history:
            self.execution_history.extend(results)
            self._trim_history()

        return results

    def _execute_hook(self, hook: Hook, context: Dict[str, Any]) -> HookResult:
        """
        Execute a single hook.

        Args:
            hook: Hook to execute
            context: Context to pass to hook

        Returns:
            HookResult
        """
        import time

        start_time = time.time()
        result = HookResult(
            hook_id=hook.hook_id,
            event=hook.event,
            timing=hook.timing,
            success=False
        )

        try:
            logger.debug(f"Executing hook {hook.hook_id}")
            output = hook.callback(context)
            result.result = output
            result.success = True
            logger.debug(f"Hook {hook.hook_id} executed successfully")

        except Exception as e:
            result.error = e
            result.success = False
            logger.error(f"Hook {hook.hook_id} failed: {e}")

        finally:
            result.execution_time = time.time() - start_time

        return result

    def enable_hook(self, hook_id: str) -> bool:
        """
        Enable a hook.

        Args:
            hook_id: Hook ID

        Returns:
            True if enabled, False if not found
        """
        for event, timings in self.hooks.items():
            for timing, hooks in timings.items():
                for hook in hooks:
                    if hook.hook_id == hook_id:
                        hook.enabled = True
                        logger.info(f"Enabled hook: {hook_id}")
                        return True
        return False

    def disable_hook(self, hook_id: str) -> bool:
        """
        Disable a hook.

        Args:
            hook_id: Hook ID

        Returns:
            True if disabled, False if not found
        """
        for event, timings in self.hooks.items():
            for timing, hooks in timings.items():
                for hook in hooks:
                    if hook.hook_id == hook_id:
                        hook.enabled = False
                        logger.info(f"Disabled hook: {hook_id}")
                        return True
        return False

    def list_hooks(
        self,
        event: Optional[str] = None,
        timing: Optional[HookTiming] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered hooks.

        Args:
            event: Filter by event name (optional)
            timing: Filter by timing (optional)

        Returns:
            List of hook information dictionaries
        """
        hooks_list = []

        events_to_check = [event] if event else self.hooks.keys()

        for evt in events_to_check:
            if evt not in self.hooks:
                continue

            timings_to_check = [timing] if timing else HookTiming

            for tm in timings_to_check:
                for hook in self.hooks[evt].get(tm, []):
                    hooks_list.append({
                        "hook_id": hook.hook_id,
                        "event": hook.event,
                        "timing": hook.timing.value,
                        "priority": hook.priority,
                        "enabled": hook.enabled,
                        "metadata": hook.metadata
                    })

        return hooks_list

    def get_hook_info(self, hook_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific hook.

        Args:
            hook_id: Hook ID

        Returns:
            Hook information or None if not found
        """
        for event, timings in self.hooks.items():
            for timing, hooks in timings.items():
                for hook in hooks:
                    if hook.hook_id == hook_id:
                        return {
                            "hook_id": hook.hook_id,
                            "event": hook.event,
                            "timing": hook.timing.value,
                            "priority": hook.priority,
                            "enabled": hook.enabled,
                            "metadata": hook.metadata
                        }
        return None

    def clear_event_hooks(self, event: str) -> int:
        """
        Clear all hooks for an event.

        Args:
            event: Event name

        Returns:
            Number of hooks cleared
        """
        if event not in self.hooks:
            return 0

        count = sum(len(hooks) for hooks in self.hooks[event].values())
        del self.hooks[event]

        logger.info(f"Cleared {count} hooks for event '{event}'")
        return count

    def clear_all_hooks(self) -> int:
        """
        Clear all registered hooks.

        Returns:
            Number of hooks cleared
        """
        count = sum(
            len(hooks)
            for event_hooks in self.hooks.values()
            for hooks in event_hooks.values()
        )

        self.hooks.clear()
        logger.info(f"Cleared all {count} hooks")
        return count

    def get_history(
        self,
        event: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution history.

        Args:
            event: Filter by event name (optional)
            limit: Limit number of results (optional)

        Returns:
            List of execution results
        """
        history = self.execution_history

        if event:
            history = [h for h in history if h.event == event]

        if limit:
            history = history[-limit:]

        return [h.to_dict() for h in history]

    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        logger.info("Cleared hook execution history")

    def _trim_history(self) -> None:
        """Trim history to limit."""
        if len(self.execution_history) > self.history_limit:
            self.execution_history = self.execution_history[-self.history_limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about hooks and executions.

        Returns:
            Dictionary with statistics
        """
        total_hooks = sum(
            len(hooks)
            for event_hooks in self.hooks.values()
            for hooks in event_hooks.values()
        )

        enabled_hooks = sum(
            1
            for event_hooks in self.hooks.values()
            for hooks in event_hooks.values()
            for hook in hooks
            if hook.enabled
        )

        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h.success)
        failed_executions = sum(1 for h in self.execution_history if not h.success)

        return {
            "total_hooks": total_hooks,
            "enabled_hooks": enabled_hooks,
            "disabled_hooks": total_hooks - enabled_hooks,
            "total_events": len(self.hooks),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0
        }
