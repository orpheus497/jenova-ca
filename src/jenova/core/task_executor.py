##Script function and purpose: Task executor bridging CognitiveScheduler to engine subsystems
"""
Cognitive Task Executor

Implements TaskExecutorProtocol to dispatch scheduled cognitive tasks
to the appropriate engine subsystems (insight manager, assumption manager,
knowledge graph). Created as part of WIRING-001.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import structlog

from jenova.core.scheduler import TaskType
from jenova.exceptions import AssumptionDuplicateError
from jenova.utils.sanitization import sanitize_for_prompt

if TYPE_CHECKING:
    from jenova.assumptions.manager import AssumptionManager
    from jenova.core.knowledge import KnowledgeStore
    from jenova.insights.manager import InsightManager

##Class purpose: Define logger for task executor operations
logger = structlog.get_logger(__name__)

##Update: WIRING-001 (2026-02-13T11:26:36Z) — Bridge scheduler tasks to engine subsystems
##Note: LLMProtocol used here matches graph.LLMProtocol (generate_text(text, system_prompt))

##Note: Maximum recent history turns used for autonomous insight/assumption generation
_MAX_HISTORY_FOR_GENERATION = 5

##Note: Maximum characters of AI response to include in history summary
_MAX_RESPONSE_PREVIEW_LENGTH = 200


##Class purpose: Protocol for LLM operations used by task executor
class TaskExecutorLLMProtocol(Protocol):
    """Protocol for LLM text generation used by CognitiveTaskExecutor."""

    def generate_text(self, text: str, system_prompt: str = ...) -> str: ...

##Note: System prompts for autonomous cognitive generation
_INSIGHT_SYSTEM_PROMPT = (
    "You are a cognitive analyst. Based on the conversation history, "
    "generate a single concise insight about the user — something learned, "
    "a pattern noticed, or a preference observed. Return ONLY the insight text, "
    "no preamble."
)

_ASSUMPTION_SYSTEM_PROMPT = (
    "You are a cognitive analyst. Based on the conversation history, "
    "generate a single testable assumption about the user — something that "
    "could be confirmed or denied later. Return ONLY the assumption text, "
    "no preamble."
)


##Class purpose: Execute cognitive tasks dispatched by the scheduler
class CognitiveTaskExecutor:
    """Executes cognitive tasks dispatched by CognitiveScheduler.

    Bridges the scheduler's task dispatch to the actual engine subsystems:
    insight manager, assumption manager, and knowledge graph.

    Implements TaskExecutorProtocol from jenova.core.scheduler.
    """

    ##Method purpose: Initialize executor with engine subsystem references
    def __init__(
        self,
        insight_manager: InsightManager | None,
        assumption_manager: AssumptionManager | None,
        knowledge_store: KnowledgeStore,
        llm: TaskExecutorLLMProtocol,
        get_recent_history: Callable[[], list[tuple[str, str]]],
    ) -> None:
        """Initialize the task executor.

        Args:
            insight_manager: Insight manager for insight generation/storage
            assumption_manager: Assumption manager for assumption generation/storage
            knowledge_store: Knowledge store providing graph access
            llm: LLM interface satisfying TaskExecutorLLMProtocol
            get_recent_history: Callable returning recent history as list[tuple[str, str]]
        """
        self._insight_manager = insight_manager
        self._assumption_manager = assumption_manager
        self._knowledge_store = knowledge_store
        self._llm = llm
        self._get_recent_history = get_recent_history

    ##Method purpose: Dispatch a cognitive task to the appropriate handler
    def execute_task(self, task_type: TaskType, username: str) -> bool:
        """Execute a cognitive task.

        Args:
            task_type: The type of task to execute
            username: The user context for the task

        Returns:
            True if task completed successfully, False otherwise
        """
        dispatch = {
            TaskType.GENERATE_INSIGHT: self._generate_insight,
            TaskType.GENERATE_ASSUMPTION: self._generate_assumption,
            TaskType.VERIFY_ASSUMPTION: self._check_pending_verifications,
            TaskType.REFLECT: self._reflect,
            TaskType.PRUNE_GRAPH: self._prune_graph,
            TaskType.LINK_ORPHANS: self._link_orphans,
        }

        handler = dispatch.get(task_type)
        if handler is None:
            logger.warning("task_executor_unknown_type", task_type=task_type.name)
            return False

        return handler(username)

    ##Method purpose: Build a history summary for LLM generation prompts
    def _build_history_summary(self) -> str:
        """Build a summary of recent conversation history.

        Returns:
            Formatted string of recent conversation turns, or empty string
        """
        history = self._get_recent_history()
        if not history:
            return ""

        recent = history[-_MAX_HISTORY_FOR_GENERATION:]
        lines: list[str] = []
        for user_msg, ai_response in recent:
            ##Sec: Sanitize messages before injecting into generation prompts (PATCH-006)
            safe_user = sanitize_for_prompt(user_msg)
            safe_ai = sanitize_for_prompt(ai_response)

            lines.append(f"User: {safe_user}")
            preview = safe_ai[:_MAX_RESPONSE_PREVIEW_LENGTH]
            if len(safe_ai) > _MAX_RESPONSE_PREVIEW_LENGTH:
                preview += "..."
            lines.append(f"JENOVA: {preview}")
        return "\n".join(lines)

    ##Refactor: Extracted common logic for history-based generation (WIRING-001)
    def _generate_from_history(
        self,
        username: str,
        manager: object | None,
        system_prompt: str,
        item_name: str,
        save_method_name: str,
        log_success: str,
        log_skip: str,
        log_fail: str,
        min_length: int = 10,
    ) -> bool:
        """Generate and save an item from recent conversation history.

        Args:
            username: User context.
            manager: Manager instance (checked for None).
            system_prompt: System prompt for LLM.
            item_name: Name of item to generate (e.g. 'insight', 'assumption').
            save_method_name: Name of the method on manager to save the item.
            log_success: Logger event name for success.
            log_skip: Logger event name for skip.
            log_fail: Logger event name for failure.
            min_length: Minimum content length.

        Returns:
            True if generated and saved.
        """
        if manager is None:
            logger.debug(log_skip, reason=f"no_{item_name}_manager")
            return False

        ##Fix: Ensure sanitize_for_prompt errors are caught (BH-2026-02-14)
        try:
            history_summary = self._build_history_summary()
            if not history_summary:
                logger.debug(log_skip, reason="no_history")
                return False

            ##Step purpose: Ask LLM to generate item from conversation context
            prompt = f"Recent conversation:\n{history_summary}\n\nGenerate an {item_name}:"
            content = self._llm.generate_text(prompt, system_prompt=system_prompt)
            stripped = content.strip() if content else ""

            if not stripped or len(stripped) < min_length:
                logger.debug(log_skip, reason="empty_generation")
                return False

            ##Refactor: Ensure manager has the expected save method (D3-2026-02-14)
            if not hasattr(manager, save_method_name):
                logger.error(
                    "missing_save_method",
                    manager_type=type(manager).__name__,
                    method=save_method_name,
                )
                return False

            ##Step purpose: Save the generated item via the manager
            getattr(manager, save_method_name)(
                content=stripped,
                username=username,
            )

            logger.info(log_success, username=username)
            return True

        except AssumptionDuplicateError:
            # Duplicate assumptions are expected and not failures
            logger.debug(log_skip, reason="duplicate")
            return False
        except Exception as e:
            logger.error(log_fail, error=str(e), exc_info=True)
            return False

    ##Method purpose: Generate and save an insight from recent conversation
    def _generate_insight(self, username: str) -> bool:
        """Generate an insight from recent conversation history.

        Args:
            username: User context for insight generation

        Returns:
            True if insight was generated and saved
        """
        return self._generate_from_history(
            username=username,
            manager=self._insight_manager,
            system_prompt=_INSIGHT_SYSTEM_PROMPT,
            item_name="insight",
            save_method_name="save_insight",
            log_success="task_executor_insight_generated",
            log_skip="task_executor_skip_insight",
            log_fail="task_executor_insight_failed",
        )

    ##Method purpose: Generate and save an assumption from recent conversation
    def _generate_assumption(self, username: str) -> bool:
        """Generate an assumption from recent conversation history.

        Args:
            username: User context for assumption generation

        Returns:
            True if assumption was generated and saved
        """
        return self._generate_from_history(
            username=username,
            manager=self._assumption_manager,
            system_prompt=_ASSUMPTION_SYSTEM_PROMPT,
            item_name="assumption",
            save_method_name="add_assumption",
            log_success="task_executor_assumption_generated",
            log_skip="task_executor_assumption_skipped",
            log_fail="task_executor_assumption_failed",
        )

    ##Method purpose: Check and log pending assumption verifications
    def _check_pending_verifications(self, username: str) -> bool:
        """Check pending assumption verifications (non-interactive).

        Logs the count of pending verifications. Actual verification requires
        user interaction and is handled by the UI layer.

        Args:
            username: User context for verification check

        Returns:
            True if check completed, False if assumption_manager unavailable
        """
        if self._assumption_manager is None:
            return False

        try:
            count = self._assumption_manager.unverified_count(username)
            if count > 0:
                logger.info(
                    "task_executor_pending_verifications",
                    username=username,
                    count=count,
                )
            return True
        except Exception as e:
            ##Fix: Handle unexpected errors in verification check
            logger.error("task_executor_verification_check_failed", error=str(e), exc_info=True)
            return False

    ##Method purpose: Run a full reflection cycle on the knowledge graph
    def _reflect(self, username: str) -> bool:
        """Run a reflection cycle on the cognitive graph.

        Args:
            username: User context for reflection

        Returns:
            True if reflection completed successfully
        """
        try:
            graph = self._knowledge_store.graph
            result = graph.reflect(username, self._llm)

            logger.info(
                "task_executor_reflect_completed",
                username=username,
                orphans_linked=result.get("orphans_linked", 0),
                insights_generated=len(result.get("insights_generated", [])),
                nodes_pruned=result.get("nodes_pruned", 0),
            )
            return True

        except Exception as e:
            logger.error("task_executor_reflect_failed", error=str(e), exc_info=True)
            return False

    ##Method purpose: Prune stale nodes from the knowledge graph
    def _prune_graph(self, username: str) -> bool:
        """Prune stale nodes from the cognitive graph.

        Args:
            username: User context for scoped pruning

        Returns:
            True if pruning completed successfully
        """
        try:
            graph = self._knowledge_store.graph
            pruned = graph.prune_graph(username=username)

            logger.info(
                "task_executor_prune_completed",
                username=username,
                nodes_pruned=pruned,
            )
            return True

        except Exception as e:
            logger.error("task_executor_prune_failed", error=str(e), exc_info=True)
            return False

    ##Method purpose: Link orphan nodes in the knowledge graph
    def _link_orphans(self, username: str) -> bool:
        """Link orphan nodes to related nodes in the cognitive graph.

        Args:
            username: User context for orphan linking

        Returns:
            True if linking completed successfully
        """
        try:
            graph = self._knowledge_store.graph
            linked = graph.link_orphans(username, self._llm)

            logger.info(
                "task_executor_orphans_linked",
                username=username,
                links_created=linked,
            )
            return True

        except Exception as e:
            logger.error("task_executor_link_orphans_failed", error=str(e), exc_info=True)
            return False
