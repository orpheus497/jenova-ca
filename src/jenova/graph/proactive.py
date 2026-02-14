##Script function and purpose: Proactive engine for autonomous cognitive suggestions
"""
Proactive Engine - Autonomous suggestion generation based on cognitive state.

This module provides the infrastructure for JENOVA to proactively suggest
topics, questions, and areas for exploration based on the cognitive graph
state and conversation patterns.

Reference: .devdocs/resources/src/jenova/cortex/proactive_engine.py
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Protocol

import structlog

from jenova.assumptions.types import Assumption
from jenova.exceptions import (
    AssumptionError,
    GraphError,
    LLMGenerationError,
    NodeNotFoundError,
    ProactiveError,
)
from jenova.graph.types import Node

##Class purpose: Define logger for proactive engine operations
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for assumption manager access
class AssumptionManagerProtocol(Protocol):
    """Protocol for AssumptionManager access."""

    ##Method purpose: Get assumption to verify
    def get_assumption_to_verify(self, username: str) -> tuple[Assumption, str] | None:
        """Get an unverified assumption and generate a verification question."""
        ...


##Class purpose: Categories of proactive suggestions
class SuggestionCategory(Enum):
    """Categories of proactive suggestions."""

    EXPLORE = auto()  # Explore new topics
    VERIFY = auto()  # Verify assumptions
    DEVELOP = auto()  # Develop existing insights
    CONNECT = auto()  # Connect related concepts
    REFLECT = auto()  # Reflect on patterns


##Class purpose: A proactive suggestion from the engine
@dataclass(frozen=True)
class Suggestion:
    """A proactive suggestion for the user.

    Attributes:
        category: The category of suggestion
        content: The suggestion text
        topic: Related topic if applicable
        priority: Priority score (0-1, higher = more urgent)
        node_ids: Related graph node IDs
        timestamp: When the suggestion was generated
    """

    category: SuggestionCategory
    content: str
    topic: str | None = None
    priority: float = 0.5
    node_ids: tuple[str, ...] = field(default_factory=tuple)
    timestamp: datetime = field(default_factory=datetime.now)


##Class purpose: Configuration for the proactive engine
@dataclass
class ProactiveConfig:
    """Configuration for the ProactiveEngine.

    Attributes:
        cooldown_minutes: Minimum minutes between suggestions of same category
        max_suggestions_per_session: Maximum suggestions to generate per session
        priority_threshold: Minimum priority to show suggestion (0-1)
        enable_explore: Enable exploration suggestions
        enable_verify: Enable verification suggestions
        enable_develop: Enable development suggestions
        enable_connect: Enable connection suggestions
        enable_reflect: Enable reflection suggestions
        rotation_enabled: Enable category rotation for variety
    """

    cooldown_minutes: int = 15
    max_suggestions_per_session: int = 10
    priority_threshold: float = 0.3
    enable_explore: bool = True
    enable_verify: bool = True
    enable_develop: bool = True
    enable_connect: bool = True
    enable_reflect: bool = True
    rotation_enabled: bool = True


##Class purpose: Protocol for graph access
class GraphProtocol(Protocol):
    """Protocol for CognitiveGraph access."""

    ##Method purpose: Get nodes for a user
    def get_nodes_by_user(self, username: str) -> list[Node]:
        """Get all nodes for a user."""
        ...

    ##Method purpose: Search nodes by content
    def search(
        self,
        query: str,
        max_results: int = 10,
        node_types: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Search nodes by content."""
        ...

    ##Method purpose: Get node by ID
    def get_node(self, node_id: str) -> Node:
        """Get a node by ID."""
        ...


##Class purpose: Protocol for LLM access
class LLMProtocol(Protocol):
    """Protocol for LLM access."""

    ##Method purpose: Generate text from prompt
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


##Class purpose: Track engagement with suggestions
@dataclass
class EngagementTracker:
    """Track user engagement with suggestions.

    Attributes:
        suggestions_shown: Total suggestions shown
        suggestions_accepted: Suggestions user engaged with
        category_shown: Count by category
        category_accepted: Accepted count by category
    """

    suggestions_shown: int = 0
    suggestions_accepted: int = 0
    category_shown: dict[SuggestionCategory, int] = field(default_factory=dict)
    category_accepted: dict[SuggestionCategory, int] = field(default_factory=dict)

    ##Method purpose: Record a suggestion was shown
    def record_shown(self, category: SuggestionCategory) -> None:
        """Record that a suggestion was shown."""
        self.suggestions_shown += 1
        self.category_shown[category] = self.category_shown.get(category, 0) + 1

    ##Method purpose: Record a suggestion was accepted
    def record_accepted(self, category: SuggestionCategory) -> None:
        """Record that a suggestion was accepted."""
        self.suggestions_accepted += 1
        self.category_accepted[category] = self.category_accepted.get(category, 0) + 1

    ##Method purpose: Get acceptance rate for a category
    def get_acceptance_rate(self, category: SuggestionCategory) -> float:
        """Get acceptance rate for a category.

        Args:
            category: The suggestion category to check

        Returns:
            Acceptance rate as float (0.0-1.0), or 0.5 if no suggestions shown yet
        """
        shown = self.category_shown.get(category, 0)
        ##Condition purpose: Avoid division by zero
        if shown == 0:
            return 0.5  # Default neutral rate
        accepted = self.category_accepted.get(category, 0)
        return accepted / shown


##Class purpose: Main proactive engine for autonomous suggestions
class ProactiveEngine:
    """Engine for generating proactive cognitive suggestions.

    The proactive engine analyzes the cognitive graph and conversation
    patterns to generate helpful suggestions for the user.

    Example:
        >>> engine = ProactiveEngine(config, graph, llm)
        >>> suggestion = engine.get_suggestion(username="user1")
        >>> if suggestion:
        ...     print(f"Suggestion: {suggestion.content}")
    """

    ##Method purpose: Initialize the proactive engine
    def __init__(
        self,
        config: ProactiveConfig,
        graph: GraphProtocol | None = None,
        llm: LLMProtocol | None = None,
        assumption_manager: AssumptionManagerProtocol | None = None,
    ) -> None:
        """Initialize the proactive engine.

        Args:
            config: Engine configuration
            graph: Optional graph access (can be set later)
            llm: Optional LLM access (can be set later)
            assumption_manager: Optional assumption manager (can be set later)
        """
        ##Step purpose: Store configuration and dependencies
        self._config = config
        self._graph = graph
        self._llm = llm
        self._assumption_manager = assumption_manager

        ##Step purpose: Initialize tracking state
        self._last_suggestion_time: dict[SuggestionCategory, datetime] = {}
        self._session_suggestion_count = 0
        self._engagement = EngagementTracker()
        self._last_category: SuggestionCategory | None = None

        logger.info(
            "proactive_engine_initialized",
            enabled_categories=self._get_enabled_categories(),
        )

    ##Method purpose: Get list of enabled categories
    def _get_enabled_categories(self) -> list[str]:
        """Get list of enabled category names.

        Returns:
            List of enabled category names as strings
        """
        enabled = []
        if self._config.enable_explore:
            enabled.append("EXPLORE")
        if self._config.enable_verify:
            enabled.append("VERIFY")
        if self._config.enable_develop:
            enabled.append("DEVELOP")
        if self._config.enable_connect:
            enabled.append("CONNECT")
        if self._config.enable_reflect:
            enabled.append("REFLECT")
        return enabled

    ##Method purpose: Set the graph dependency
    def set_graph(self, graph: GraphProtocol) -> None:
        """Set the graph dependency."""
        self._graph = graph

    ##Method purpose: Set the LLM dependency
    def set_llm(self, llm: LLMProtocol) -> None:
        """Set the LLM dependency."""
        self._llm = llm

    ##Method purpose: Set the assumption manager dependency
    def set_assumption_manager(self, manager: AssumptionManagerProtocol) -> None:
        """Set the assumption manager dependency."""
        self._assumption_manager = manager

    ##Method purpose: Check if a category is on cooldown
    def _is_on_cooldown(self, category: SuggestionCategory) -> bool:
        """Check if a category is on cooldown.

        Args:
            category: The suggestion category to check

        Returns:
            True if category is on cooldown, False otherwise
        """
        last_time = self._last_suggestion_time.get(category)
        ##Condition purpose: No cooldown if never suggested
        if last_time is None:
            return False

        cooldown = timedelta(minutes=self._config.cooldown_minutes)
        return datetime.now() - last_time < cooldown

    ##Method purpose: Get the next category to suggest
    def _get_next_category(self) -> SuggestionCategory | None:
        """Get the next category to suggest based on rotation and engagement.

        Returns:
            Next suggestion category to use, or None if no categories available
        """
        ##Step purpose: Build list of available categories
        available: list[SuggestionCategory] = []

        if self._config.enable_explore and not self._is_on_cooldown(SuggestionCategory.EXPLORE):
            available.append(SuggestionCategory.EXPLORE)
        if self._config.enable_verify and not self._is_on_cooldown(SuggestionCategory.VERIFY):
            available.append(SuggestionCategory.VERIFY)
        if self._config.enable_develop and not self._is_on_cooldown(SuggestionCategory.DEVELOP):
            available.append(SuggestionCategory.DEVELOP)
        if self._config.enable_connect and not self._is_on_cooldown(SuggestionCategory.CONNECT):
            available.append(SuggestionCategory.CONNECT)
        if self._config.enable_reflect and not self._is_on_cooldown(SuggestionCategory.REFLECT):
            available.append(SuggestionCategory.REFLECT)

        ##Condition purpose: No available categories
        if not available:
            return None

        ##Condition purpose: Apply rotation to avoid repeating
        if self._config.rotation_enabled and self._last_category in available:
            available.remove(self._last_category)
            ##Condition purpose: If only one category was available, allow repeat
            if not available:
                available = [self._last_category]

        ##Step purpose: Weight by engagement rate
        weights = [self._engagement.get_acceptance_rate(cat) for cat in available]

        ##Condition purpose: Use weighted random selection
        total_weight = sum(weights)
        if total_weight > 0:
            r = random.random() * total_weight
            cumulative = 0.0
            for cat, weight in zip(available, weights, strict=False):
                cumulative += weight
                if r <= cumulative:
                    return cat

        ##Step purpose: Fallback to random selection
        return random.choice(available)

    ##Method purpose: Generate a suggestion for a category
    def _generate_suggestion(
        self,
        category: SuggestionCategory,
        username: str,
    ) -> Suggestion | None:
        """Generate a suggestion for a category.

        Args:
            category: The category to generate for
            username: The user context

        Returns:
            A suggestion or None if generation failed
        """
        ##Condition purpose: Check if graph is available
        if self._graph is None:
            logger.warning("proactive_no_graph")
            return None

        ##Error purpose: Handle suggestion generation errors
        try:
            ##Condition purpose: Route to appropriate generator
            if category == SuggestionCategory.EXPLORE:
                return self._generate_explore_suggestion(username)
            elif category == SuggestionCategory.VERIFY:
                return self._generate_verify_suggestion(username)
            elif category == SuggestionCategory.DEVELOP:
                return self._generate_develop_suggestion(username)
            elif category == SuggestionCategory.CONNECT:
                return self._generate_connect_suggestion(username)
            elif category == SuggestionCategory.REFLECT:
                return self._generate_reflect_suggestion(username)
            else:
                ##Fix: Raise exception for unknown category - indicates programming error
                raise ProactiveError(f"Unknown suggestion category: {category}")

        except (GraphError, NodeNotFoundError, ValueError, AttributeError) as e:
            ##Fix: Catch specific exceptions and re-raise with context - prevents silent failures
            logger.error(
                "proactive_generation_failed",
                category=category.name,
                error=str(e),
            )
            raise ProactiveError(f"Failed to generate {category.name} suggestion: {e}") from e
        except Exception as e:
            ##Fix: Re-raise unexpected exceptions with context - prevents hiding critical errors
            logger.error(
                "unexpected_error_in_proactive_generation",
                category=category.name,
                error=str(e),
            )
            raise ProactiveError(f"Unexpected error generating suggestion: {e}") from e

    ##Method purpose: Generate an exploration suggestion
    def _generate_explore_suggestion(self, username: str) -> Suggestion | None:
        """Generate a suggestion to explore new topics."""
        ##Step purpose: Get user's nodes to find gaps
        nodes = self._graph.get_nodes_by_user(username) if self._graph else []

        ##Condition purpose: No nodes means suggest general exploration
        if not nodes:
            return Suggestion(
                category=SuggestionCategory.EXPLORE,
                content="What topics interest you most? Let's explore them together.",
                priority=0.7,
            )

        ##Step purpose: Find topics with few connections
        topic_counts: dict[str, int] = {}
        for node in nodes:
            node_type = node.node_type
            topic_counts[node_type] = topic_counts.get(node_type, 0) + 1

        ##Condition purpose: Suggest exploring less-developed areas
        if topic_counts:
            min_topic = min(topic_counts.keys(), key=lambda t: topic_counts[t])
            return Suggestion(
                category=SuggestionCategory.EXPLORE,
                content=f"We haven't explored '{min_topic}' much. Would you like to dive deeper?",
                topic=min_topic,
                priority=0.6,
            )

        return None

    ##Method purpose: Generate a verification suggestion
    def _generate_verify_suggestion(self, username: str) -> Suggestion | None:
        """Generate a suggestion to verify assumptions."""
        ##Step purpose: Check if assumption manager is available
        if self._assumption_manager:
            try:
                result = self._assumption_manager.get_assumption_to_verify(username)
                if result:
                    assumption, question = result
                    node_ids = (assumption.cortex_id,) if assumption.cortex_id else ()
                    return Suggestion(
                        category=SuggestionCategory.VERIFY,
                        content=question,
                        priority=0.8,
                        node_ids=node_ids,
                    )
            except (AssumptionError, LLMGenerationError) as e:
                logger.warning("assumption_manager_error", error=str(e), exc_type=type(e).__name__)
                # Fall through to generic suggestion

        # Fallback to a generic verification prompt if no specific assumption found
        # or manager not available
        return Suggestion(
            category=SuggestionCategory.VERIFY,
            content="I've made some assumptions about your preferences. Would you like to review them?",
            priority=0.5,
        )

    ##Method purpose: Generate a development suggestion
    def _generate_develop_suggestion(self, username: str) -> Suggestion | None:
        """Generate a suggestion to develop existing insights."""
        ##Step purpose: Find insights that could be developed
        nodes = self._graph.get_nodes_by_user(username) if self._graph else []

        ##Step purpose: Find nodes with potential for development
        insight_nodes = [n for n in nodes if n.node_type == "insight"]

        ##Condition purpose: Suggest developing an existing insight
        if insight_nodes:
            node = random.choice(insight_nodes)
            content = node.content[:100]
            return Suggestion(
                category=SuggestionCategory.DEVELOP,
                content=f'We could develop this insight further: "{content}..."',
                priority=0.5,
                node_ids=(node.id,),
            )

        return None

    ##Method purpose: Generate a connection suggestion
    def _generate_connect_suggestion(self, username: str) -> Suggestion | None:
        """Generate a suggestion to connect related concepts."""
        ##Step purpose: Find nodes that could be connected
        return Suggestion(
            category=SuggestionCategory.CONNECT,
            content="I notice some of your ideas might be related. Would you like me to help connect them?",
            priority=0.4,
        )

    ##Method purpose: Generate a reflection suggestion
    def _generate_reflect_suggestion(self, username: str) -> Suggestion | None:
        """Generate a suggestion to reflect on patterns."""
        return Suggestion(
            category=SuggestionCategory.REFLECT,
            content="Would you like me to reflect on patterns in our conversations?",
            priority=0.3,
        )

    ##Method purpose: Get a proactive suggestion for the user
    def get_suggestion(self, username: str) -> Suggestion | None:
        """Get a proactive suggestion for the user.

        Args:
            username: The user to generate suggestion for

        Returns:
            A suggestion or None if no suggestion available
        """
        ##Condition purpose: Check session limit
        if self._session_suggestion_count >= self._config.max_suggestions_per_session:
            logger.debug("proactive_session_limit_reached")
            return None

        ##Step purpose: Get next category
        category = self._get_next_category()
        if category is None:
            logger.debug("proactive_all_on_cooldown")
            return None

        ##Step purpose: Generate suggestion
        suggestion = self._generate_suggestion(category, username)

        ##Condition purpose: Update state if suggestion generated
        if suggestion and suggestion.priority >= self._config.priority_threshold:
            self._last_suggestion_time[category] = datetime.now()
            self._session_suggestion_count += 1
            self._last_category = category
            self._engagement.record_shown(category)

            logger.debug(
                "proactive_suggestion_generated",
                category=category.name,
                priority=suggestion.priority,
            )

            return suggestion

        return None

    ##Method purpose: Record that user accepted a suggestion
    def record_acceptance(self, category: SuggestionCategory) -> None:
        """Record that a user accepted/engaged with a suggestion.

        Args:
            category: The category of accepted suggestion
        """
        self._engagement.record_accepted(category)
        logger.debug(
            "proactive_suggestion_accepted",
            category=category.name,
            acceptance_rate=self._engagement.get_acceptance_rate(category),
        )

    ##Method purpose: Reset session state
    def reset_session(self) -> None:
        """Reset session-specific state."""
        self._session_suggestion_count = 0
        self._last_category = None
        logger.info("proactive_session_reset")

    ##Method purpose: Get engine status for diagnostics
    def get_status(self) -> dict[str, object]:
        """Get engine status for diagnostics."""
        return {
            "session_count": self._session_suggestion_count,
            "max_per_session": self._config.max_suggestions_per_session,
            "last_category": self._last_category.name if self._last_category else None,
            "cooldowns": {cat.name: self._is_on_cooldown(cat) for cat in SuggestionCategory},
            "engagement": {
                "total_shown": self._engagement.suggestions_shown,
                "total_accepted": self._engagement.suggestions_accepted,
            },
        }
