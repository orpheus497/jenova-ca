##Script function and purpose: Concern Manager for The JENOVA Cognitive Architecture
"""
Concern Manager

Manages the lifecycle of concerns (topic categories) for organizing insights.
Concerns provide a hierarchical organization system for insights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from jenova.exceptions import ConcernError, LLMGenerationError
from jenova.insights.types import Concern
from jenova.llm.types import GenerationParams
from jenova.utils.migrations import load_json_with_migration, save_json_atomic

##Step purpose: Get module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for LLM operations
@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM dependency."""

    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        params: GenerationParams | None = None,
    ) -> str:
        """Generate text completion."""
        ...


##Class purpose: Manages concern topics for grouping and organizing insights
class ConcernManager:
    """
    Manages the lifecycle of concerns (topic categories).

    Concerns are used to organize insights into topical groups.
    The LLM is used to automatically categorize insights into
    existing concerns or create new ones when needed.
    """

    ##Method purpose: Initialize concern manager with storage path and LLM
    def __init__(
        self,
        insights_root: Path,
        llm: LLMProtocol,
    ) -> None:
        """
        Initialize concern manager.

        Args:
            insights_root: Root directory for insights storage
            llm: LLM interface for topic classification
        """
        ##Step purpose: Store configuration
        self._insights_root = insights_root
        self._concerns_file = insights_root / "concerns.json"
        self._llm = llm

        ##Action purpose: Load concerns from persistent storage
        self._concerns: dict[str, Concern] = self._load()

        logger.info(
            "concern_manager_initialized",
            storage_path=str(insights_root),
            concern_count=len(self._concerns),
        )

    ##Method purpose: Load concerns from persistent storage
    def _load(self) -> dict[str, Concern]:
        """Load concerns from disk with migration support."""

        ##Step purpose: Define default factory for empty concerns
        def default_factory() -> dict[str, object]:
            return {}

        ##Action purpose: Load with migration support
        data = load_json_with_migration(
            self._concerns_file,
            default_factory=default_factory,
        )

        ##Step purpose: Convert dict to Concern objects
        concerns: dict[str, Concern] = {}
        ##Loop purpose: Reconstruct Concern objects from data
        for name, concern_data in data.items():
            ##Condition purpose: Skip schema version key
            if name == "schema_version":
                continue
            ##Condition purpose: Skip non-dict values
            if not isinstance(concern_data, dict):
                continue
            concerns[name] = Concern.from_dict(name, concern_data)

        return concerns

    ##Method purpose: Save concerns to persistent storage
    def _save(self) -> None:
        """Save concerns to disk atomically."""
        data: dict[str, object] = {
            name: concern.to_dict() for name, concern in self._concerns.items()
        }
        data["schema_version"] = 1
        save_json_atomic(self._concerns_file, data)

    ##Method purpose: Get all concern names
    def get_all_concerns(self) -> list[str]:
        """
        Get all existing concern topic names.

        Returns:
            List of concern names
        """
        return list(self._concerns.keys())

    ##Method purpose: Get a specific concern by name
    def get_concern(self, name: str) -> Concern | None:
        """
        Get a specific concern by name.

        Args:
            name: Concern name to look up

        Returns:
            Concern if found, None otherwise
        """
        return self._concerns.get(name)

    ##Method purpose: Find matching concern or create new one for insight
    def find_or_create_concern(
        self,
        insight_content: str,
        existing_topics: list[str] | None = None,
    ) -> str:
        """
        Find the most relevant concern for an insight, or create a new one.

        Uses LLM to classify the insight content into an existing topic
        or generate a new topic name if no good fit exists.

        Args:
            insight_content: The insight content to classify
            existing_topics: List of existing topics (uses all if None)

        Returns:
            Topic name (existing or newly created)
        """
        topics = existing_topics if existing_topics is not None else self.get_all_concerns()

        ##Condition purpose: Create new concern if no existing topics
        if not topics:
            return self._create_new_concern(insight_content)

        ##Step purpose: Ask LLM to classify insight
        topics_formatted = "\n- ".join(topics)
        prompt_text = f'''Analyze the following insight and determine if it belongs to any of the existing topics. Respond with the most relevant topic name from the list if a good fit is found. If no existing topic is a good fit, respond with "new".

Existing Topics:
- {topics_formatted}

Insight: "{insight_content}"

Relevant Topic:'''

        ##Error purpose: Handle LLM failure with fallback
        try:
            params = GenerationParams(
                max_tokens=50,
                temperature=0.2,
            )
            chosen_topic = self._llm.generate_text(
                prompt_text,
                system_prompt="You are a topic classifier. Respond only with the topic name or 'new'.",
                params=params,
            ).strip()

            ##Condition purpose: Check if LLM chose existing topic
            if chosen_topic.lower() != "new" and chosen_topic in topics:
                logger.debug(
                    "insight_classified",
                    topic=chosen_topic,
                    content_preview=insight_content[:30],
                )
                return chosen_topic
            else:
                return self._create_new_concern(insight_content)

        except LLMGenerationError as e:
            logger.warning(
                "concern_classification_failed",
                error=str(e),
                fallback="creating_new",
            )
            ##Step purpose: Fallback to creating new concern
            return self._create_new_concern(insight_content)

    ##Method purpose: Create a new concern topic based on insight content
    def _create_new_concern(self, insight_content: str) -> str:
        """
        Create a new concern based on insight content.

        Uses LLM to generate an appropriate short topic name.

        Args:
            insight_content: Content to base topic name on

        Returns:
            The new topic name
        """
        prompt_text = f'''Create a short, one or two-word topic name for the following insight. Use underscores instead of spaces.

Insight: "{insight_content}"

Topic:'''

        ##Error purpose: Handle LLM failure with fallback
        try:
            params = GenerationParams(
                max_tokens=20,
                temperature=0.3,
            )
            new_topic = (
                self._llm.generate_text(
                    prompt_text,
                    system_prompt="You are a topic generator. Respond only with a short topic name using underscores.",
                    params=params,
                )
                .strip()
                .replace(" ", "_")
            )

            ##Step purpose: Sanitize topic name
            new_topic = "".join(c for c in new_topic if c.isalnum() or c == "_").lower()

            ##Condition purpose: Ensure we have a valid topic name
            if not new_topic:
                new_topic = "general"

        except LLMGenerationError as e:
            logger.warning(
                "topic_generation_failed",
                error=str(e),
                fallback="general",
            )
            new_topic = "general"

        ##Condition purpose: Add concern if it doesn't exist
        if new_topic not in self._concerns:
            self._concerns[new_topic] = Concern(
                name=new_topic,
                description=insight_content[:200],
                related_concerns=[],
            )
            self._save()

            logger.info(
                "concern_created",
                topic=new_topic,
            )

        return new_topic

    ##Method purpose: Add a concern manually
    def add_concern(
        self,
        name: str,
        description: str,
        related_concerns: list[str] | None = None,
    ) -> Concern:
        """
        Add a concern manually.

        Args:
            name: Concern name
            description: Description of the concern
            related_concerns: List of related concern names

        Returns:
            The created Concern
        """
        ##Step purpose: Sanitize name
        clean_name = name.replace(" ", "_").lower()

        concern = Concern(
            name=clean_name,
            description=description,
            related_concerns=related_concerns or [],
        )

        self._concerns[clean_name] = concern
        self._save()

        logger.info(
            "concern_added",
            name=clean_name,
        )

        return concern

    ##Method purpose: Link two concerns as related
    def link_concerns(self, concern_a: str, concern_b: str) -> None:
        """
        Link two concerns as related.

        Args:
            concern_a: First concern name
            concern_b: Second concern name

        Raises:
            ConcernError: If either concern doesn't exist
        """
        ##Condition purpose: Validate both concerns exist
        if concern_a not in self._concerns:
            raise ConcernError(f"Concern not found: {concern_a}")
        if concern_b not in self._concerns:
            raise ConcernError(f"Concern not found: {concern_b}")

        ##Step purpose: Add bidirectional relationship
        if concern_b not in self._concerns[concern_a].related_concerns:
            self._concerns[concern_a].related_concerns.append(concern_b)
        if concern_a not in self._concerns[concern_b].related_concerns:
            self._concerns[concern_b].related_concerns.append(concern_a)

        self._save()

        logger.info(
            "concerns_linked",
            concern_a=concern_a,
            concern_b=concern_b,
        )

    ##Method purpose: Factory method for production use
    @classmethod
    def create(
        cls,
        insights_root: Path,
        llm: LLMProtocol,
    ) -> ConcernManager:
        """
        Factory method to create ConcernManager.

        Args:
            insights_root: Root directory for insights storage
            llm: LLM interface for topic classification

        Returns:
            Configured ConcernManager instance
        """
        ##Step purpose: Ensure storage directory exists
        insights_root.mkdir(parents=True, exist_ok=True)

        return cls(
            insights_root=insights_root,
            llm=llm,
        )
