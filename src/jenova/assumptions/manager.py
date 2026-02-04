##Script function and purpose: Assumption Manager for The JENOVA Cognitive Architecture
"""
Assumption Manager

Manages the lifecycle of user assumptions including creation, verification,
and resolution. Assumptions are linked to the cognitive graph and can be
converted to insights when verified as true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

from jenova.assumptions.types import (
    Assumption,
    AssumptionStatus,
    AssumptionStore,
    CortexId,
)
from jenova.exceptions import (
    AssumptionDuplicateError,
    AssumptionError,
    AssumptionNotFoundError,
    LLMGenerationError,
)
from jenova.graph.types import Node
from jenova.llm.types import GenerationParams
from jenova.utils.migrations import load_json_with_migration, save_json_atomic

##Step purpose: Get module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for cognitive graph operations
@runtime_checkable
class GraphProtocol(Protocol):
    """Protocol for cognitive graph dependency."""

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        ...

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        ...


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


##Class purpose: Manages lifecycle of assumptions about users
class AssumptionManager:
    """
    Manages the lifecycle of assumptions about the user.

    Assumptions are hypotheses formed during conversation that can be
    verified through explicit user confirmation. Once verified, true
    assumptions can be converted to permanent insights.
    """

    ##Method purpose: Initialize assumption manager with dependencies
    def __init__(
        self,
        storage_path: Path,
        graph: GraphProtocol,
        llm: LLMProtocol,
    ) -> None:
        """
        Initialize assumption manager.

        Args:
            storage_path: Directory for assumption data
            graph: Cognitive graph for node linking
            llm: LLM interface for verification questions
        """
        ##Step purpose: Store dependencies
        self._storage_path = storage_path
        self._assumptions_file = storage_path / "assumptions.json"
        self._graph = graph
        self._llm = llm

        ##Action purpose: Load assumptions from persistent storage
        self._store = self._load()

        logger.info(
            "assumption_manager_initialized",
            storage_path=str(storage_path),
            total_assumptions=len(self._store.all_assumptions()),
        )

    ##Method purpose: Load assumptions from persistent storage
    def _load(self) -> AssumptionStore:
        """Load assumptions from disk with migration support."""

        ##Step purpose: Define default factory for empty store
        def default_factory() -> dict[str, object]:
            return {
                "unverified": [],
                "true": [],
                "false": [],
            }

        ##Action purpose: Load with migration support
        data = load_json_with_migration(
            self._assumptions_file,
            default_factory=default_factory,
        )

        return AssumptionStore.from_dict(data)

    ##Method purpose: Save assumptions to persistent storage
    def _save(self) -> None:
        """Save assumptions to disk atomically."""
        data = self._store.to_dict()
        data["schema_version"] = 1
        save_json_atomic(self._assumptions_file, data)

    ##Method purpose: Add a new assumption, rejecting duplicates
    def add_assumption(
        self,
        content: str,
        username: str,
        linked_to: list[CortexId] | None = None,
    ) -> CortexId:
        """
        Add a new assumption.

        Args:
            content: The assumption statement
            username: User this assumption relates to
            linked_to: Optional list of existing cortex IDs to link

        Returns:
            The cortex ID of the new assumption node

        Raises:
            AssumptionDuplicateError: If assumption already exists
        """
        ##Step purpose: Check for duplicates
        existing = self._store.find_by_content(content, username)

        ##Condition purpose: Reject duplicate assumptions
        if existing is not None:
            assumption, status = existing
            logger.warning(
                "assumption_duplicate",
                content=content[:50],
                existing_status=status.value,
            )
            raise AssumptionDuplicateError(content, status.value)

        ##Step purpose: Create cortex node for assumption
        node = Node.create(
            label=content[:50],
            content=content,
            node_type="assumption",
            metadata={"username": username},
        )
        self._graph.add_node(node)

        ##Step purpose: Create assumption record
        assumption = Assumption(
            content=content,
            username=username,
            status=AssumptionStatus.UNVERIFIED,
            cortex_id=node.id,
        )

        ##Action purpose: Add to store and persist
        self._store.unverified.append(assumption)
        self._save()

        logger.info(
            "assumption_added",
            cortex_id=node.id,
            username=username,
            content_preview=content[:50],
        )

        return node.id

    ##Method purpose: Get all assumptions as store
    def get_all_assumptions(self) -> AssumptionStore:
        """
        Get all assumptions.

        Returns:
            AssumptionStore containing all assumptions by status
        """
        return self._store

    ##Method purpose: Get an unverified assumption and generate verification question
    def get_assumption_to_verify(
        self,
        username: str,
    ) -> tuple[Assumption, str] | None:
        """
        Get an unverified assumption and generate a verification question.

        Args:
            username: Username to filter assumptions for

        Returns:
            Tuple of (assumption, question) if found, None otherwise
        """
        ##Step purpose: Find first unverified assumption for user
        user_unverified = [a for a in self._store.unverified if a.username == username]

        ##Condition purpose: Return None if no unverified assumptions
        if not user_unverified:
            return None

        assumption = user_unverified[0]

        ##Step purpose: Generate verification question using LLM
        prompt_text = (
            f"You have an unverified assumption about the user '{username}'. "
            "Ask them a clear, direct question to confirm or deny this assumption. "
            "The user's response will determine if the assumption is true or false.\n\n"
            f"Assumption: \"{assumption.content}\"\n\n"
            "Your question to the user:"
        )

        ##Error purpose: Handle LLM generation failure
        try:
            params = GenerationParams(
                max_tokens=256,
                temperature=0.3,
            )
            question = self._llm.generate_text(
                prompt_text,
                system_prompt="You are a helpful assistant asking clarifying questions.",
                params=params,
            )
            return (assumption, question.strip())
        except LLMGenerationError as e:
            ##Fix: Re-raise LLM errors with context - critical errors should
            ##     propagate, not be silently swallowed
            logger.error(
                "assumption_verification_question_failed",
                error=str(e),
                assumption_content=assumption.content[:50],
            )
            raise AssumptionError(f"Failed to generate verification question: {e}") from e

    ##Method purpose: Resolve an assumption based on user response
    def resolve_assumption(
        self,
        assumption: Assumption,
        user_response: str,
        username: str,
    ) -> AssumptionStatus:
        """
        Resolve an assumption based on user response.

        Uses LLM to analyze the user's response and determine if the
        assumption is confirmed or denied.

        Args:
            assumption: The assumption to resolve
            user_response: User's response to verification question
            username: Username for validation

        Returns:
            The resolved status (TRUE or FALSE)

        Raises:
            AssumptionNotFoundError: If assumption not in unverified list
        """
        ##Step purpose: Verify assumption exists in unverified list
        found = False
        found_index = -1
        ##Loop purpose: Find the assumption in unverified list
        for idx, a in enumerate(self._store.unverified):
            if a.content == assumption.content and a.username == username:
                found = True
                found_index = idx
                break

        ##Condition purpose: Raise if not found
        if not found:
            raise AssumptionNotFoundError(assumption.content)

        ##Step purpose: Use LLM to analyze response
        prompt_text = (
            "Analyze the following user response to the assumption "
            f"\"{assumption.content}\" to determine if it confirms or "
            "denies the assumption. Respond with exactly \"true\" or \"false\".\n\n"
            f"User response: \"{user_response}\"\n\n"
            "Result:"
        )

        ##Error purpose: Handle LLM errors gracefully
        try:
            params = GenerationParams(
                max_tokens=10,
                temperature=0.0,
            )
            result = self._llm.generate_text(
                prompt_text,
                system_prompt=(
                    "You are a classifier determining whether a response "
                    "confirms or denies an assumption. Respond only with "
                    "'true' or 'false'."
                ),
                params=params,
            ).strip().lower()
        except LLMGenerationError as e:
            logger.error(
                "assumption_resolution_failed",
                error=str(e),
            )
            ##Step purpose: Default to false on error
            result = "false"

        ##Step purpose: Determine final status
        is_true = result == "true"
        new_status = AssumptionStatus.TRUE if is_true else AssumptionStatus.FALSE

        ##Step purpose: Update assumption with new status
        updated = assumption.with_updates(status=new_status)

        ##Step purpose: Move to appropriate list
        self._store.unverified.pop(found_index)

        ##Condition purpose: Add to correct verified list and create insight if true
        if is_true:
            self._store.verified_true.append(updated)

            ##Step purpose: Create insight node linked to assumption (legacy behavior)
            insight_node = Node.create(
                label=f"Insight: {assumption.content[:40]}",
                content=assumption.content,
                node_type="insight",
                metadata={
                    "username": username,
                    "source": "assumption_verification",
                    "linked_assumption_id": updated.cortex_id or "",
                },
            )
            self._graph.add_node(insight_node)

            logger.info(
                "assumption_confirmed_and_converted_to_insight",
                assumption_content=assumption.content[:50],
                insight_node_id=insight_node.id,
            )
        else:
            self._store.verified_false.append(updated)
            logger.info(
                "assumption_denied",
                content=assumption.content[:50],
            )

        self._save()

        return new_status

    ##Method purpose: Update an existing assumption's content
    def update_assumption(
        self,
        old_content: str,
        new_content: str,
        username: str,
    ) -> Assumption:
        """
        Update an existing assumption's content.

        Args:
            old_content: Current content to find
            new_content: New content to set
            username: Username for validation

        Returns:
            Updated assumption

        Raises:
            AssumptionNotFoundError: If assumption not found
        """
        ##Step purpose: Search all lists for the assumption
        lists = [
            ("unverified", self._store.unverified),
            ("true", self._store.verified_true),
            ("false", self._store.verified_false),
        ]

        ##Loop purpose: Find and update assumption
        for list_name, assumption_list in lists:
            for idx, assumption in enumerate(assumption_list):
                if assumption.content == old_content and assumption.username == username:
                    ##Step purpose: Create updated assumption
                    updated = assumption.with_updates(content=new_content)
                    assumption_list[idx] = updated
                    self._save()

                    logger.info(
                        "assumption_updated",
                        old_content=old_content[:30],
                        new_content=new_content[:30],
                        list_name=list_name,
                    )

                    return updated

        raise AssumptionNotFoundError(old_content)

    ##Method purpose: Get count of unverified assumptions for user
    def unverified_count(self, username: str) -> int:
        """Get count of unverified assumptions for a user."""
        return len([a for a in self._store.unverified if a.username == username])

    ##Method purpose: Get all pending verifications for a user
    def get_pending_verifications(self, username: str) -> list[Assumption]:
        """
        Get all assumptions needing verification for a user.

        Args:
            username: Username to filter by

        Returns:
            List of unverified Assumptions for the user
        """
        return [a for a in self._store.unverified if a.username == username]

    ##Method purpose: Get all assumptions for a user regardless of status
    def get_assumptions_for_user(self, username: str) -> list[Assumption]:
        """
        Get all assumptions for a user across all statuses.

        Args:
            username: Username to filter by

        Returns:
            List of all Assumptions for the user
        """
        return [a for a in self._store.all_assumptions() if a.username == username]

    ##Method purpose: Factory method for production use
    @classmethod
    def create(
        cls,
        storage_path: Path,
        graph: GraphProtocol,
        llm: LLMProtocol,
    ) -> AssumptionManager:
        """
        Factory method to create AssumptionManager.

        Args:
            storage_path: Directory for assumption data
            graph: Cognitive graph for node linking
            llm: LLM interface for verification questions

        Returns:
            Configured AssumptionManager instance
        """
        ##Step purpose: Ensure storage directory exists
        storage_path.mkdir(parents=True, exist_ok=True)

        return cls(
            storage_path=storage_path,
            graph=graph,
            llm=llm,
        )
