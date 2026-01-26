##Script function and purpose: Type definitions for the assumptions module
"""
Assumption Types

Type definitions for assumption tracking. Assumptions are hypotheses
that JENOVA forms about the user and world, which can be verified
or disproven through conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypeAlias

##Step purpose: Define type alias for cortex node ID
CortexId: TypeAlias = str


##Class purpose: Enum defining assumption verification states
class AssumptionStatus(Enum):
    """Status of an assumption in its lifecycle."""

    UNVERIFIED = "unverified"
    """Assumption has not been verified yet."""

    TRUE = "true"
    """Assumption was confirmed by user."""

    FALSE = "false"
    """Assumption was denied by user."""


##Class purpose: Immutable assumption record
@dataclass(frozen=True)
class Assumption:
    """An assumption about the user or world."""

    content: str
    """The assumption content/statement."""

    username: str
    """User this assumption relates to."""

    status: AssumptionStatus
    """Current verification status."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """ISO timestamp of creation."""

    cortex_id: CortexId | None = None
    """Linked node ID in cognitive graph."""

    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "username": self.username,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "cortex_id": self.cortex_id,
        }

    ##Method purpose: Create from dict during deserialization
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Assumption:
        """Create from dictionary."""
        return cls(
            content=str(data["content"]),
            username=str(data["username"]),
            status=AssumptionStatus(data.get("status", "unverified")),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
            cortex_id=data.get("cortex_id"),
        )

    ##Method purpose: Create new assumption with updated fields
    def with_updates(
        self,
        content: str | None = None,
        status: AssumptionStatus | None = None,
        cortex_id: CortexId | None = None,
    ) -> Assumption:
        """
        Create new Assumption with updated fields.

        Args:
            content: New content (None to keep existing)
            status: New status (None to keep existing)
            cortex_id: New cortex ID (None to keep existing)

        Returns:
            New Assumption instance with updates
        """
        return Assumption(
            content=content if content is not None else self.content,
            username=self.username,
            status=status if status is not None else self.status,
            timestamp=datetime.now().isoformat(),
            cortex_id=cortex_id if cortex_id is not None else self.cortex_id,
        )


##Class purpose: Container for all assumptions organized by status
@dataclass
class AssumptionStore:
    """
    Container for assumptions organized by status.

    Maintains separate lists for each status to enable efficient
    lookup by verification state.
    """

    unverified: list[Assumption] = field(default_factory=list)
    """Unverified assumptions awaiting confirmation."""

    verified_true: list[Assumption] = field(default_factory=list)
    """Assumptions confirmed as true."""

    verified_false: list[Assumption] = field(default_factory=list)
    """Assumptions confirmed as false."""

    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, list[dict[str, str | None]]]:
        """Convert to dictionary for serialization."""
        return {
            "unverified": [a.to_dict() for a in self.unverified],
            "true": [a.to_dict() for a in self.verified_true],
            "false": [a.to_dict() for a in self.verified_false],
        }

    ##Method purpose: Create from dict during deserialization
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AssumptionStore:
        """Create from dictionary."""
        return cls(
            unverified=[Assumption.from_dict(a) for a in data.get("unverified", [])],
            verified_true=[Assumption.from_dict(a) for a in data.get("true", [])],
            verified_false=[Assumption.from_dict(a) for a in data.get("false", [])],
        )

    ##Method purpose: Get all assumptions as flat list
    def all_assumptions(self) -> list[Assumption]:
        """Get all assumptions regardless of status."""
        return self.unverified + self.verified_true + self.verified_false

    ##Method purpose: Find assumption by content and username
    def find_by_content(
        self,
        content: str,
        username: str,
    ) -> tuple[Assumption, AssumptionStatus] | None:
        """
        Find assumption by content and username.

        Args:
            content: Assumption content to search for
            username: Username to match

        Returns:
            Tuple of (assumption, status) if found, None otherwise
        """
        ##Loop purpose: Search unverified assumptions
        for assumption in self.unverified:
            if assumption.content == content and assumption.username == username:
                return (assumption, AssumptionStatus.UNVERIFIED)

        ##Loop purpose: Search verified true assumptions
        for assumption in self.verified_true:
            if assumption.content == content and assumption.username == username:
                return (assumption, AssumptionStatus.TRUE)

        ##Loop purpose: Search verified false assumptions
        for assumption in self.verified_false:
            if assumption.content == content and assumption.username == username:
                return (assumption, AssumptionStatus.FALSE)

        return None
