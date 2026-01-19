##Script function and purpose: Type definitions for the insights module
"""
Insight Types

Type definitions for insight management. Insights are organized knowledge
extracted from conversations and stored under topical concerns (categories).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeAlias


##Step purpose: Define type alias for cortex node ID
CortexId: TypeAlias = str


##Class purpose: Immutable insight record
@dataclass(frozen=True)
class Insight:
    """An insight about the user or topic."""
    
    content: str
    """The insight content/statement."""
    
    username: str
    """User this insight relates to."""
    
    topic: str
    """Topic/concern this insight belongs to."""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """ISO timestamp of creation."""
    
    cortex_id: CortexId | None = None
    """Linked node ID in cognitive graph."""
    
    related_concerns: list[str] = field(default_factory=list)
    """Related topic/concern names."""
    
    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, str | list[str] | None]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "user": self.username,
            "topic": self.topic,
            "timestamp": self.timestamp,
            "cortex_id": self.cortex_id,
            "related_concerns": list(self.related_concerns),
        }
    
    ##Method purpose: Create from dict during deserialization
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Insight":
        """Create from dictionary."""
        return cls(
            content=str(data["content"]),
            username=str(data.get("user", data.get("username", ""))),
            topic=str(data.get("topic", "general")),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
            cortex_id=data.get("cortex_id"),
            related_concerns=list(data.get("related_concerns", [])),
        )
    
    ##Method purpose: Create new insight with updated fields
    def with_updates(
        self,
        content: str | None = None,
        topic: str | None = None,
        cortex_id: CortexId | None = None,
        related_concerns: list[str] | None = None,
    ) -> "Insight":
        """
        Create new Insight with updated fields.
        
        Args:
            content: New content (None to keep existing)
            topic: New topic (None to keep existing)
            cortex_id: New cortex ID (None to keep existing)
            related_concerns: New related concerns (None to keep existing)
            
        Returns:
            New Insight instance with updates
        """
        return Insight(
            content=content if content is not None else self.content,
            username=self.username,
            topic=topic if topic is not None else self.topic,
            timestamp=datetime.now().isoformat(),
            cortex_id=cortex_id if cortex_id is not None else self.cortex_id,
            related_concerns=related_concerns if related_concerns is not None else list(self.related_concerns),
        )


##Class purpose: Concern (topic category) for organizing insights
@dataclass
class Concern:
    """A concern/topic for organizing insights."""
    
    name: str
    """The concern name (used as directory name)."""
    
    description: str
    """Description of what this concern covers."""
    
    related_concerns: list[str] = field(default_factory=list)
    """Names of related concerns."""
    
    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, str | list[str]]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "related_concerns": list(self.related_concerns),
        }
    
    ##Method purpose: Create from dict during deserialization
    @classmethod
    def from_dict(cls, name: str, data: dict[str, object]) -> "Concern":
        """Create from dictionary."""
        return cls(
            name=name,
            description=str(data.get("description", "")),
            related_concerns=list(data.get("related_concerns", [])),
        )


##Class purpose: Container for insight search results
@dataclass(frozen=True)
class InsightSearchResult:
    """Result from insight search."""
    
    insight: Insight
    """The matched insight."""
    
    score: float
    """Relevance score (0-1, higher is better)."""
    
    filepath: Path | None = None
    """Path to insight file on disk."""
