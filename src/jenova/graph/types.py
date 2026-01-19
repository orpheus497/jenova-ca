##Script function and purpose: Graph type definitions - Node, Edge, and query types
"""
Graph Types

Type definitions for the cognitive graph. Uses dataclasses for
immutable data containers, avoiding heavy dependencies like networkx.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


##Class purpose: Enum defining relationship types between nodes
class EdgeType(Enum):
    """Types of edges in the cognitive graph."""
    
    RELATES_TO = "relates_to"
    """General relationship."""
    
    IMPLIES = "implies"
    """Logical implication."""
    
    CONTRADICTS = "contradicts"
    """Contradiction relationship."""
    
    SUPPORTS = "supports"
    """Supporting evidence."""
    
    CAUSED_BY = "caused_by"
    """Causal relationship."""
    
    INSTANCE_OF = "instance_of"
    """Type/instance relationship."""
    
    HAS_CHILD = "has_child"
    """Parent-child hierarchical relationship."""
    
    PART_OF = "part_of"
    """Part-whole relationship."""


##Class purpose: Immutable node in the cognitive graph
@dataclass(frozen=True)
class Node:
    """A node in the cognitive graph."""
    
    id: str
    """Unique node identifier."""
    
    label: str
    """Human-readable label."""
    
    content: str
    """Full content/description."""
    
    node_type: str
    """Type category (concept, fact, event, etc.)."""
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    """ISO timestamp of creation."""
    
    metadata: dict[str, str] = field(default_factory=dict)
    """Additional metadata."""
    
    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, str | dict[str, str]]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "content": self.content,
            "node_type": self.node_type,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    ##Method purpose: Create from dict during deserialization
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Node":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            label=data["label"],
            content=data["content"],
            node_type=data["node_type"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )
    
    ##Method purpose: Factory method to create a new node with generated ID
    @classmethod
    def create(
        cls,
        label: str,
        content: str,
        node_type: str,
        metadata: dict[str, str] | None = None,
    ) -> "Node":
        """
        Create a new node with auto-generated ID.
        
        Args:
            label: Human-readable label.
            content: Full content/description.
            node_type: Type category (concept, fact, event, etc.).
            metadata: Optional additional metadata.
            
        Returns:
            New Node instance with generated UUID.
        """
        return cls(
            id=str(uuid.uuid4()),
            label=label,
            content=content,
            node_type=node_type,
            metadata=metadata or {},
        )


##Class purpose: Immutable edge connecting two nodes
@dataclass(frozen=True)
class Edge:
    """An edge connecting two nodes in the graph."""
    
    source_id: str
    """ID of source node."""
    
    target_id: str
    """ID of target node."""
    
    edge_type: EdgeType
    """Type of relationship."""
    
    weight: float = 1.0
    """Edge weight (0-1, higher = stronger)."""
    
    metadata: dict[str, str] = field(default_factory=dict)
    """Additional metadata."""
    
    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, str | float | dict[str, str]]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }
    
    ##Method purpose: Create from dict during deserialization
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Edge":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


##Class purpose: Query parameters for graph searches
@dataclass
class GraphQuery:
    """Query parameters for graph search."""
    
    query_text: str
    """Text to search for in node content."""
    
    node_types: list[str] | None = None
    """Filter by node types (None = all)."""
    
    edge_types: list[EdgeType] | None = None
    """Filter by edge types (None = all)."""
    
    max_depth: int = 2
    """Maximum traversal depth from matches."""
    
    max_results: int = 10
    """Maximum nodes to return."""


##Class purpose: Enum defining emotion categories for content analysis
class Emotion(Enum):
    """Emotion categories for content analysis."""
    
    JOY = "joy"
    """Positive happiness or delight."""
    
    SADNESS = "sadness"
    """Negative sorrow or disappointment."""
    
    ANGER = "anger"
    """Negative frustration or irritation."""
    
    SURPRISE = "surprise"
    """Unexpected revelation or astonishment."""
    
    FEAR = "fear"
    """Apprehension or anxiety."""
    
    DISGUST = "disgust"
    """Revulsion or strong disapproval."""
    
    LOVE = "love"
    """Affection or deep caring."""
    
    CURIOSITY = "curiosity"
    """Interest or eagerness to learn."""
    
    NEUTRAL = "neutral"
    """No strong emotional content."""


##Class purpose: Result container for emotion analysis
@dataclass(frozen=True)
class EmotionResult:
    """Result of emotion analysis on content."""
    
    primary_emotion: Emotion
    """The dominant emotion detected."""
    
    confidence: float
    """Confidence score (0.0 to 1.0)."""
    
    emotion_scores: dict[str, float] = field(default_factory=dict)
    """Scores for all detected emotions."""
    
    content_preview: str = ""
    """Preview of analyzed content."""
    
    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "primary_emotion": self.primary_emotion.value,
            "confidence": self.confidence,
            "emotion_scores": self.emotion_scores,
            "content_preview": self.content_preview,
        }
    
    ##Method purpose: Create from dict during deserialization
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EmotionResult":
        """Create from dictionary."""
        return cls(
            primary_emotion=Emotion(data["primary_emotion"]),
            confidence=float(data["confidence"]),
            emotion_scores=data.get("emotion_scores", {}),
            content_preview=str(data.get("content_preview", "")),
        )


##Class purpose: Result container for graph clustering operations
@dataclass
class ClusterResult:
    """Result of clustering nodes in the graph."""
    
    clusters: dict[str, list[str]]
    """Mapping of cluster ID to list of node IDs."""
    
    cluster_labels: dict[str, str]
    """Human-readable labels for each cluster."""
    
    orphan_nodes: list[str]
    """Node IDs that couldn't be clustered."""
    
    quality_score: float = 0.0
    """Overall clustering quality (0.0 to 1.0)."""
    
    ##Method purpose: Get total number of clusters
    @property
    def cluster_count(self) -> int:
        """Get total number of clusters."""
        return len(self.clusters)
    
    ##Method purpose: Get total clustered node count
    @property
    def clustered_node_count(self) -> int:
        """Get total number of clustered nodes."""
        return sum(len(nodes) for nodes in self.clusters.values())


##Class purpose: Result container for contradiction detection
@dataclass(frozen=True)
class ContradictionResult:
    """A detected contradiction between two nodes."""
    
    node_a_id: str
    """First node in the contradiction."""
    
    node_b_id: str
    """Second node in the contradiction."""
    
    explanation: str
    """LLM-generated explanation of the contradiction."""
    
    confidence: float
    """Confidence that this is a true contradiction (0.0 to 1.0)."""
    
    ##Method purpose: Get tuple representation
    def as_tuple(self) -> tuple[str, str]:
        """Return as simple tuple of node IDs."""
        return (self.node_a_id, self.node_b_id)


##Class purpose: Result container for connection suggestions
@dataclass
class ConnectionSuggestion:
    """A suggested connection between nodes."""
    
    source_id: str
    """Source node ID."""
    
    target_id: str
    """Target node ID."""
    
    suggested_edge_type: EdgeType
    """Recommended edge type."""
    
    reasoning: str
    """Why this connection is suggested."""
    
    strength: float = 0.5
    """Suggested connection strength (0.0 to 1.0)."""
