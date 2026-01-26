##Script function and purpose: Graph package initialization - exposes CognitiveGraph and types
"""Graph-based cognitive memory for JENOVA."""

from jenova.graph.graph import CognitiveGraph, LLMProtocol
from jenova.graph.proactive import (
    EngagementTracker,
    ProactiveConfig,
    ProactiveEngine,
    Suggestion,
    SuggestionCategory,
)
from jenova.graph.types import (
    ClusterResult,
    ConnectionSuggestion,
    ContradictionResult,
    Edge,
    EdgeType,
    Emotion,
    EmotionResult,
    GraphQuery,
    Node,
)

__all__ = [
    # Core graph
    "CognitiveGraph",
    "LLMProtocol",
    # Types
    "Node",
    "Edge",
    "EdgeType",
    "GraphQuery",
    # Emotion analysis
    "Emotion",
    "EmotionResult",
    # Clustering
    "ClusterResult",
    # Contradiction detection
    "ContradictionResult",
    # Connection suggestions
    "ConnectionSuggestion",
    # Proactive engine
    "ProactiveEngine",
    "ProactiveConfig",
    "Suggestion",
    "SuggestionCategory",
    "EngagementTracker",
]
