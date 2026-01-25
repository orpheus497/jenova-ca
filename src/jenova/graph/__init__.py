##Script function and purpose: Graph package initialization - exposes CognitiveGraph and types
"""Graph-based cognitive memory for JENOVA."""

from jenova.graph.types import (
    Node,
    Edge,
    EdgeType,
    GraphQuery,
    Emotion,
    EmotionResult,
    ClusterResult,
    ContradictionResult,
    ConnectionSuggestion,
)
from jenova.graph.graph import CognitiveGraph, LLMProtocol
from jenova.graph.proactive import (
    ProactiveEngine,
    ProactiveConfig,
    Suggestion,
    SuggestionCategory,
    EngagementTracker,
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
