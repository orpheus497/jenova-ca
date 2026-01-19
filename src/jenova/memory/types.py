##Script function and purpose: Memory type definitions - MemoryType enum and MemoryResult dataclass
"""
Memory Types

Type definitions for the memory system. MemoryType enum replaces
the legacy 3-class approach (EpisodicMemory, SemanticMemory, ProceduralMemory).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


##Class purpose: Enum defining the three memory types
class MemoryType(Enum):
    """Types of memory in the cognitive architecture."""
    
    EPISODIC = "episodic"
    """Personal experiences and events."""
    
    SEMANTIC = "semantic"
    """Facts and general knowledge."""
    
    PROCEDURAL = "procedural"
    """Skills and how-to knowledge."""


##Class purpose: Immutable result from memory search
@dataclass(frozen=True)
class MemoryResult:
    """Result from a memory search operation."""
    
    id: str
    """Unique identifier for this memory."""
    
    content: str
    """The stored content text."""
    
    score: float
    """Relevance score (0-1, higher is more relevant)."""
    
    memory_type: MemoryType
    """Type of memory this result came from."""
    
    metadata: dict[str, str]
    """Additional metadata stored with the memory."""
    
    ##Method purpose: String representation for debugging
    def __repr__(self) -> str:
        return (
            f"MemoryResult(id={self.id!r}, score={self.score:.3f}, "
            f"type={self.memory_type.value}, content={self.content[:50]!r}...)"
        )
