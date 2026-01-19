##Script function and purpose: Unified KnowledgeStore combining memory and graph systems
"""
Knowledge Store

Unified interface to all knowledge sources. Combines vector memory (ChromaDB)
and graph memory (CognitiveGraph) into a single searchable knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from jenova.config.models import MemoryConfig, GraphConfig
from jenova.exceptions import GraphError
from jenova.memory import Memory, MemoryResult, MemoryType

if TYPE_CHECKING:
    from jenova.graph import CognitiveGraph

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Search result combining memory and graph results
@dataclass
class KnowledgeContext:
    """Combined search results from all knowledge sources."""
    
    memories: list[MemoryResult]
    """Results from vector memory search."""
    
    graph_context: list[dict[str, str]]
    """Related nodes from graph traversal."""
    
    query: str
    """Original search query."""
    
    ##Method purpose: Check if any results were found
    def is_empty(self) -> bool:
        """Check if no results found."""
        return len(self.memories) == 0 and len(self.graph_context) == 0
    
    ##Method purpose: Get combined text context for LLM
    def as_context_string(self, max_length: int = 4000) -> str:
        """Format as context string for LLM prompt."""
        ##Step purpose: Build context parts
        parts: list[str] = []
        
        ##Condition purpose: Add memory results if present
        if self.memories:
            memory_text = "\n".join(
                f"- [{m.memory_type.value}] {m.content}"
                for m in self.memories[:5]
            )
            parts.append(f"Relevant memories:\n{memory_text}")
        
        ##Condition purpose: Add graph context if present
        if self.graph_context:
            graph_text = "\n".join(
                f"- {node.get('label', 'Unknown')}: {node.get('content', '')}"
                for node in self.graph_context[:5]
            )
            parts.append(f"Related knowledge:\n{graph_text}")
        
        result = "\n\n".join(parts)
        
        ##Condition purpose: Truncate if too long
        if len(result) > max_length:
            return result[:max_length] + "..."
        
        return result


##Class purpose: Unified interface to all knowledge sources
class KnowledgeStore:
    """
    Unified interface to all knowledge sources.
    
    Combines vector memory and graph memory into a single
    searchable knowledge base.
    """
    
    ##Method purpose: Initialize with memory and graph configs
    def __init__(
        self,
        memory_config: MemoryConfig,
        graph_config: GraphConfig,
    ) -> None:
        """
        Initialize knowledge store.
        
        Args:
            memory_config: Configuration for memory system
            graph_config: Configuration for graph system
        """
        ##Step purpose: Store configuration
        self._memory_config = memory_config
        self._graph_config = graph_config
        
        ##Action purpose: Initialize memory instances for each type
        self._memories: dict[MemoryType, Memory] = {}
        ##Loop purpose: Create memory for each type
        for memory_type in MemoryType:
            self._memories[memory_type] = Memory(
                memory_type=memory_type,
                storage_path=memory_config.storage_path / memory_type.value,
            )
        
        ##Step purpose: Graph will be lazy-loaded to avoid circular import
        self._graph: CognitiveGraph | None = None
    
    ##Method purpose: Get or create the cognitive graph
    @property
    def graph(self) -> "CognitiveGraph":
        """Get the cognitive graph (lazy-loaded)."""
        ##Condition purpose: Initialize graph on first access
        if self._graph is None:
            from jenova.graph import CognitiveGraph
            self._graph = CognitiveGraph(self._graph_config.storage_path)
        return self._graph
    
    ##Method purpose: Get memory instance by type
    def get_memory(self, memory_type: MemoryType) -> Memory:
        """Get memory instance for a specific type."""
        return self._memories[memory_type]
    
    ##Method purpose: Add content to a specific memory type
    def add(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Add content to memory.
        
        Args:
            content: Text content to store
            memory_type: Which memory type to store in
            metadata: Optional metadata
            
        Returns:
            ID of stored memory
        """
        return self._memories[memory_type].add(content, metadata)
    
    ##Method purpose: Search all memory types and graph
    def search(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        n_results: int = 5,
        include_graph: bool = True,
    ) -> KnowledgeContext:
        """
        Search all knowledge sources.
        
        Args:
            query: Search query
            memory_types: Types to search (None = all)
            n_results: Max results per source
            include_graph: Whether to include graph context
            
        Returns:
            KnowledgeContext with combined results
        """
        ##Step purpose: Determine which memory types to search
        types_to_search = memory_types or list(MemoryType)
        
        ##Step purpose: Search all specified memories
        all_results: list[MemoryResult] = []
        ##Loop purpose: Search each memory type
        for memory_type in types_to_search:
            results = self._memories[memory_type].search(query, n_results)
            all_results.extend(results)
        
        ##Action purpose: Sort by score and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:n_results]
        
        ##Step purpose: Get graph context if requested
        graph_context: list[dict[str, str]] = []
        ##Condition purpose: Search graph if requested
        if include_graph:
            ##Error purpose: Handle graph errors gracefully
            try:
                graph_context = self.graph.search(query, max_results=n_results)
            except GraphError as e:
                ##Step purpose: Log graph-specific errors and continue
                logger.warning("graph_search_failed", error=str(e), query=query)
            except Exception as e:
                ##Step purpose: Log unexpected errors but don't crash search
                logger.error("unexpected_graph_error", error=str(e), query=query, exc_info=True)
        
        return KnowledgeContext(
            memories=all_results,
            graph_context=graph_context,
            query=query,
        )
    
    ##Method purpose: Factory method for production use
    @classmethod
    def create(
        cls,
        memory_config: MemoryConfig,
        graph_config: GraphConfig,
    ) -> "KnowledgeStore":
        """Factory method to create a KnowledgeStore."""
        return cls(memory_config, graph_config)
