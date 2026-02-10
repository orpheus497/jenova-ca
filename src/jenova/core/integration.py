##Script function and purpose: Integration Layer - Coordinates Memory, Cortex, and Insights systems
##Dependency purpose: Provides unified knowledge representation and bidirectional feedback loops
"""
Integration Layer

Coordinates integration between Memory systems and Cortex graph.
Provides unified knowledge representation and bidirectional feedback loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

from jenova.exceptions import (
    ConsistencyError,
    GraphError,
    IntegrationError,
    NodeNotFoundError,
)

if TYPE_CHECKING:
    from jenova.graph.types import Node
    from jenova.memory.types import MemoryResult


##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for graph operations needed by integration
@runtime_checkable
class GraphProtocol(Protocol):
    """Protocol defining graph operations needed by IntegrationHub.

    Defines the graph interface for Memory-Cortex integration.
    Implementations: CognitiveGraph (src/jenova/graph/graph.py)

    Contract:
        - search: Text search returning node dicts with id, label, content keys
        - all_nodes: Return all nodes in graph (may be expensive for large graphs)
        - get_node: Return node by ID, raise NodeNotFoundError if missing
        - has_node: Return True if node exists
        - add_node: Persist node to graph storage
        - neighbors: Return connected nodes in specified direction
    """

    ##Method purpose: Search graph for matching nodes
    def search(
        self,
        query: str,
        max_results: int = 10,
        node_types: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Search graph for nodes matching query text.

        Args:
            query: Text to search for in node labels and content
            max_results: Maximum nodes to return
            node_types: Filter by node types (None = all types)

        Returns:
            List of dicts with 'id', 'label', 'content' keys
        """
        ...

    ##Method purpose: Get all nodes in the graph
    def all_nodes(self) -> list[Node]:
        """Get all nodes in the graph.

        Warning: May be expensive for large graphs. Use sparingly.

        Returns:
            List of all Node objects
        """
        ...

    ##Method purpose: Get a node by ID
    def get_node(self, node_id: str) -> Node:
        """Get a specific node by ID.

        Args:
            node_id: UUID string of the node

        Returns:
            Node object

        Raises:
            NodeNotFoundError: If node does not exist
        """
        ...

    ##Method purpose: Check if node exists
    def has_node(self, node_id: str) -> bool:
        """Check if node exists in graph.

        Args:
            node_id: UUID string of the node

        Returns:
            True if node exists, False otherwise
        """
        ...

    ##Method purpose: Add a node to the graph
    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node object to add

        Raises:
            GraphError: If node cannot be persisted
        """
        ...

    ##Method purpose: Get neighboring nodes
    def neighbors(self, node_id: str, direction: str = "out") -> list[Node]:
        """Get nodes connected to the specified node.

        Args:
            node_id: UUID string of the source node
            direction: 'out' for outgoing edges, 'in' for incoming, 'both' for all

        Returns:
            List of connected Node objects
        """
        ...


##Class purpose: Protocol for memory search operations
@runtime_checkable
class MemorySearchProtocol(Protocol):
    """Protocol defining memory search operations.

    Defines semantic search interface for integration with Cortex.
    Implementations: Memory (src/jenova/memory/memory.py)

    Note: Memory.search() returns list[MemoryResult] - this protocol
    matches that signature. The IntegrationHub expects MemoryResult objects.

    Contract:
        - search: Must return MemoryResult objects sorted by relevance (score)
        - Higher score = more relevant
    """

    ##Method purpose: Search memory for relevant content
    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[MemoryResult]:
        """Search memory for semantically similar content.

        Args:
            query: Search query text
            n_results: Maximum number of results to return

        Returns:
            List of MemoryResult objects, sorted by relevance (higher score first)
        """
        ...


##Class purpose: Result of finding related Cortex nodes
@dataclass(frozen=True)
class RelatedNodeResult:
    """Result of finding a related Cortex node."""

    node_id: str
    """ID of the related node."""

    content: str
    """Content of the node."""

    label: str
    """Label/title of the node."""

    similarity_score: float
    """Similarity score (0.0 to 1.0)."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional node metadata."""


##Class purpose: Cross-reference between Memory and Cortex
@dataclass(frozen=True)
class CrossReference:
    """Cross-reference between a memory item and Cortex nodes."""

    memory_content: str
    """Content from memory."""

    related_node_ids: list[str]
    """IDs of related Cortex nodes."""

    similarity_scores: list[float]
    """Similarity scores for each related node."""


##Class purpose: Knowledge gap between Memory and Cortex
@dataclass(frozen=True)
class KnowledgeGap:
    """A gap in knowledge between Memory and Cortex."""

    gap_type: str
    """Type of gap: 'high_centrality_not_in_memory' or 'isolated_knowledge'."""

    node_id: str
    """ID of the node representing the gap."""

    content: str
    """Content of the knowledge gap."""

    severity: float
    """Severity score (0.0 to 1.0)."""


##Class purpose: Knowledge duplication between Memory and Cortex
@dataclass(frozen=True)
class KnowledgeDuplication:
    """Duplication of knowledge between Memory and Cortex."""

    memory_content: str
    """Content from memory."""

    node_id: str
    """ID of the duplicating node."""

    similarity_score: float
    """How similar the content is (0.0 to 1.0)."""


##Class purpose: Unified knowledge map combining Memory and Cortex
@dataclass
class UnifiedKnowledgeMap:
    """Unified representation of knowledge from Memory and Cortex."""

    memory_items: list[str]
    """Sample of memory items."""

    cortex_nodes: list[dict[str, str]]
    """Sample of Cortex nodes as dicts."""

    cross_references: list[CrossReference]
    """Cross-references between Memory and Cortex."""

    knowledge_gaps: list[KnowledgeGap]
    """Identified knowledge gaps."""

    ##Method purpose: Check if knowledge map is empty
    def is_empty(self) -> bool:
        """Check if knowledge map has no content."""
        return len(self.memory_items) == 0 and len(self.cortex_nodes) == 0


##Class purpose: Report on knowledge consistency between systems
@dataclass
class ConsistencyReport:
    """Report on consistency between Memory and Cortex knowledge."""

    is_consistent: bool
    """Whether knowledge is consistent across systems."""

    gaps: list[KnowledgeGap]
    """Identified knowledge gaps."""

    duplications: list[KnowledgeDuplication]
    """Identified duplications."""

    recommendations: list[str]
    """Recommendations for improving consistency."""

    ##Method purpose: Get summary of consistency issues
    def summary(self) -> str:
        """Get human-readable summary of consistency report."""
        if self.is_consistent:
            return "Knowledge is consistent across Memory and Cortex."

        parts: list[str] = []
        if self.gaps:
            parts.append(f"{len(self.gaps)} knowledge gaps found")
        if self.duplications:
            parts.append(f"{len(self.duplications)} duplications found")

        return "; ".join(parts) if parts else "No issues found."


##Class purpose: Configuration for IntegrationHub
@dataclass
class IntegrationConfig:
    """Configuration for IntegrationHub."""

    enabled: bool = True
    """Whether integration is enabled."""

    max_related_nodes: int = 5
    """Maximum related nodes to find per query."""

    max_context_expansion: int = 3
    """Maximum items to add during context expansion."""

    similarity_threshold: float = 0.7
    """Minimum similarity for creating cross-references."""

    high_centrality_threshold: float = 2.0
    """Centrality threshold for identifying important nodes."""

    duplication_threshold: float = 0.9
    """Similarity threshold for flagging duplications."""

    memory_to_cortex_feedback: bool = True
    """Whether to create Memory → Cortex feedback links."""


##Class purpose: Central hub coordinating Memory, Cortex, and Insights integration
class IntegrationHub:
    """
    Coordinates integration between Memory systems and Cortex graph.

    Provides:
    - Finding related Cortex nodes for memory content
    - Calculating centrality scores
    - Expanding context with relationships
    - Creating unified knowledge maps
    - Bidirectional feedback between Memory and Cortex
    - Knowledge consistency checking
    """

    ##Method purpose: Initialize integration hub with graph and memory references
    def __init__(
        self,
        graph: GraphProtocol,
        memory: MemorySearchProtocol,
        config: IntegrationConfig | None = None,
    ) -> None:
        """
        Initialize IntegrationHub.

        Args:
            graph: Cognitive graph for relationship storage
            memory: Memory system for vector search
            config: Optional integration configuration
        """
        ##Step purpose: Store dependencies
        self._graph = graph
        self._memory = memory
        self._config = config or IntegrationConfig()

        ##Action purpose: Log initialization
        logger.info(
            "integration_hub_initialized",
            enabled=self._config.enabled,
            max_related_nodes=self._config.max_related_nodes,
        )

    ##Method purpose: Find Cortex nodes semantically related to content
    def find_related_nodes(
        self,
        content: str,
        username: str,
        max_nodes: int | None = None,
    ) -> list[RelatedNodeResult]:
        """
        Find Cortex nodes semantically related to content.

        Args:
            content: Content to find related nodes for
            username: Username to filter nodes by
            max_nodes: Maximum nodes to return (defaults to config)

        Returns:
            List of RelatedNodeResult with similarity scores
        """
        ##Condition purpose: Return empty if integration disabled
        if not self._config.enabled:
            return []

        max_nodes = max_nodes or self._config.max_related_nodes

        ##Step purpose: Search graph for related nodes
        try:
            ##Action purpose: Get graph search results
            search_results = self._graph.search(
                query=content,
                max_results=max_nodes * 2,  # Get extra to filter by user
            )

            ##Step purpose: Filter results by username and convert to typed results
            related: list[RelatedNodeResult] = []

            ##Loop purpose: Process each search result
            for i, result in enumerate(search_results):
                ##Step purpose: Get full node to check username
                node_id = result.get("id", "")
                if not node_id:
                    continue

                ##Condition purpose: Check if node belongs to user
                try:
                    node = self._graph.get_node(node_id)
                    node_user = node.metadata.get("user", "")

                    ##Condition purpose: Skip if different user
                    if node_user and node_user != username:
                        continue

                    ##Step purpose: Calculate similarity score based on position
                    ##Note: Simple position-based scoring; could be enhanced with embeddings
                    similarity = 1.0 / (1.0 + i * 0.2)

                    ##Fix: Guard against None metadata
                    related.append(
                        RelatedNodeResult(
                            node_id=node.id,
                            content=node.content,
                            label=node.label,
                            similarity_score=similarity,
                            metadata=dict(node.metadata) if node.metadata else {},
                        )
                    )

                    ##Condition purpose: Stop if we have enough
                    if len(related) >= max_nodes:
                        break

                except Exception as e:
                    logger.warning("failed_to_get_node", node_id=node_id, error=str(e))
                    continue

            return related

        except Exception as e:
            logger.error("find_related_nodes_failed", error=str(e), content=content[:100])
            return []

    ##Method purpose: Get centrality score for content from Cortex
    def get_centrality_score(self, content: str, username: str) -> float:
        """
        Get centrality score for content based on related Cortex nodes.

        Args:
            content: Content to get centrality for
            username: Username to filter by

        Returns:
            Normalized centrality score (0.0 to 1.0)
        """
        ##Condition purpose: Return zero if integration disabled
        if not self._config.enabled:
            return 0.0

        ##Step purpose: Find related nodes
        related = self.find_related_nodes(content, username, max_nodes=3)

        ##Condition purpose: Return zero if no related nodes
        if not related:
            return 0.0

        ##Step purpose: Calculate weighted average centrality
        total_centrality = 0.0
        total_weight = 0.0

        ##Loop purpose: Sum weighted centralities
        for result in related:
            centrality = float(result.metadata.get("centrality", "0"))
            weight = result.similarity_score

            total_centrality += centrality * weight
            total_weight += weight

        ##Condition purpose: Avoid division by zero
        if total_weight == 0:
            return 0.0

        avg_centrality = total_centrality / total_weight

        ##Step purpose: Normalize to 0-1 range (assuming max centrality ~10)
        normalized = min(avg_centrality / 10.0, 1.0)

        return normalized

    ##Method purpose: Expand context by following Cortex relationships
    def expand_context_with_relationships(
        self,
        context_items: list[str],
        username: str,
        max_expansion: int | None = None,
    ) -> list[str]:
        """
        Expand context by following Cortex graph relationships.

        Args:
            context_items: Initial context items
            username: Username to filter by
            max_expansion: Maximum items to add (defaults to config)

        Returns:
            Expanded context list with related content
        """
        ##Condition purpose: Return original if integration disabled
        if not self._config.enabled:
            return context_items

        max_expansion = max_expansion or self._config.max_context_expansion

        ##Step purpose: Track what we've added
        expanded = list(context_items)
        added_content: set[str] = set(context_items)
        items_added = 0

        ##Loop purpose: Find related content for initial items
        for item in context_items[:5]:  # Limit to avoid explosion
            related = self.find_related_nodes(item, username, max_nodes=max_expansion)

            ##Loop purpose: Add related content if not duplicate
            for result in related:
                ##Condition purpose: Skip if already added
                if result.content in added_content:
                    continue

                expanded.append(result.content)
                added_content.add(result.content)
                items_added += 1

                ##Condition purpose: Stop if we've added enough
                if items_added >= max_expansion:
                    break

            if items_added >= max_expansion:
                break

        logger.debug(
            "context_expanded",
            original_count=len(context_items),
            expanded_count=len(expanded),
            items_added=items_added,
        )

        return expanded

    ##Method purpose: Create unified knowledge representation
    def build_unified_context(
        self,
        username: str,
        sample_query: str = "general knowledge",
        max_items: int = 10,
    ) -> UnifiedKnowledgeMap:
        """
        Create unified representation combining Memory and Cortex knowledge.

        Args:
            username: Username to get knowledge for
            sample_query: Query to sample memory with
            max_items: Maximum items per source

        Returns:
            UnifiedKnowledgeMap with combined knowledge
        """
        ##Condition purpose: Return empty map if integration disabled
        if not self._config.enabled:
            return UnifiedKnowledgeMap(
                memory_items=[],
                cortex_nodes=[],
                cross_references=[],
                knowledge_gaps=[],
            )

        ##Step purpose: Sample memory items
        memory_items: list[str] = []
        try:
            results = self._memory.search(sample_query, n_results=max_items)
            memory_items = [r.content for r in results]
        except Exception as e:
            logger.error("failed_to_sample_memory", error=str(e))

        ##Step purpose: Get Cortex nodes for user
        cortex_nodes: list[dict[str, str]] = []
        try:
            ##Update: Use existing get_nodes_by_user() method for better performance
            user_nodes = self._graph.get_nodes_by_user(username)
            ##Loop purpose: Convert to cortex format with limit
            for node in user_nodes[: max_items * 2]:
                cortex_nodes.append(
                    {
                        "id": node.id,
                        "label": node.label,
                        "content": node.content,
                    }
                )
        except Exception as e:
            logger.error("failed_to_get_cortex_nodes", error=str(e))

        ##Step purpose: Find cross-references
        cross_references: list[CrossReference] = []
        ##Loop purpose: Find related nodes for each memory item
        for memory_item in memory_items[:5]:
            related = self.find_related_nodes(memory_item, username, max_nodes=2)
            if related:
                cross_references.append(
                    CrossReference(
                        memory_content=memory_item[:200],
                        related_node_ids=[r.node_id for r in related],
                        similarity_scores=[r.similarity_score for r in related],
                    )
                )

        ##Step purpose: Identify knowledge gaps (high centrality not in memory)
        knowledge_gaps: list[KnowledgeGap] = []
        ##Loop purpose: Check high centrality nodes
        for node_dict in cortex_nodes:
            node_id = node_dict.get("id", "")
            if not node_id:
                continue

            try:
                node = self._graph.get_node(node_id)
                centrality = float(node.metadata.get("centrality", "0"))

                ##Condition purpose: Flag high centrality nodes not well represented
                if centrality > self._config.high_centrality_threshold:
                    ##Step purpose: Check if represented in memory
                    related = self.find_related_nodes(node.content, username, max_nodes=1)

                    if not related or related[0].similarity_score < 0.5:
                        knowledge_gaps.append(
                            KnowledgeGap(
                                gap_type="high_centrality_not_in_memory",
                                node_id=node_id,
                                content=node.content[:200],
                                severity=min(centrality / 10.0, 1.0),
                            )
                        )
            except (NodeNotFoundError, GraphError, AttributeError, ValueError, KeyError) as e:
                ##Fix: Handle expected exceptions gracefully with logging - prevents silent failures
                logger.warning("node_processing_failed", node_id=node_id, error=str(e))
                continue
            except Exception as e:
                ##Fix: Re-raise unexpected exceptions with context - prevents hiding critical errors
                logger.error("unexpected_error_in_context_building", node_id=node_id, error=str(e))
                raise IntegrationError(f"Unexpected error processing node {node_id}") from e

        logger.info(
            "unified_context_built",
            username=username,
            memory_items=len(memory_items),
            cortex_nodes=len(cortex_nodes),
            cross_references=len(cross_references),
            knowledge_gaps=len(knowledge_gaps),
        )

        return UnifiedKnowledgeMap(
            memory_items=memory_items,
            cortex_nodes=cortex_nodes[:max_items],
            cross_references=cross_references,
            knowledge_gaps=knowledge_gaps,
        )

    ##Method purpose: Check knowledge consistency between Memory and Cortex
    def check_consistency(self, username: str) -> ConsistencyReport:
        """
        Check for consistency and gaps between Memory and Cortex knowledge.

        Args:
            username: Username to check consistency for

        Returns:
            ConsistencyReport with gaps, duplications, and recommendations
        """
        ##Condition purpose: Return consistent if integration disabled
        if not self._config.enabled:
            return ConsistencyReport(
                is_consistent=True,
                gaps=[],
                duplications=[],
                recommendations=[],
            )

        ##Step purpose: Build unified context to analyze
        knowledge_map = self.build_unified_context(username)

        ##Step purpose: Collect duplications
        duplications: list[KnowledgeDuplication] = []

        ##Loop purpose: Check each memory item for high-similarity nodes
        for memory_item in knowledge_map.memory_items[:5]:
            related = self.find_related_nodes(memory_item, username, max_nodes=1)

            if related and related[0].similarity_score > self._config.duplication_threshold:
                duplications.append(
                    KnowledgeDuplication(
                        memory_content=memory_item[:200],
                        node_id=related[0].node_id,
                        similarity_score=related[0].similarity_score,
                    )
                )

        ##Step purpose: Generate recommendations
        recommendations: list[str] = []

        ##Condition purpose: Add recommendation for gaps
        if knowledge_map.knowledge_gaps:
            recommendations.append(
                "Consider adding high-centrality Cortex nodes to memory for better retrieval"
            )

        ##Condition purpose: Add recommendation for duplications
        if duplications:
            recommendations.append(
                "Consider consolidating duplicate knowledge between Memory and Cortex"
            )

        is_consistent = len(knowledge_map.knowledge_gaps) == 0 and len(duplications) == 0

        return ConsistencyReport(
            is_consistent=is_consistent,
            gaps=knowledge_map.knowledge_gaps,
            duplications=duplications,
            recommendations=recommendations,
        )

    ##Method purpose: Propagate update from Memory to Cortex
    def propagate_memory_to_cortex(
        self,
        memory_content: str,
        memory_type: str,
        username: str,
    ) -> str | None:
        """
        Propagate an update from Memory to Cortex, creating connections if needed.

        Args:
            memory_content: Content from memory
            memory_type: Type of memory (episodic, semantic, procedural)
            username: Username for the memory

        Returns:
            Node ID if a reference node was created, None otherwise
        """
        ##Condition purpose: Return None if feedback disabled
        if not self._config.enabled or not self._config.memory_to_cortex_feedback:
            return None

        ##Step purpose: Find related nodes
        related = self.find_related_nodes(memory_content, username, max_nodes=3)

        ##Condition purpose: Return None if no related nodes
        if not related:
            return None

        ##Step purpose: Check if best match is strong enough
        best_match = related[0]
        if best_match.similarity_score < self._config.similarity_threshold:
            return None

        ##Step purpose: Create memory reference node in Cortex
        try:
            import uuid
            from datetime import datetime

            from jenova.graph.types import Node

            node_id = f"memory_ref_{uuid.uuid4().hex[:8]}"

            memory_node = Node(
                id=node_id,
                label=f"Memory Reference: {memory_type}",
                content=memory_content[:500],
                node_type="memory_reference",
                metadata={
                    "user": username,
                    "memory_type": memory_type,
                    "similarity_score": str(best_match.similarity_score),
                    "linked_to": best_match.node_id,
                    "created": datetime.now().isoformat(),
                },
            )

            self._graph.add_node(memory_node)

            logger.info(
                "memory_reference_created",
                node_id=node_id,
                linked_to=best_match.node_id,
                memory_type=memory_type,
            )

            return node_id

        except (GraphError, NodeNotFoundError, ValueError) as e:
            ##Fix: Catch specific exceptions and re-raise with context - prevents silent failures
            logger.error(
                "failed_to_create_memory_reference",
                error=str(e),
                memory_type=memory_type,
            )
            raise IntegrationError(f"Failed to create memory reference: {e}") from e
        except Exception as e:
            ##Fix: Re-raise unexpected exceptions with context - prevents hiding critical errors
            logger.error(
                "unexpected_error_creating_memory_reference",
                error=str(e),
                memory_type=memory_type,
            )
            raise IntegrationError(f"Unexpected error creating memory reference: {e}") from e

    ##Method purpose: Get unified knowledge view for a query
    def get_knowledge_map(
        self,
        query: str,
        username: str,
        max_items: int = 10,
    ) -> UnifiedKnowledgeMap:
        """
        Get unified knowledge view for a specific query.

        Args:
            query: Query to get knowledge for
            username: Username to filter by
            max_items: Maximum items per source

        Returns:
            UnifiedKnowledgeMap focused on the query
        """
        return self.build_unified_context(
            username=username,
            sample_query=query,
            max_items=max_items,
        )
