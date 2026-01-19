##Script function and purpose: Dict-based cognitive graph without networkx dependency
"""
Cognitive Graph

Dict-based graph implementation for cognitive relationships.
Replaces networkx dependency with lightweight dict operations.
Includes advanced cognitive features: emotion analysis, clustering,
orphan linking, meta-insight generation, and contradiction detection.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, TYPE_CHECKING

import structlog

from jenova.exceptions import (
    NodeNotFoundError,
    GraphError,
    GraphPruneError,
    GraphClusterError,
    GraphAnalysisError,
    LLMError,
)
from jenova.graph.types import (
    Node,
    Edge,
    EdgeType,
    GraphQuery,
    Emotion,
    EmotionResult,
)
from jenova.utils.migrations import load_json_with_migration, save_json_atomic
from jenova.utils.json_safe import safe_json_loads, extract_json_from_response, JSONSizeError
from jenova.utils.sanitization import sanitize_for_prompt
##Sec: Import Pydantic validators for LLM output validation (P1-001)
from jenova.graph.llm_schemas import (
    EmotionAnalysisResponse,
    RelationshipAnalysisResponse,
    ContradictionCheckResponse,
    ConnectionSuggestionsResponse,
)
from pydantic import ValidationError

if TYPE_CHECKING:
    from jenova.llm.interface import LLMInterface

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)

##Step purpose: Define constants for prompt sanitization and limits
MAX_CONTENT_LENGTH: int = 500  # Maximum content length for prompts
MAX_CONTRADICTION_NODES: int = 20  # Maximum nodes to check for contradictions
MAX_CONTRADICTION_COMPARISONS: int = 50  # Maximum LLM calls for contradiction detection
MAX_ORPHANS_PER_BATCH: int = 10  # Maximum orphan nodes to process per call
MAX_CONTEXT_NODES: int = 20  # Maximum context nodes for prompts
MAX_CLUSTER_NODES_FOR_INSIGHT: int = 10  # Maximum nodes per cluster for meta-insight generation
MAX_CLUSTER_NODES_FOR_LABEL: int = 5  # Maximum nodes per cluster for label generation
MAX_CANDIDATE_NODES: int = 30  # Maximum candidate nodes for connection suggestions


##Class purpose: Protocol for LLM operations (dependency injection)
class LLMProtocol(Protocol):
    """Protocol for LLM operations used by CognitiveGraph."""
    
    ##Method purpose: Generate text from a prompt
    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate text completion."""
        ...


##Class purpose: Dict-based cognitive graph with persistence
class CognitiveGraph:
    """
    Cognitive graph for relationship storage.
    
    Uses dict-based storage instead of networkx for simplicity.
    Persists to JSON with schema versioning.
    """
    
    ##Method purpose: Initialize graph with storage path
    def __init__(self, storage_path: Path) -> None:
        """
        Initialize cognitive graph.
        
        Args:
            storage_path: Directory for graph data persistence
        """
        ##Step purpose: Store configuration
        self.storage_path = storage_path
        self._graph_file = storage_path / "graph.json"
        
        ##Step purpose: Initialize data structures
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[Edge]] = defaultdict(list)
        self._reverse_edges: dict[str, list[Edge]] = defaultdict(list)
        
        ##Action purpose: Load existing data if present
        self._load()
    
    ##Method purpose: Sanitize user content to prevent prompt injection
    def _sanitize_for_prompt(self, content: str) -> str:
        """
        Remove potential injection patterns from user content.
        
        Uses centralized sanitization utility for consistency.
        
        Args:
            content: Raw user content to sanitize
            
        Returns:
            Sanitized content safe for LLM prompts
        """
        ##Action purpose: Use centralized sanitization function
        return sanitize_for_prompt(content)
    
    ##Method purpose: Load graph from persistent storage
    def _load(self) -> None:
        """Load graph data from disk."""
        ##Step purpose: Define default factory for empty graph
        def default_factory() -> dict[str, object]:
            return {"nodes": {}, "edges": []}
        
        ##Action purpose: Load with migration support
        data = load_json_with_migration(
            self._graph_file,
            default_factory=default_factory,
        )
        
        ##Step purpose: Reconstruct nodes
        self._nodes = {}
        ##Loop purpose: Convert node dicts to Node objects
        for node_id, node_data in data.get("nodes", {}).items():
            self._nodes[node_id] = Node.from_dict(node_data)
        
        ##Step purpose: Reconstruct edges
        self._edges = defaultdict(list)
        self._reverse_edges = defaultdict(list)
        ##Loop purpose: Convert edge dicts to Edge objects
        for edge_data in data.get("edges", []):
            edge = Edge.from_dict(edge_data)
            self._edges[edge.source_id].append(edge)
            self._reverse_edges[edge.target_id].append(edge)
    
    ##Method purpose: Save graph to persistent storage
    def _save(self) -> None:
        """Save graph data to disk."""
        ##Step purpose: Build serializable data structure
        data = {
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
            "edges": [
                edge.to_dict()
                for edges in self._edges.values()
                for edge in edges
            ],
        }
        
        ##Action purpose: Save atomically
        save_json_atomic(self._graph_file, data)
    
    ##Method purpose: Add a node to the graph
    def add_node(self, node: Node, persist: bool = True) -> None:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
            persist: Whether to immediately persist to disk (default: True)
        """
        self._nodes[node.id] = node
        ##Condition purpose: Only persist if requested
        if persist:
            self._save()
    
    ##Method purpose: Add multiple nodes with single disk write
    def add_nodes_batch(self, nodes: list[Node]) -> None:
        """
        Add multiple nodes with a single disk write for performance.
        
        Args:
            nodes: List of nodes to add
        """
        ##Loop purpose: Add all nodes to in-memory structure
        for node in nodes:
            self._nodes[node.id] = node
        
        ##Action purpose: Single disk write for all nodes
        self._save()
    
    ##Method purpose: Get a node by ID
    def get_node(self, node_id: str) -> Node:
        """
        Get a node by ID.
        
        Args:
            node_id: ID of node to retrieve
            
        Returns:
            The node
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        ##Condition purpose: Check node exists
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return self._nodes[node_id]
    
    ##Method purpose: Check if node exists
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self._nodes
    
    ##Method purpose: Remove a node and its edges
    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all its edges.
        
        Args:
            node_id: ID of node to remove
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        ##Condition purpose: Check node exists
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        
        ##Action purpose: Remove node
        del self._nodes[node_id]
        
        ##Perf: Use _reverse_edges index to find sources with edges TO this node (O(degree) not O(n))
        ##      This fixes P1-003 from Daedelus audit - was O(n²), now O(degree)
        if node_id in self._reverse_edges:
            ##Step purpose: Get all edges that point TO this node
            incoming_edges = self._reverse_edges[node_id]
            ##Loop purpose: Remove this node from each source's edge list
            for edge in incoming_edges:
                source_id = edge.source_id
                if source_id in self._edges:
                    self._edges[source_id] = [
                        e for e in self._edges[source_id]
                        if e.target_id != node_id
                    ]
            ##Action purpose: Remove the reverse edge index entry
            del self._reverse_edges[node_id]
        
        ##Action purpose: Remove outgoing edges and update reverse index
        if node_id in self._edges:
            ##Loop purpose: Clean up reverse index entries for outgoing edges
            for edge in self._edges[node_id]:
                target_id = edge.target_id
                if target_id in self._reverse_edges:
                    self._reverse_edges[target_id] = [
                        e for e in self._reverse_edges[target_id]
                        if e.source_id != node_id
                    ]
            del self._edges[node_id]
        
        self._save()
    
    ##Method purpose: Add an edge between two nodes
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.RELATES_TO,
        weight: float = 1.0,
        metadata: dict[str, str] | None = None,
    ) -> Edge:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: ID of source node
            target_id: ID of target node
            edge_type: Type of relationship
            weight: Edge weight
            metadata: Optional metadata
            
        Returns:
            The created edge
            
        Raises:
            NodeNotFoundError: If either node doesn't exist
        """
        ##Condition purpose: Validate both nodes exist
        if source_id not in self._nodes:
            raise NodeNotFoundError(source_id)
        if target_id not in self._nodes:
            raise NodeNotFoundError(target_id)
        
        ##Step purpose: Create edge
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        
        ##Action purpose: Add to indexes
        self._edges[source_id].append(edge)
        self._reverse_edges[target_id].append(edge)
        
        ##Action purpose: Persist changes (always persist edges for consistency)
        self._save()
        return edge
    
    ##Method purpose: Get neighbors of a node
    def neighbors(self, node_id: str, direction: str = "out") -> list[Node]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: ID of node to get neighbors for
            direction: 'out' for outgoing edges, 'in' for incoming, 'both' for all
            
        Returns:
            List of neighboring Node objects. Returns empty list if node
            doesn't exist or has no neighbors (no exception raised).
        """
        neighbor_ids: set[str] = set()
        
        ##Condition purpose: Get outgoing neighbors
        if direction in ("out", "both"):
            ##Loop purpose: Collect target nodes
            for edge in self._edges.get(node_id, []):
                neighbor_ids.add(edge.target_id)
        
        ##Condition purpose: Get incoming neighbors
        if direction in ("in", "both"):
            ##Loop purpose: Collect source nodes
            for edge in self._reverse_edges.get(node_id, []):
                neighbor_ids.add(edge.source_id)
        
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]
    
    ##Method purpose: Search graph for matching nodes
    def search(
        self,
        query: str,
        max_results: int = 10,
        node_types: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """
        Search for nodes matching query text.
        
        Performs case-insensitive substring matching on node labels and content.
        Results are scored (label matches count more than content matches)
        and sorted by relevance.
        
        Args:
            query: Text to search for in labels and content
            max_results: Maximum number of results to return
            node_types: Filter by node types (None = all types)
            
        Returns:
            List of dicts with 'id', 'label', 'content' keys, sorted by relevance
        """
        query_lower = query.lower()
        matches: list[tuple[float, Node]] = []
        
        ##Loop purpose: Score each node against query
        for node in self._nodes.values():
            ##Condition purpose: Filter by node type if specified
            if node_types and node.node_type not in node_types:
                continue
            
            ##Step purpose: Simple relevance scoring
            score = 0.0
            ##Condition purpose: Check label match
            if query_lower in node.label.lower():
                score += 2.0
            ##Condition purpose: Check content match
            if query_lower in node.content.lower():
                score += 1.0
            
            ##Condition purpose: Only include if matched
            if score > 0:
                matches.append((score, node))
        
        ##Step purpose: Sort by score and limit
        matches.sort(key=lambda x: x[0], reverse=True)
        matches = matches[:max_results]
        
        ##Step purpose: Convert to dicts
        return [
            {"id": node.id, "label": node.label, "content": node.content}
            for _, node in matches
        ]
    
    ##Method purpose: Get all nodes
    def all_nodes(self) -> list[Node]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())
    
    ##Method purpose: Get node count
    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)
    
    ##Method purpose: Get edge count
    def edge_count(self) -> int:
        """Get total number of edges."""
        return sum(len(edges) for edges in self._edges.values())
    
    ##Method purpose: Get nodes by user (filtering by metadata.user field)
    def get_nodes_by_user(self, username: str) -> list[Node]:
        """
        Get all nodes belonging to a user.
        
        Args:
            username: Username to filter by
            
        Returns:
            List of nodes with matching user metadata
        """
        ##Step purpose: Filter nodes by user metadata
        return [
            node for node in self._nodes.values()
            if node.metadata.get("user") == username
        ]
    
    ##Method purpose: Get connection count for a node (centrality proxy)
    def get_connection_count(self, node_id: str) -> int:
        """
        Get total connection count for a node (in + out edges).
        
        Args:
            node_id: Node ID to check
            
        Returns:
            Total edge count for node
        """
        out_count = len(self._edges.get(node_id, []))
        in_count = len(self._reverse_edges.get(node_id, []))
        return out_count + in_count
    
    ##Method purpose: Analyze emotional content using LLM
    def analyze_emotion(
        self,
        content: str,
        llm: LLMProtocol,
    ) -> EmotionResult:
        """
        Analyze emotional content of text using LLM.
        
        Args:
            content: Text content to analyze
            llm: LLM interface for analysis
            
        Returns:
            EmotionResult with detected emotions
            
        Raises:
            GraphAnalysisError: If emotion analysis fails
        """
        ##Step purpose: Sanitize content to prevent prompt injection
        sanitized_content = self._sanitize_for_prompt(content)
        
        ##Step purpose: Build emotion analysis prompt
        prompt = """Analyze the emotional content of the following text.
Respond with a valid JSON object containing:
- "primary_emotion": one of ["joy", "sadness", "anger", "surprise", "fear", "disgust", "love", "curiosity", "neutral"]
- "confidence": a number between 0.0 and 1.0
- "emotion_scores": an object mapping emotion names to scores (0.0-1.0)

Text: "{content}"

JSON Response:""".format(content=sanitized_content)
        
        system_prompt = "You are an emotion analysis expert. Respond only with valid JSON."
        
        ##Error purpose: Handle LLM errors gracefully
        try:
            response = llm.generate_text(prompt, system_prompt)
            
            ##Step purpose: Parse JSON response with size limits
            ##Sec: Validate LLM output with Pydantic schema (P1-001)
            try:
                ##Step purpose: Extract JSON from response if needed
                try:
                    json_str = extract_json_from_response(response)
                except ValueError:
                    json_str = response
                
                ##Action purpose: Parse with size limits
                data = safe_json_loads(json_str)
                ##Sec: Validate parsed data against schema
                validated = EmotionAnalysisResponse.model_validate(data)
            except (json.JSONDecodeError, JSONSizeError) as e:
                ##Error purpose: Re-raise with context
                raise GraphAnalysisError("emotion", f"Invalid JSON: {e}") from e
            except ValidationError as e:
                ##Sec: Handle schema validation failures with fallback
                logger.warning("emotion_analysis_validation_failed", error=str(e))
                validated = EmotionAnalysisResponse()
            
            ##Step purpose: Construct result from validated data
            ##Condition purpose: Validate emotion value against Emotion enum
            try:
                primary_emotion = Emotion(validated.primary_emotion.lower())
            except ValueError:
                primary_emotion = Emotion.NEUTRAL
            
            ##Sec: Use validated fields from Pydantic model (P1-001)
            confidence = validated.confidence
            emotion_scores = validated.emotion_scores
            
            ##Action purpose: Log successful analysis
            logger.debug(
                "emotion_analysis_complete",
                primary_emotion=primary_emotion.value,
                confidence=confidence,
            )
            
            return EmotionResult(
                primary_emotion=primary_emotion,
                confidence=confidence,
                emotion_scores=emotion_scores,
                content_preview=content[:100],
            )
            
        except GraphAnalysisError:
            raise
        except Exception as e:
            raise GraphAnalysisError("emotion", str(e)) from e
    
    ##Method purpose: Connect isolated nodes to nearest semantic neighbors
    def link_orphans(
        self,
        username: str,
        llm: LLMProtocol,
        max_links_per_orphan: int = 2,
    ) -> int:
        """
        Find orphan nodes (no connections) and link to related nodes using LLM.
        
        Args:
            username: Username to filter nodes
            llm: LLM interface for finding relationships
            max_links_per_orphan: Maximum links to create per orphan
            
        Returns:
            Number of new links created
        """
        ##Step purpose: Find user nodes
        user_nodes = self.get_nodes_by_user(username)
        
        ##Condition purpose: Check minimum node count
        if len(user_nodes) < 2:
            logger.debug("link_orphans_skipped", reason="insufficient_nodes")
            return 0
        
        ##Step purpose: Identify orphan nodes (no connections)
        orphan_nodes: list[Node] = []
        connected_nodes: list[Node] = []
        
        ##Loop purpose: Categorize nodes by connection status
        for node in user_nodes:
            if self.get_connection_count(node.id) == 0:
                orphan_nodes.append(node)
            else:
                connected_nodes.append(node)
        
        ##Condition purpose: Check for orphans to process
        if not orphan_nodes:
            logger.debug("link_orphans_skipped", reason="no_orphans")
            return 0
        
        ##Condition purpose: Need connected nodes as targets
        if not connected_nodes:
            ##Step purpose: If all orphans, connect first two
            if len(orphan_nodes) >= 2:
                self.add_edge(
                    orphan_nodes[0].id,
                    orphan_nodes[1].id,
                    EdgeType.RELATES_TO,
                )
                return 1
            return 0
        
        links_created = 0
        
        ##Step purpose: Pre-compute node summaries once (not in loop)
        node_summaries = "\n".join([
            f"- Node {n.id[:8]}: {n.content[:100]}"
            for n in connected_nodes[:MAX_CONTEXT_NODES]
        ])
        
        ##Loop purpose: Process each orphan node
        for orphan in orphan_nodes[:MAX_ORPHANS_PER_BATCH]:
            ##Step purpose: Sanitize orphan content to prevent prompt injection
            sanitized_orphan = self._sanitize_for_prompt(orphan.content)
            
            prompt = f"""Given an isolated node and a list of connected nodes, identify which nodes are most semantically related.

Isolated Node: "{sanitized_orphan}"

Connected Nodes:
{node_summaries}

Respond with a JSON object:
{{"related_node_ids": ["<8-char-prefix>", ...], "relationship": "relates_to"}}

Relationship types: relates_to, implies, supports, caused_by

JSON Response:"""
            
            ##Error purpose: Handle LLM errors gracefully
            try:
                response = llm.generate_text(
                    prompt,
                    "You are a semantic relationship analyzer. Respond only with valid JSON.",
                )
                
                ##Sec: Parse and validate JSON with Pydantic schema (P1-001)
                try:
                    ##Step purpose: Extract JSON from response if needed
                    try:
                        json_str = extract_json_from_response(response)
                    except ValueError:
                        json_str = response
                    
                    ##Action purpose: Parse with size limits
                    data = safe_json_loads(json_str)
                    ##Sec: Validate with Pydantic schema
                    validated = RelationshipAnalysisResponse.model_validate(data)
                except (json.JSONDecodeError, JSONSizeError) as e:
                    logger.warning(
                        "link_orphan_json_parse_failed",
                        orphan_id=orphan.id[:8],
                        error=str(e),
                    )
                    continue
                except ValidationError as e:
                    logger.warning(
                        "link_orphan_validation_failed",
                        orphan_id=orphan.id[:8],
                        error=str(e),
                    )
                    validated = RelationshipAnalysisResponse()
                
                ##Sec: Use validated fields from Pydantic model (P1-001)
                related_ids = validated.related_node_ids
                relationship = validated.relationship
                
                ##Step purpose: Map relationship to EdgeType
                edge_type_map = {
                    "relates_to": EdgeType.RELATES_TO,
                    "implies": EdgeType.IMPLIES,
                    "supports": EdgeType.SUPPORTS,
                    "caused_by": EdgeType.CAUSED_BY,
                }
                edge_type = edge_type_map.get(relationship, EdgeType.RELATES_TO)
                
                ##Loop purpose: Create links to related nodes
                links_for_orphan = 0
                for related_prefix in related_ids[:max_links_per_orphan]:
                    ##Step purpose: Find full node ID from prefix
                    for node in connected_nodes:
                        if node.id.startswith(related_prefix):
                            ##Action purpose: Create edge
                            self.add_edge(orphan.id, node.id, edge_type)
                            links_created += 1
                            links_for_orphan += 1
                            break
                    
                    ##Condition purpose: Limit links per orphan
                    if links_for_orphan >= max_links_per_orphan:
                        break
                        
            except Exception as e:
                logger.warning(
                    "link_orphan_failed",
                    orphan_id=orphan.id[:8],
                    error=str(e),
                )
                continue
        
        ##Action purpose: Log result
        logger.info(
            "link_orphans_complete",
            orphans_processed=min(len(orphan_nodes), 10),
            links_created=links_created,
        )
        
        return links_created
    
    ##Method purpose: Generate higher-order insights from node patterns
    def generate_meta_insights(
        self,
        username: str,
        llm: LLMProtocol,
        min_cluster_size: int = 3,
    ) -> list[str]:
        """
        Analyze clusters of nodes to generate meta-insights.
        
        Args:
            username: Username to filter nodes
            llm: LLM interface for insight generation
            min_cluster_size: Minimum nodes needed for meta-insight
            
        Returns:
            List of generated meta-insight strings
        """
        ##Step purpose: Get user nodes with connections
        user_nodes = self.get_nodes_by_user(username)
        
        ##Condition purpose: Check minimum node count
        if len(user_nodes) < min_cluster_size:
            return []
        
        ##Step purpose: Find clusters using simple connected components
        clusters = self._find_connected_components(user_nodes)
        
        meta_insights: list[str] = []
        
        ##Loop purpose: Generate insight from each qualifying cluster
        for cluster in clusters:
            ##Condition purpose: Skip small clusters
            if len(cluster) < min_cluster_size:
                continue
            
            ##Step purpose: Get content from cluster nodes and sanitize
            cluster_content_parts: list[str] = []
            for node in cluster[:MAX_CLUSTER_NODES_FOR_INSIGHT]:
                sanitized_node = self._sanitize_for_prompt(node.content)
                cluster_content_parts.append(f"- {sanitized_node[:150]}")
            cluster_content = "\n".join(cluster_content_parts)
            
            prompt = f"""Analyze these related pieces of information and synthesize a single, novel meta-insight.
A meta-insight is a new conclusion or pattern that emerges from combining the information, not just a summary.

Related Information:
{cluster_content}

Generate a concise meta-insight (1-2 sentences) that reveals a non-obvious pattern or conclusion:"""
            
            ##Error purpose: Handle LLM errors gracefully
            try:
                insight = llm.generate_text(
                    prompt,
                    "You are a pattern recognition expert. Generate insightful observations.",
                )
                
                ##Condition purpose: Validate insight quality
                if insight and len(insight.strip()) > 20:
                    meta_insights.append(insight.strip())
                    
            except Exception as e:
                logger.warning(
                    "meta_insight_generation_failed",
                    cluster_size=len(cluster),
                    error=str(e),
                )
                continue
        
        ##Action purpose: Log result
        logger.info(
            "meta_insights_generated",
            clusters_analyzed=len([c for c in clusters if len(c) >= min_cluster_size]),
            insights_generated=len(meta_insights),
        )
        
        return meta_insights
    
    ##Method purpose: Find connected components in user's subgraph
    def _find_connected_components(self, nodes: list[Node]) -> list[list[Node]]:
        """
        Find connected components using BFS.
        
        Args:
            nodes: Nodes to analyze
            
        Returns:
            List of connected component node lists
        """
        node_ids = {n.id for n in nodes}
        node_map = {n.id: n for n in nodes}
        visited: set[str] = set()
        components: list[list[Node]] = []
        
        ##Loop purpose: Find all components via BFS
        for node in nodes:
            ##Condition purpose: Skip already visited
            if node.id in visited:
                continue
            
            ##Step purpose: BFS from this node (use deque for O(1) popleft)
            component: list[Node] = []
            queue: deque[str] = deque([node.id])
            
            while queue:
                current_id = queue.popleft()
                ##Condition purpose: Skip if visited or not in subset
                if current_id in visited or current_id not in node_ids:
                    continue
                
                visited.add(current_id)
                component.append(node_map[current_id])
                
                ##Step purpose: Add neighbors to queue
                for neighbor in self.neighbors(current_id, direction="both"):
                    if neighbor.id not in visited and neighbor.id in node_ids:
                        queue.append(neighbor.id)
            
            ##Condition purpose: Only add non-empty components
            if component:
                components.append(component)
        
        return components
    
    ##Method purpose: Remove stale or isolated nodes
    def prune_graph(
        self,
        max_age_days: int = 30,
        min_connections: int = 1,
        username: str | None = None,
    ) -> int:
        """
        Remove nodes that are old and have few connections.
        
        Args:
            max_age_days: Maximum node age in days
            min_connections: Minimum connections to keep node
            username: Optional username filter
            
        Returns:
            Number of nodes pruned
            
        Raises:
            GraphPruneError: If pruning fails
        """
        ##Step purpose: Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        nodes_to_prune: list[str] = []
        
        ##Error purpose: Handle pruning errors
        try:
            ##Loop purpose: Identify nodes to prune
            for node_id, node in self._nodes.items():
                ##Condition purpose: Filter by username if specified
                if username and node.metadata.get("user") != username:
                    continue
                
                ##Step purpose: Parse node creation date
                try:
                    created_at = datetime.fromisoformat(node.created_at)
                except (ValueError, TypeError):
                    ##Step purpose: Skip nodes with invalid dates
                    continue
                
                ##Condition purpose: Check age and connection thresholds
                is_old = created_at < cutoff_date
                connection_count = self.get_connection_count(node_id)
                is_isolated = connection_count < min_connections
                
                if is_old and is_isolated:
                    nodes_to_prune.append(node_id)
            
            ##Condition purpose: Check if anything to prune
            if not nodes_to_prune:
                logger.debug("prune_graph_skipped", reason="no_qualifying_nodes")
                return 0
            
            ##Loop purpose: Remove nodes
            for node_id in nodes_to_prune:
                ##Step purpose: Remove node (this also cleans edges)
                del self._nodes[node_id]
                
                ##Perf: Use _reverse_edges index to find sources with edges TO this node (O(degree) not O(n))
                ##      This fixes P1-003 from Daedelus audit - was O(n²), now O(degree)
                if node_id in self._reverse_edges:
                    ##Loop purpose: Remove this node from each source's edge list
                    for edge in self._reverse_edges[node_id]:
                        source_id = edge.source_id
                        if source_id in self._edges:
                            self._edges[source_id] = [
                                e for e in self._edges[source_id]
                                if e.target_id != node_id
                            ]
                    del self._reverse_edges[node_id]
                
                ##Step purpose: Remove outgoing edges and update reverse index
                if node_id in self._edges:
                    ##Loop purpose: Clean up reverse index entries for outgoing edges
                    for edge in self._edges[node_id]:
                        target_id = edge.target_id
                        if target_id in self._reverse_edges:
                            self._reverse_edges[target_id] = [
                                e for e in self._reverse_edges[target_id]
                                if e.source_id != node_id
                            ]
                    del self._edges[node_id]
            
            ##Action purpose: Persist changes
            self._save()
            
            ##Action purpose: Log result
            logger.info(
                "prune_graph_complete",
                nodes_pruned=len(nodes_to_prune),
                max_age_days=max_age_days,
                min_connections=min_connections,
            )
            
            return len(nodes_to_prune)
            
        except Exception as e:
            raise GraphPruneError(len(nodes_to_prune), str(e)) from e
    
    ##Method purpose: Group related nodes into clusters
    def cluster_nodes(
        self,
        username: str,
        llm: LLMProtocol | None = None,
    ) -> dict[str, list[str]]:
        """
        Cluster user's nodes by semantic similarity.
        
        Uses connected components as base clusters. If LLM provided,
        attempts to label clusters meaningfully.
        
        Args:
            username: Username to filter nodes
            llm: Optional LLM for cluster labeling
            
        Returns:
            Dict mapping cluster label to list of node IDs
        """
        ##Step purpose: Get user nodes
        user_nodes = self.get_nodes_by_user(username)
        
        ##Condition purpose: Handle empty case
        if not user_nodes:
            return {}
        
        ##Step purpose: Find connected components
        components = self._find_connected_components(user_nodes)
        
        clusters: dict[str, list[str]] = {}
        
        ##Loop purpose: Build cluster dict with labels
        for i, component in enumerate(components):
            ##Step purpose: Generate cluster label
            if llm and len(component) >= 2:
                ##Step purpose: Use LLM for meaningful label (sanitize content)
                content_parts: list[str] = []
                for n in component[:MAX_CLUSTER_NODES_FOR_LABEL]:
                    sanitized = self._sanitize_for_prompt(n.content)
                    content_parts.append(sanitized[:50])
                content_sample = " | ".join(content_parts)
                prompt = f"Generate a 2-3 word label for this cluster of related content: {content_sample}"
                
                ##Error purpose: Fall back to default label on error
                try:
                    label = llm.generate_text(
                        prompt,
                        "You are a categorization expert. Respond with just the label.",
                    ).strip()[:50]
                except (GraphAnalysisError, LLMError) as e:
                    logger.debug("label_extraction_failed", error=str(e))
                    label = f"cluster_{i + 1}"
                except Exception as e:
                    ##Error purpose: Log unexpected errors but don't crash
                    logger.warning("unexpected_label_extraction_error", error=str(e), exc_info=True)
                    label = f"cluster_{i + 1}"
            else:
                label = f"cluster_{i + 1}"
            
            ##Step purpose: Add cluster
            clusters[label] = [n.id for n in component]
        
        ##Action purpose: Log result
        logger.debug(
            "cluster_nodes_complete",
            username=username,
            cluster_count=len(clusters),
        )
        
        return clusters
    
    ##Method purpose: Find potentially contradicting nodes
    def find_contradictions(
        self,
        username: str,
        llm: LLMProtocol,
        max_comparisons: int = MAX_CONTRADICTION_COMPARISONS,
    ) -> list[tuple[str, str]]:
        """
        Find pairs of nodes that may contain contradictory information.
        
        Args:
            username: Username to filter nodes
            llm: LLM interface for contradiction detection
            max_comparisons: Maximum number of LLM calls to make (prevents O(n²) explosion)
            
        Returns:
            List of (node_id_a, node_id_b) tuples for contradicting pairs
        """
        ##Step purpose: Get user nodes
        user_nodes = self.get_nodes_by_user(username)
        
        ##Condition purpose: Need at least 2 nodes
        if len(user_nodes) < 2:
            return []
        
        contradictions: list[tuple[str, str]] = []
        
        ##Step purpose: Limit nodes to check to prevent O(n²) explosion
        max_nodes = min(MAX_CONTRADICTION_NODES, len(user_nodes))
        nodes_to_check = user_nodes[:max_nodes]
        
        ##Step purpose: Calculate maximum pairs to check
        max_pairs = len(nodes_to_check) * (len(nodes_to_check) - 1) // 2
        comparisons_made = 0
        
        ##Loop purpose: Check pairs for contradictions (with limit)
        for i, node_a in enumerate(nodes_to_check):
            ##Condition purpose: Stop if we've hit the comparison limit
            if comparisons_made >= max_comparisons:
                logger.debug(
                    "contradiction_check_limit_reached",
                    comparisons_made=comparisons_made,
                    max_comparisons=max_comparisons,
                )
                break
            
            for node_b in nodes_to_check[i + 1:]:
                ##Condition purpose: Stop if we've hit the comparison limit
                if comparisons_made >= max_comparisons:
                    break
                
                ##Step purpose: Sanitize node content to prevent prompt injection
                sanitized_a = self._sanitize_for_prompt(node_a.content)
                sanitized_b = self._sanitize_for_prompt(node_b.content)
                
                ##Step purpose: Build contradiction check prompt
                prompt = f"""Determine if these two statements contradict each other.

Statement A: "{sanitized_a}"
Statement B: "{sanitized_b}"

Respond with JSON: {{"contradicts": true/false, "explanation": "brief explanation"}}

JSON Response:"""
                
                ##Error purpose: Handle LLM errors gracefully
                try:
                    response = llm.generate_text(
                        prompt,
                        "You are a logical consistency analyzer. Respond only with valid JSON.",
                    )
                    
                    ##Sec: Parse and validate JSON with Pydantic schema (P1-001)
                    try:
                        ##Step purpose: Extract JSON from response if needed
                        try:
                            json_str = extract_json_from_response(response)
                        except ValueError:
                            json_str = response
                        
                        ##Action purpose: Parse with size limits
                        data = safe_json_loads(json_str)
                        ##Sec: Validate with Pydantic schema
                        validated = ContradictionCheckResponse.model_validate(data)
                    except (json.JSONDecodeError, JSONSizeError) as e:
                        logger.warning(
                            "contradiction_check_json_parse_failed",
                            node_a=node_a.id[:8],
                            node_b=node_b.id[:8],
                            error=str(e),
                        )
                        comparisons_made += 1
                        continue
                    except ValidationError as e:
                        logger.warning(
                            "contradiction_check_validation_failed",
                            node_a=node_a.id[:8],
                            node_b=node_b.id[:8],
                            error=str(e),
                        )
                        comparisons_made += 1
                        continue
                    
                    ##Sec: Use validated field from Pydantic model (P1-001)
                    if validated.contradicts:
                        contradictions.append((node_a.id, node_b.id))
                    
                    comparisons_made += 1
                        
                except Exception as e:
                    logger.warning(
                        "contradiction_check_failed",
                        node_a=node_a.id[:8],
                        node_b=node_b.id[:8],
                        error=str(e),
                    )
                    comparisons_made += 1
                    continue
        
        ##Action purpose: Log result
        logger.info(
            "find_contradictions_complete",
            pairs_checked=comparisons_made,
            max_pairs_available=max_pairs,
            contradictions_found=len(contradictions),
        )
        
        return contradictions
    
    ##Method purpose: Suggest new connections for a node
    def suggest_connections(
        self,
        node_id: str,
        llm: LLMProtocol,
        max_suggestions: int = 5,
    ) -> list[str]:
        """
        Suggest potential connections for a node based on content similarity.
        
        Args:
            node_id: Node to find connections for
            llm: LLM interface for similarity analysis
            max_suggestions: Maximum suggestions to return
            
        Returns:
            List of suggested target node IDs
            
        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        ##Step purpose: Get source node
        source_node = self.get_node(node_id)
        
        ##Step purpose: Get existing connections
        existing_connections = {n.id for n in self.neighbors(node_id, "both")}
        existing_connections.add(node_id)  # Don't suggest self-connection
        
        ##Step purpose: Find candidate nodes (not already connected)
        candidates = [
            n for n in self._nodes.values()
            if n.id not in existing_connections
        ]
        
        ##Condition purpose: Handle no candidates
        if not candidates:
            return []
        
        ##Step purpose: Build suggestion prompt (sanitize content)
        sanitized_source = self._sanitize_for_prompt(source_node.content)
        candidate_summaries_parts: list[str] = []
        for n in candidates[:MAX_CANDIDATE_NODES]:
            sanitized_candidate = self._sanitize_for_prompt(n.content)
            candidate_summaries_parts.append(f"- {n.id[:8]}: {sanitized_candidate[:80]}")
        candidate_summaries = "\n".join(candidate_summaries_parts)
        
        prompt = f"""Given a source node and candidate nodes, identify which candidates are most semantically related to the source.

Source Node: "{sanitized_source}"

Candidate Nodes:
{candidate_summaries}

Respond with JSON: {{"suggested_ids": ["id1", "id2", ...]}} containing the 8-character ID prefixes of the most related nodes (up to {max_suggestions}).

JSON Response:"""
        
        ##Error purpose: Handle LLM errors gracefully
        try:
            response = llm.generate_text(
                prompt,
                "You are a semantic relationship analyzer. Respond only with valid JSON.",
            )
            
            ##Step purpose: Parse JSON with size limits (no regex fallback to prevent ReDoS)
            try:
                ##Step purpose: Extract JSON from response if needed
                try:
                    json_str = extract_json_from_response(response)
                except ValueError:
                    json_str = response
                
                ##Action purpose: Parse with size limits
                data = safe_json_loads(json_str)
                
                ##Sec: Validate LLM output with Pydantic schema (P1-001)
                validated = ConnectionSuggestionsResponse.model_validate(data)
            except ValidationError as e:
                logger.warning(
                    "suggest_connections_validation_failed",
                    node_id=node_id[:8],
                    error=str(e),
                )
                validated = ConnectionSuggestionsResponse()
            except (json.JSONDecodeError, JSONSizeError) as e:
                logger.warning(
                    "suggest_connections_json_parse_failed",
                    node_id=node_id[:8],
                    error=str(e),
                )
                return []
            
            suggested_prefixes = validated.suggested_ids
            
            ##Step purpose: Map prefixes to full IDs
            suggestions: list[str] = []
            for prefix in suggested_prefixes[:max_suggestions]:
                for candidate in candidates:
                    if candidate.id.startswith(prefix):
                        suggestions.append(candidate.id)
                        break
            
            ##Action purpose: Log result
            logger.debug(
                "suggest_connections_complete",
                node_id=node_id[:8],
                suggestions_count=len(suggestions),
            )
            
            return suggestions
            
        except Exception as e:
            logger.warning(
                "suggest_connections_failed",
                node_id=node_id[:8],
                error=str(e),
            )
            return []
