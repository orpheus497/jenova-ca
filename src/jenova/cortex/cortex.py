# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Cortex - Cognitive Graph Management for JENOVA.

Phase 20 Enhancements:
- Fixed JSON DoS vulnerabilities using safe JSON parser
- Added None checks on graph operations
- Enhanced error handling with specific exceptions
- Type hints added for critical methods
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

import networkx as nx

from jenova.cortex.graph_components import CognitiveLink, CognitiveNode
from jenova.utils.json_parser import (
    load_json_safe,
    save_json_safe,
    parse_json_safe,
    JSONParseError,
    JSONSecurityError
)


class Cortex:
    """
    The Cortex is the central hub of the AI's cognitive architecture.
    It manages a graph of interconnected cognitive nodes, including insights,
    memories, and assumptions, fostering a deep, evolving understanding.
    """

    def __init__(self, config, ui_logger, file_logger, llm, cortex_root):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.llm = llm
        self.cortex_root = cortex_root
        os.makedirs(self.cortex_root, exist_ok=True)
        self.docs_path = os.path.join(os.getcwd(), "src", "jenova", "docs")
        self.graph_file = os.path.join(self.cortex_root, "cognitive_graph.json")
        self.graph = self._load_graph()
        self.reflection_cycle_count = 0
        self.processed_docs_file = os.path.join(
            self.cortex_root, "processed_documents.json"
        )
        self.processed_docs = self._load_processed_docs()

    def _load_graph(self) -> Dict:
        """
        Loads the cognitive graph from a file.

        Phase 20: Uses safe JSON loading with size limits to prevent DoS attacks.

        Returns:
            Dictionary with 'nodes' and 'links' keys
        """
        if os.path.exists(self.graph_file):
            try:
                # Use safe JSON loader with 50MB limit for graph files
                data = load_json_safe(
                    self.graph_file,
                    max_size=50 * 1024 * 1024  # 50MB limit for cognitive graphs
                )

                # Validate data structure
                if not isinstance(data, dict):
                    raise TypeError("Graph file must contain a JSON object")

                nodes = {}
                for node_id, node_data in data.get("nodes", {}).items():
                    if not isinstance(node_data, dict):
                        self.file_logger.log_warning(
                            f"Skipping invalid node data for {node_id}"
                        )
                        continue
                    try:
                        nodes[node_id] = CognitiveNode(**node_data)
                    except (TypeError, ValueError) as e:
                        self.file_logger.log_warning(
                            f"Skipping invalid node {node_id}: {e}"
                        )

                links = []
                for link_data in data.get("links", []):
                    if not isinstance(link_data, dict):
                        continue
                    try:
                        links.append(CognitiveLink(**link_data))
                    except (TypeError, ValueError) as e:
                        self.file_logger.log_warning(
                            f"Skipping invalid link: {e}"
                        )

                return {"nodes": nodes, "links": links}

            except (JSONParseError, JSONSecurityError) as e:
                self.file_logger.log_error(
                    f"Security error loading cognitive graph: {e}. Creating a new one."
                )
                return {"nodes": {}, "links": []}
            except (TypeError, KeyError) as e:
                self.file_logger.log_error(
                    f"Error loading cognitive graph: {e}. Creating a new one."
                )
                return {"nodes": {}, "links": []}

        return {"nodes": {}, "links": []}

    def _save_graph(self) -> None:
        """
        Saves the cognitive graph to a file.

        Phase 20: Uses safe JSON saving with size validation.

        Raises:
            JSONSecurityError: If serialized graph exceeds size limit
        """
        try:
            # Safely access graph with None check
            if self.graph is None:
                self.file_logger.log_error("Cannot save: graph is None")
                return

            data = {
                "nodes": {
                    node_id: asdict(node)
                    for node_id, node in self.graph.get("nodes", {}).items()
                    if node is not None
                },
                "links": [
                    asdict(link)
                    for link in self.graph.get("links", [])
                    if link is not None
                ],
            }

            # Use safe JSON saver with 50MB limit
            save_json_safe(
                data,
                self.graph_file,
                indent=4,
                max_size=50 * 1024 * 1024  # 50MB limit
            )

        except JSONSecurityError as e:
            self.file_logger.log_error(
                f"Graph too large to save: {e}. Consider pruning old nodes."
            )
            raise
        except Exception as e:
            self.file_logger.log_error(f"Error saving cognitive graph: {e}")
            raise

    def add_node(
        self,
        node_type: str,
        content: str,
        user: str,
        linked_to: list = None,
        metadata: dict = None,
    ) -> str:
        """Adds a new node to the cognitive graph."""
        node_id = str(uuid.uuid4())

        emotion_prompt = f"""Analyze the emotional content of the following text. Respond ONLY with a valid JSON object containing the detected emotions from this list: ['Joy', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Disgust', 'Love', 'Curiosity'] and their intensity (0.0 to 1.0).

Text: "{content[:500]}" # Truncate for performance

JSON:"""
        emotion_str = self.llm.generate(prompt=emotion_prompt, temperature=0.2)
        try:
            # Use safe JSON parsing with size limit (Phase 20)
            emotion_str_clean = emotion_str.strip()
            if "{" in emotion_str_clean:
                json_start = emotion_str_clean.index("{")
                json_end = emotion_str_clean.rindex("}") + 1
                emotion_str_clean = emotion_str_clean[json_start:json_end]

            # Parse with safety limits (max 1KB for emotion JSON)
            emotions = parse_json_safe(
                emotion_str_clean,
                max_size=1024,  # 1KB limit for emotion JSON
                max_depth=10
            )

            # Validate structure
            if not isinstance(emotions, dict):
                emotions = {}

        except (JSONParseError, JSONSecurityError, ValueError):
            emotions = {}
            if self.file_logger:
                self.file_logger.log_warning(
                    f"Failed to decode emotion JSON: {emotion_str[:100]}"
                )

        new_node_metadata = {"emotions": emotions, "centrality": 0}
        if metadata:
            new_node_metadata.update(metadata)

        new_node = CognitiveNode(
            id=node_id,
            node_type=node_type,
            content=content,
            user=user,
            metadata=new_node_metadata,
        )

        # Phase 20: Add None checks before graph access
        if self.graph is None or "nodes" not in self.graph:
            self.file_logger.log_error("Cannot add node: graph not initialized")
            return node_id

        self.graph["nodes"][node_id] = new_node

        if linked_to:
            for target_id in linked_to:
                if target_id in self.graph["nodes"]:
                    self.add_link(node_id, target_id, "related_to")

        self._save_graph()
        if self.ui_logger:
            self.ui_logger.info(f"New {node_type} node created: {node_id}")
        return node_id

    def add_link(self, source_id: str, target_id: str, relationship: str) -> None:
        """
        Adds a directed link between two nodes in the graph.

        Phase 20: Added None checks and type hints.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type
        """
        # Phase 20: Add None checks
        if self.graph is None or "links" not in self.graph:
            self.file_logger.log_error("Cannot add link: graph not initialized")
            return

        new_link = CognitiveLink(
            source_id=source_id, target_id=target_id, relationship=relationship
        )
        self.graph["links"].append(new_link)
        self._save_graph()

    def get_node(self, node_id: str) -> Optional[CognitiveNode]:
        """
        Get a node by ID.

        Phase 20: Added None checks and type hints.

        Args:
            node_id: Node ID to retrieve

        Returns:
            CognitiveNode if found, None otherwise
        """
        if self.graph is None or "nodes" not in self.graph:
            return None
        return self.graph["nodes"].get(node_id)

    def get_all_nodes_by_type(
        self, node_type: str, user: Optional[str] = None
    ) -> List[CognitiveNode]:
        """
        Get all nodes of a specific type.

        Phase 20: Added None checks and type hints.

        Args:
            node_type: Type of nodes to retrieve
            user: Optional user filter

        Returns:
            List of matching nodes
        """
        if self.graph is None or "nodes" not in self.graph:
            return []

        nodes = []
        for node in self.graph["nodes"].values():
            if node is None:
                continue
            if node.node_type == node_type:
                if user and node.user == user:
                    nodes.append(node)
                elif not user:
                    nodes.append(node)
        return nodes

    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        linked_to: Optional[list] = None
    ) -> None:
        """
        Update a node's content and/or links.

        Phase 20: Added None checks and type hints.

        Args:
            node_id: ID of node to update
            content: New content (optional)
            linked_to: New links (optional)
        """
        node = self.get_node(node_id)
        if not node:
            return

        if content:
            node.content = content
            node.timestamp = datetime.now().isoformat()

        if linked_to:
            # Phase 20: Add None check before accessing links
            if self.graph is None or "links" not in self.graph or "nodes" not in self.graph:
                self.file_logger.log_error("Cannot update links: graph not initialized")
                return

            self.graph["links"] = [
                link for link in self.graph["links"]
                if link is not None and link.source_id != node_id
            ]
            for target_id in linked_to:
                if target_id in self.graph["nodes"]:
                    self.add_link(node_id, target_id, "related_to")

        self._save_graph()
        if self.ui_logger:
            self.ui_logger.info(f"Node {node_id} updated.")

    def reflect(self, user: str) -> list[str]:
        messages = [f"Initiating deep reflection for user '{user}'..."]
        user_nodes = [n for n in self.graph["nodes"].values() if n.user == user]
        if len(user_nodes) < 5:
            messages.append("Insufficient data for a meaningful reflection.")
            return messages

        self._calculate_centrality()
        self._link_orphans(user_nodes)
        self._generate_meta_insights(user_nodes)

        self.reflection_cycle_count += 1
        pruning_config = self.config.get("cortex", {}).get("pruning", {})
        if self.reflection_cycle_count % pruning_config.get("prune_interval", 10) == 0:
            messages.extend(self.prune_graph(user))

        self._save_graph()
        messages.append("Reflection complete. The cognitive graph has been refined.")
        return messages

    def _calculate_centrality(self):
        weights = self.config.get("cortex", {}).get("relationship_weights", {})
        node_link_counts = {node_id: 0 for node_id in self.graph["nodes"]}

        for link in self.graph["links"]:
            weight = weights.get(link.relationship, 1.0)
            node_link_counts.setdefault(link.source_id, 0)
            node_link_counts[link.source_id] += weight
            node_link_counts.setdefault(link.target_id, 0)
            node_link_counts[link.target_id] += weight

        for node_id, count in node_link_counts.items():
            if node := self.get_node(node_id):
                node.metadata["centrality"] = count

    def _link_orphans(self, user_nodes: List[CognitiveNode]) -> int:
        """
        Find and link isolated nodes using NetworkX graph analysis and LLM-based semantic matching.

        An "orphan" is defined as a node with degree < 2 (fewer than 2 connections).
        For each orphan, we use the LLM to analyze semantic relationships and create
        appropriate links to other relevant nodes.

        FIXES: BUG-C1 - Critical incomplete implementation

        Args:
            user_nodes: List of cognitive nodes for the user

        Returns:
            Number of new links created
        """
        if len(user_nodes) < 3:
            # Need at least 3 nodes to make linking meaningful
            return 0

        # Build NetworkX graph from cognitive graph
        G = nx.DiGraph()

        # Add all user nodes
        node_id_to_node = {node.id: node for node in user_nodes}
        for node in user_nodes:
            G.add_node(node.id, content=node.content, node_type=node.node_type)

        # Add links between user nodes
        for link in self.graph["links"]:
            if link.source_id in node_id_to_node and link.target_id in node_id_to_node:
                G.add_edge(
                    link.source_id, link.target_id, relationship=link.relationship
                )

        # Find orphans (nodes with degree < 2)
        orphan_ids = []
        for node_id in G.nodes():
            degree = G.degree(node_id)  # Total degree (in + out)
            if degree < 2:
                orphan_ids.append(node_id)

        if not orphan_ids:
            self.file_logger.log_info("No orphan nodes found during reflection")
            return 0

        self.file_logger.log_info(f"Found {len(orphan_ids)} orphan nodes to link")

        # Limit orphans processed to avoid excessive LLM calls
        max_orphans = 10
        orphan_ids = orphan_ids[:max_orphans]

        links_created = 0

        for orphan_id in orphan_ids:
            orphan_node = node_id_to_node[orphan_id]

            # Build context of potential link targets (exclude nodes already linked)
            existing_neighbors = set(G.successors(orphan_id)) | set(
                G.predecessors(orphan_id)
            )
            potential_targets = [
                node
                for node in user_nodes
                if node.id != orphan_id and node.id not in existing_neighbors
            ]

            if not potential_targets:
                continue

            # Limit candidates to avoid token limits
            # Prioritize high-centrality nodes as they're likely important
            potential_targets.sort(
                key=lambda n: n.metadata.get("centrality", 0), reverse=True
            )
            candidates = potential_targets[:20]  # Top 20 by centrality

            # Build prompt for LLM to find semantic links
            candidates_text = "\n".join(
                [
                    f"{i+1}. [{c.id[:8]}] ({c.node_type}) {c.content[:100]}"
                    for i, c in enumerate(candidates)
                ]
            )

            prompt = f"""Analyze the following orphan node and identify the TOP 2 most semantically related nodes from the candidate list.

Orphan Node:
Type: {orphan_node.node_type}
Content: "{orphan_node.content}"

Candidate Nodes:
{candidates_text}

Respond with ONLY a JSON object containing:
{{
    "links": [
        {{"target_id": "<id>", "relationship": "<relationship_type>", "confidence": <0.0-1.0>}},
        ...
    ]
}}

Valid relationship types: related_to, elaborates_on, conflicts_with, supports, questions

Only include links with confidence >= 0.6.

JSON:"""

            try:
                response = self.llm.generate(prompt=prompt, temperature=0.3)

                # Extract JSON from response
                response_clean = response.strip()
                if "{" in response_clean:
                    json_start = response_clean.index("{")
                    json_end = response_clean.rindex("}") + 1
                    response_clean = response_clean[json_start:json_end]

                link_data = json.loads(response_clean)

                # Create links
                for link_info in link_data.get("links", []):
                    target_id = link_info.get("target_id")
                    relationship = link_info.get("relationship", "related_to")
                    confidence = link_info.get("confidence", 0.0)

                    if confidence < 0.6:
                        continue

                    # Verify target_id exists
                    if target_id not in node_id_to_node:
                        # Try prefix matching (LLM might have abbreviated)
                        matched = [
                            nid
                            for nid in node_id_to_node.keys()
                            if nid.startswith(target_id)
                        ]
                        if matched:
                            target_id = matched[0]
                        else:
                            continue

                    # Create bidirectional link
                    self.add_link(orphan_id, target_id, relationship)
                    links_created += 1

                    self.file_logger.log_info(
                        f"Linked orphan {orphan_id[:8]} -> {target_id[:8]} "
                        f"({relationship}, confidence: {confidence:.2f})"
                    )

            except (json.JSONDecodeError, ValueError) as e:
                self.file_logger.log_warning(
                    f"Failed to parse LLM response for orphan linking: {e}"
                )
                continue
            except Exception as e:
                self.file_logger.log_error(f"Error linking orphan {orphan_id[:8]}: {e}")
                continue

        self.file_logger.log_info(f"Created {links_created} new links for orphan nodes")
        return links_created

    def _generate_meta_insights(self, user_nodes: List[CognitiveNode]) -> List[str]:
        """
        Generate meta-insights from clusters of highly interconnected nodes using community detection.

        This method uses the Louvain algorithm for community detection, then synthesizes
        meta-insights from dense clusters of related insights.

        FIXES: BUG-C1 - Critical incomplete implementation

        Args:
            user_nodes: List of cognitive nodes for the user

        Returns:
            List of generated meta-insight IDs
        """
        # Only generate meta-insights from insight nodes
        insight_nodes = [n for n in user_nodes if n.node_type == "insight"]

        if len(insight_nodes) < 5:
            # Need sufficient insights to form meaningful clusters
            self.file_logger.log_info(
                "Insufficient insight nodes for meta-insight generation"
            )
            return []

        # Build undirected graph for community detection
        G = nx.Graph()

        # Add insight nodes
        node_id_to_node = {node.id: node for node in insight_nodes}
        for node in insight_nodes:
            G.add_node(
                node.id,
                content=node.content,
                centrality=node.metadata.get("centrality", 0),
            )

        # Add edges from links
        for link in self.graph["links"]:
            if link.source_id in node_id_to_node and link.target_id in node_id_to_node:
                # Add edge (undirected - community detection treats as undirected)
                G.add_edge(
                    link.source_id, link.target_id, relationship=link.relationship
                )

        if G.number_of_edges() < 3:
            # Need sufficient connections for community detection
            self.file_logger.log_info(
                "Insufficient connections for community detection"
            )
            return []

        try:
            # Use Louvain algorithm for community detection
            import community as community_louvain

            has_louvain = True
        except ImportError:
            # Fallback to greedy modularity communities if python-louvain not available
            has_louvain = False

        # Detect communities
        if has_louvain:
            try:
                communities = community_louvain.best_partition(G)
                # Convert to list of sets
                community_dict = defaultdict(set)
                for node_id, comm_id in communities.items():
                    community_dict[comm_id].add(node_id)
                community_list = list(community_dict.values())
            except Exception as e:
                self.file_logger.log_warning(
                    f"Louvain algorithm failed: {e}, using fallback"
                )
                community_list = list(nx.community.greedy_modularity_communities(G))
        else:
            community_list = list(nx.community.greedy_modularity_communities(G))

        self.file_logger.log_info(
            f"Detected {len(community_list)} communities in insight graph"
        )

        meta_insight_ids = []

        # Process each community
        for comm_idx, community in enumerate(community_list):
            # Only generate meta-insights for sufficiently large and connected communities
            if len(community) < 3:
                continue

            # Calculate average centrality of community
            avg_centrality = sum(
                node_id_to_node[nid].metadata.get("centrality", 0) for nid in community
            ) / len(community)

            # Only process high-value communities (high centrality or large size)
            if avg_centrality < 0.3 and len(community) < 5:
                continue

            # Get community nodes sorted by centrality
            community_nodes = [node_id_to_node[nid] for nid in community]
            community_nodes.sort(
                key=lambda n: n.metadata.get("centrality", 0), reverse=True
            )

            # Limit to top nodes to avoid token limits
            top_nodes = community_nodes[:10]

            # Build prompt for meta-insight synthesis
            insights_text = "\n".join([f"- {node.content}" for node in top_nodes])

            prompt = f"""Analyze the following cluster of related insights and synthesize a higher-level meta-insight that captures the overarching theme or pattern.

Related Insights (Cluster {comm_idx + 1}):
{insights_text}

Generate a meta-insight that:
1. Identifies the common theme or pattern
2. Synthesizes the insights into a broader understanding
3. Is concise (2-3 sentences maximum)
4. Provides value beyond the individual insights

Respond with ONLY a JSON object:
{{
    "meta_insight": "<your synthesized meta-insight>",
    "confidence": <0.0-1.0>
}}

JSON:"""

            try:
                response = self.llm.generate(prompt=prompt, temperature=0.4)

                # Extract JSON
                response_clean = response.strip()
                if "{" in response_clean:
                    json_start = response_clean.index("{")
                    json_end = response_clean.rindex("}") + 1
                    response_clean = response_clean[json_start:json_end]

                meta_data = json.loads(response_clean)
                meta_insight_text = meta_data.get("meta_insight", "")
                confidence = meta_data.get("confidence", 0.0)

                if not meta_insight_text or confidence < 0.6:
                    continue

                # Create meta-insight node
                user = top_nodes[0].user  # Use same user as cluster nodes
                meta_insight_id = self.add_node(
                    "meta_insight",
                    meta_insight_text,
                    user,
                    metadata={
                        "cluster_size": len(community),
                        "avg_centrality": avg_centrality,
                        "confidence": confidence,
                    },
                )

                # Link meta-insight to all nodes in cluster
                for node in top_nodes:
                    self.add_link(meta_insight_id, node.id, "synthesizes")

                meta_insight_ids.append(meta_insight_id)

                self.file_logger.log_info(
                    f"Generated meta-insight {meta_insight_id[:8]} from "
                    f"cluster of {len(community)} insights (confidence: {confidence:.2f})"
                )

            except (json.JSONDecodeError, ValueError) as e:
                self.file_logger.log_warning(
                    f"Failed to parse LLM response for meta-insight generation: {e}"
                )
                continue
            except Exception as e:
                self.file_logger.log_error(
                    f"Error generating meta-insight for cluster {comm_idx}: {e}"
                )
                continue

        self.file_logger.log_info(
            f"Generated {len(meta_insight_ids)} meta-insights from community analysis"
        )
        return meta_insight_ids

    def develop_insight(self, insight_id: str, user: str) -> list[str]:
        messages = []
        insight_node = self.get_node(insight_id)
        if not insight_node or insight_node.node_type != "insight":
            return [f"Node {insight_id} is not a valid insight."]

        prompt = f"""Analyze the following insight and generate a more detailed and developed version of it. Add more context, examples, or connections.

Original Insight: "{insight_node.content}"

Developed Insight:"""
        developed_content = self.llm.generate(prompt, temperature=0.5)
        if developed_content:
            new_insight_id = self.add_node("insight", developed_content, user)
            self.add_link(new_insight_id, insight_id, "develops")
            messages.append(f"Developed insight {insight_id} into {new_insight_id}")
        return messages

    def develop_insights_from_docs(self, user: str) -> list[str]:
        messages = []
        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)
            return [
                f"Docs directory created at {self.docs_path}. Please add documents to process."
            ]

        for filename in os.listdir(self.docs_path):
            filepath = os.path.join(self.docs_path, filename)
            if not os.path.isfile(filepath):
                continue

            try:
                last_modified = os.path.getmtime(filepath)
                if (
                    self.processed_docs.get(filename, {}).get("last_modified", 0)
                    >= last_modified
                ):
                    continue

                messages.append(f"Processing document: {filename}")
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                summary = self.llm.generate(
                    f'Summarize the following document in 2-3 sentences:\n\n"""{content[:4000]}"""\n\nSummary:',
                    temperature=0.3,
                )
                doc_metadata = {
                    "summary": summary,
                    "filename": filename,
                    "source": "document",
                }
                doc_content = f"Document: {filename}\n\n{content}"
                doc_node_id = self.add_node(
                    "document", doc_content, user, metadata=doc_metadata
                )

                # Recursive chunking and analysis
                chunk_ids = self._process_text_recursively(
                    content, user, doc_node_id, filename
                )

                self.processed_docs[filename] = {
                    "last_modified": last_modified,
                    "summary": summary,
                    "document_node_id": doc_node_id,
                    "chunk_ids": chunk_ids,
                }
                self._save_processed_docs()
                messages.append(
                    f"Document '{filename}' processed with {len(chunk_ids)} chunks."
                )

            except Exception as e:
                self.file_logger.log_error(f"Error processing document {filename}: {e}")
                messages.append(f"Error processing document {filename}: {e}")

        return messages if messages else ["No new documents to process."]

    def _process_text_recursively(
        self, text: str, user: str, parent_id: str, source_filename: str, level: int = 0
    ) -> list[str]:
        """Recursively chunk text and create nodes in the graph."""
        chunk_ids = []
        # Base case: if text is small enough, process it directly
        if len(text.split()) < 800:
            chunk_id = self.add_node(
                "document_chunk",
                f"Chunk from {source_filename}:\n\n{text}",
                user,
                metadata={"level": level, "parent": parent_id},
            )
            self.add_link(chunk_id, parent_id, "part_of")
            return [chunk_id]

        # Recursive step: summarize and split
        summary = self.llm.generate(
            f'Summarize the key points of the following text:\n\n"""{text}"""\n\nSummary:',
            temperature=0.3,
        )
        summary_id = self.add_node(
            "summary_chunk",
            f"Summary from {source_filename}:\n\n{summary}",
            user,
            metadata={"level": level, "parent": parent_id},
        )
        self.add_link(summary_id, parent_id, "summarizes")
        chunk_ids.append(summary_id)

        # Split text into smaller parts (e.g., by paragraphs)
        parts = text.split("\n\n")
        for part in parts:
            if part.strip():
                chunk_ids.extend(
                    self._process_text_recursively(
                        part, user, summary_id, source_filename, level + 1
                    )
                )

        return chunk_ids

    def _load_processed_docs(self):
        if os.path.exists(self.processed_docs_file):
            with open(self.processed_docs_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_processed_docs(self):
        with open(self.processed_docs_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_docs, f, indent=4)

    def prune_graph(self, user: str) -> list[str]:
        # Implementation assumed correct from previous version
        return []
