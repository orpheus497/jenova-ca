# The JENOVA Cognitive Architecture - Graph Analyzer
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 28: Graph Analyzer - Analysis tools for knowledge graph.

Provides centrality metrics, clustering, community detection, and
path analysis. All algorithms are 100% offline and FOSS (NetworkX).
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx


class GraphAnalyzer:
    """
    Analyze knowledge graph structure and properties.

    Provides centrality metrics, clustering coefficients, community
    detection, and path analysis using NetworkX algorithms.

    Example:
        >>> analyzer = GraphAnalyzer(graph)
        >>> centrality = analyzer.compute_centrality("betweenness")
        >>> communities = analyzer.detect_communities()
        >>> paths = analyzer.find_all_paths("node1", "node5", max_length=3)
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize graph analyzer.

        Args:
            graph: NetworkX directed graph
        """
        self.graph = graph if graph is not None else nx.DiGraph()

    def set_graph(self, graph: nx.DiGraph) -> None:
        """Set the graph to analyze."""
        self.graph = graph

    def compute_centrality(
        self,
        metric: str = "betweenness",
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute centrality metrics for all nodes.

        Args:
            metric: Centrality type (betweenness, closeness, degree, pagerank, eigenvector)
            k: Number of top nodes to return (None = all)

        Returns:
            Dict mapping node IDs to centrality scores

        Example:
            >>> centrality = analyzer.compute_centrality("betweenness", k=10)
        """
        if metric == "betweenness":
            scores = nx.betweenness_centrality(self.graph)
        elif metric == "closeness":
            scores = nx.closeness_centrality(self.graph)
        elif metric == "degree":
            scores = nx.degree_centrality(self.graph)
        elif metric == "pagerank":
            scores = nx.pagerank(self.graph)
        elif metric == "eigenvector":
            try:
                scores = nx.eigenvector_centrality(self.graph, max_iter=100)
            except nx.PowerIterationFailedConvergence:
                scores = {}
        else:
            raise ValueError(f"Unknown centrality metric: {metric}")

        # Return top k if specified
        if k is not None and k > 0:
            sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_nodes[:k])

        return scores

    def find_most_important_nodes(
        self,
        metric: str = "betweenness",
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most important nodes by centrality metric.

        Args:
            metric: Centrality metric to use
            top_n: Number of top nodes to return

        Returns:
            List of (node_id, score) tuples

        Example:
            >>> important = analyzer.find_most_important_nodes("pagerank", top_n=5)
        """
        scores = self.compute_centrality(metric)
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def detect_communities(self, algorithm: str = "louvain") -> List[Set[str]]:
        """
        Detect communities/clusters in graph.

        Args:
            algorithm: Algorithm to use (louvain, label_propagation, greedy_modularity)

        Returns:
            List of node sets (each set is a community)

        Example:
            >>> communities = analyzer.detect_communities("louvain")
            >>> print(f"Found {len(communities)} communities")
        """
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()

        if algorithm == "louvain":
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(undirected)
                communities_dict: Dict[int, Set[str]] = {}
                for node, comm_id in partition.items():
                    if comm_id not in communities_dict:
                        communities_dict[comm_id] = set()
                    communities_dict[comm_id].add(node)
                return list(communities_dict.values())
            except ImportError:
                # Fallback to greedy modularity
                return self.detect_communities("greedy_modularity")

        elif algorithm == "label_propagation":
            communities = nx.algorithms.community.label_propagation_communities(undirected)
            return [set(comm) for comm in communities]

        elif algorithm == "greedy_modularity":
            communities = nx.algorithms.community.greedy_modularity_communities(undirected)
            return [set(comm) for comm in communities]

        else:
            raise ValueError(f"Unknown community detection algorithm: {algorithm}")

    def find_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        Find all simple paths between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length

        Returns:
            List of paths (each path is a list of node IDs)

        Example:
            >>> paths = analyzer.find_all_paths("node1", "node5", max_length=3)
        """
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length
            ))
            return paths
        except nx.NodeNotFound:
            return []

    def find_shortest_path(
        self,
        source: str,
        target: str,
        weight: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge attribute to use as weight (None = unweighted)

        Returns:
            Path as list of node IDs, or None if no path exists

        Example:
            >>> path = analyzer.find_shortest_path("node1", "node10")
        """
        try:
            path = nx.shortest_path(self.graph, source, target, weight=weight)
            return path
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

    def compute_clustering_coefficient(self) -> Dict[str, float]:
        """
        Compute clustering coefficient for all nodes.

        Returns:
            Dict mapping node IDs to clustering coefficients

        Example:
            >>> clustering = analyzer.compute_clustering_coefficient()
        """
        # Convert to undirected for clustering
        undirected = self.graph.to_undirected()
        return nx.clustering(undirected)

    def get_node_neighborhood(
        self,
        node_id: str,
        radius: int = 1
    ) -> Set[str]:
        """
        Get all nodes within radius of given node.

        Args:
            node_id: Center node ID
            radius: Number of hops

        Returns:
            Set of node IDs in neighborhood

        Example:
            >>> neighborhood = analyzer.get_node_neighborhood("node1", radius=2)
        """
        if node_id not in self.graph:
            return set()

        neighborhood = {node_id}

        for _ in range(radius):
            new_nodes = set()
            for node in neighborhood:
                # Add successors and predecessors
                new_nodes.update(self.graph.successors(node))
                new_nodes.update(self.graph.predecessors(node))
            neighborhood.update(new_nodes)

        return neighborhood

    def analyze_node_importance(
        self,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Comprehensive importance analysis for a node.

        Args:
            node_id: Node to analyze

        Returns:
            Dict with various importance metrics

        Example:
            >>> importance = analyzer.analyze_node_importance("node42")
            >>> print(importance["centrality"]["betweenness"])
        """
        if node_id not in self.graph:
            return {}

        # Compute various centrality metrics
        betweenness = nx.betweenness_centrality(self.graph)
        closeness = nx.closeness_centrality(self.graph)
        degree_cent = nx.degree_centrality(self.graph)
        pagerank = nx.pagerank(self.graph)

        # Get degree
        in_degree = self.graph.in_degree(node_id)
        out_degree = self.graph.out_degree(node_id)

        return {
            "node_id": node_id,
            "centrality": {
                "betweenness": betweenness.get(node_id, 0),
                "closeness": closeness.get(node_id, 0),
                "degree": degree_cent.get(node_id, 0),
                "pagerank": pagerank.get(node_id, 0),
            },
            "degree": {
                "in": in_degree,
                "out": out_degree,
                "total": in_degree + out_degree,
            },
            "neighborhood_size": len(self.get_node_neighborhood(node_id, radius=1)) - 1,
        }

    def find_bridges(self) -> List[Tuple[str, str]]:
        """
        Find bridge edges (edges whose removal disconnects the graph).

        Returns:
            List of (source, target) edge tuples

        Example:
            >>> bridges = analyzer.find_bridges()
        """
        # Convert to undirected
        undirected = self.graph.to_undirected()
        bridges = list(nx.bridges(undirected))
        return bridges

    def get_connected_components(self) -> List[Set[str]]:
        """
        Get weakly connected components.

        Returns:
            List of node sets (each set is a connected component)

        Example:
            >>> components = analyzer.get_connected_components()
            >>> print(f"{len(components)} connected components")
        """
        components = list(nx.weakly_connected_components(self.graph))
        return components

    def compute_graph_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive graph metrics.

        Returns:
            Dict with various graph-level metrics

        Example:
            >>> metrics = analyzer.compute_graph_metrics()
            >>> print(metrics["density"], metrics["avg_clustering"])
        """
        metrics = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_directed": self.graph.is_directed(),
        }

        # Connected components
        if self.graph.is_directed():
            metrics["weakly_connected_components"] = nx.number_weakly_connected_components(self.graph)
            metrics["strongly_connected_components"] = nx.number_strongly_connected_components(self.graph)
        else:
            metrics["connected_components"] = nx.number_connected_components(self.graph)

        # Clustering (on undirected version)
        undirected = self.graph.to_undirected()
        clustering_coeffs = nx.clustering(undirected)
        if clustering_coeffs:
            metrics["avg_clustering"] = sum(clustering_coeffs.values()) / len(clustering_coeffs)
        else:
            metrics["avg_clustering"] = 0.0

        # Degree statistics
        if self.graph.number_of_nodes() > 0:
            in_degrees = [d for _, d in self.graph.in_degree()]
            out_degrees = [d for _, d in self.graph.out_degree()]

            metrics["degree_stats"] = {
                "avg_in_degree": sum(in_degrees) / len(in_degrees),
                "avg_out_degree": sum(out_degrees) / len(out_degrees),
                "max_in_degree": max(in_degrees) if in_degrees else 0,
                "max_out_degree": max(out_degrees) if out_degrees else 0,
            }

        return metrics
