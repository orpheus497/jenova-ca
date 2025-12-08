##Script function and purpose: Advanced Clustering Module for The JENOVA Cognitive Architecture
##This module provides advanced graph clustering algorithms including community detection,
##hierarchical clustering, and pattern recognition for meta-insight generation

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

##Class purpose: Provides advanced clustering algorithms for cognitive graph analysis
class AdvancedClustering:
    ##Function purpose: Initialize clustering module with configuration
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clustering_config = config.get('cortex', {}).get('clustering', {})
        self.algorithm = self.clustering_config.get('algorithm', 'louvain')
        self.min_cluster_size = self.clustering_config.get('min_cluster_size', 3)
        self.weights = config.get('cortex', {}).get('relationship_weights', {})
    
    ##Function purpose: Convert cognitive graph JSON structure to NetworkX graph object
    def _build_nx_graph(self, graph_data: Dict[str, Any], user_nodes: List[Dict[str, Any]]) -> nx.Graph:
        """
        Converts the cognitive graph JSON structure to a NetworkX graph.
        Only includes nodes from user_nodes list.
        """
        G = nx.Graph()
        user_node_ids = {node['id'] for node in user_nodes}
        
        # Add nodes
        for node in user_nodes:
            G.add_node(node['id'], **node)
        
        # Add edges with relationship weights
        for link in graph_data.get('links', []):
            source = link.get('source')
            target = link.get('target')
            relationship = link.get('relationship', 'related_to')
            
            # Only add edge if both nodes exist in user_nodes
            if source in user_node_ids and target in user_node_ids:
                weight = self.weights.get(relationship, 1.0)
                G.add_edge(source, target, relationship=relationship, weight=weight)
        
        return G
    
    ##Function purpose: Detect communities using Louvain algorithm
    def _detect_communities_louvain(self, G: nx.Graph) -> List[List[str]]:
        """
        Detects communities using the Louvain algorithm (greedy modularity maximization).
        Returns list of communities, each community is a list of node IDs.
        """
        try:
            communities = nx.community.greedy_modularity_communities(G)
            # Convert frozensets to lists and filter by minimum cluster size
            communities_list = [list(community) for community in communities if len(community) >= self.min_cluster_size]
            return communities_list
        except Exception:
            # Fallback to basic connected components if Louvain fails
            return self._fallback_clustering(G)
    
    ##Function purpose: Detect communities using Leiden algorithm (via networkx)
    def _detect_communities_leiden(self, G: nx.Graph) -> List[List[str]]:
        """
        Detects communities using Leiden algorithm (improved Louvain).
        Falls back to Louvain if Leiden is not available.
        """
        try:
            # NetworkX doesn't have Leiden directly, so we use Louvain as approximation
            # In production, could use python-leidenalg library if needed
            return self._detect_communities_louvain(G)
        except Exception:
            return self._fallback_clustering(G)
    
    ##Function purpose: Perform hierarchical clustering based on node similarity
    def _hierarchical_clustering(self, G: nx.Graph, user_nodes: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Performs hierarchical clustering based on graph structure and node similarity.
        Uses agglomerative approach with distance based on shortest paths.
        """
        if len(G) < self.min_cluster_size:
            return []
        
        # Build distance matrix based on shortest paths
        node_ids = list(G.nodes())
        distances = {}
        
        for i, node1 in enumerate(node_ids):
            for j, node2 in enumerate(node_ids):
                if i < j:
                    try:
                        path_length = nx.shortest_path_length(G, node1, node2)
                        distances[(node1, node2)] = path_length
                    except nx.NetworkXNoPath:
                        distances[(node1, node2)] = float('inf')
        
        # Simple hierarchical clustering: merge closest nodes iteratively
        clusters = [[node_id] for node_id in node_ids]
        
        # Continue merging until we have reasonable number of clusters
        max_clusters = max(1, len(node_ids) // self.min_cluster_size)
        
        while len(clusters) > max_clusters:
            # Find two closest clusters
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate average distance between clusters
                    cluster_distance = self._cluster_distance(clusters[i], clusters[j], distances)
                    if cluster_distance < min_distance:
                        min_distance = cluster_distance
                        merge_i, merge_j = i, j
            
            if merge_i == -1 or min_distance == float('inf'):
                break
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Filter clusters by minimum size
        return [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size]
    
    ##Function purpose: Calculate distance between two clusters
    def _cluster_distance(self, cluster1: List[str], cluster2: List[str], distances: Dict[Tuple[str, str], float]) -> float:
        """
        Calculates average distance between two clusters.
        Uses average of all pairwise distances between cluster members.
        """
        total_distance = 0.0
        count = 0
        
        for node1 in cluster1:
            for node2 in cluster2:
                if (node1, node2) in distances:
                    total_distance += distances[(node1, node2)]
                    count += 1
                elif (node2, node1) in distances:
                    total_distance += distances[(node2, node1)]
                    count += 1
        
        return total_distance / count if count > 0 else float('inf')
    
    ##Function purpose: Fallback clustering using basic connected components
    def _fallback_clustering(self, G: nx.Graph) -> List[List[str]]:
        """
        Fallback clustering method using connected components.
        Used when advanced algorithms fail.
        """
        components = list(nx.connected_components(G))
        clusters = [list(component) for component in components if len(component) >= self.min_cluster_size]
        return clusters
    
    ##Function purpose: Calculate cluster quality metrics
    def calculate_cluster_quality(self, G: nx.Graph, clusters: List[List[str]]) -> Dict[str, Any]:
        """
        Calculates quality metrics for detected clusters.
        Returns metrics including modularity, intra-cluster density, inter-cluster sparsity.
        """
        if not clusters:
            return {
                'modularity': 0.0,
                'average_intra_cluster_density': 0.0,
                'average_inter_cluster_sparsity': 0.0,
                'num_clusters': 0,
                'average_cluster_size': 0.0
            }
        
        # Calculate modularity
        try:
            communities = [set(cluster) for cluster in clusters]
            modularity = nx.community.modularity(G, communities)
        except Exception:
            modularity = 0.0
        
        # Calculate intra-cluster density (average edges within clusters)
        intra_densities = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            subgraph = G.subgraph(cluster)
            if len(subgraph) > 1:
                density = nx.density(subgraph)
                intra_densities.append(density)
        
        avg_intra_density = sum(intra_densities) / len(intra_densities) if intra_densities else 0.0
        
        # Calculate inter-cluster sparsity (average edges between clusters)
        inter_edges = 0
        total_possible_inter = 0
        
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i < j:
                    cluster1_set = set(cluster1)
                    cluster2_set = set(cluster2)
                    # Count edges between clusters
                    for node1 in cluster1:
                        for node2 in cluster2:
                            total_possible_inter += 1
                            if G.has_edge(node1, node2):
                                inter_edges += 1
        
        inter_sparsity = 1.0 - (inter_edges / total_possible_inter) if total_possible_inter > 0 else 1.0
        
        return {
            'modularity': round(modularity, 4),
            'average_intra_cluster_density': round(avg_intra_density, 4),
            'average_inter_cluster_sparsity': round(inter_sparsity, 4),
            'num_clusters': len(clusters),
            'average_cluster_size': round(sum(len(c) for c in clusters) / len(clusters), 2) if clusters else 0.0
        }
    
    ##Function purpose: Detect communities using configured algorithm
    def detect_communities(self, graph_data: Dict[str, Any], user_nodes: List[Dict[str, Any]]) -> Tuple[List[List[str]], Dict[str, Any]]:
        """
        Main entry point for community detection.
        Returns clusters and quality metrics.
        """
        if len(user_nodes) < self.min_cluster_size:
            return [], {}
        
        G = self._build_nx_graph(graph_data, user_nodes)
        
        if len(G) == 0:
            return [], {}
        
        # Select clustering algorithm
        if self.algorithm == 'louvain':
            clusters = self._detect_communities_louvain(G)
        elif self.algorithm == 'leiden':
            clusters = self._detect_communities_leiden(G)
        elif self.algorithm == 'hierarchical':
            clusters = self._hierarchical_clustering(G, user_nodes)
        else:
            # Default to Louvain
            clusters = self._detect_communities_louvain(G)
        
        # Calculate quality metrics
        quality_metrics = self.calculate_cluster_quality(G, clusters)
        
        return clusters, quality_metrics
    
    ##Function purpose: Identify pattern clusters for meta-insight generation
    def identify_pattern_clusters(self, graph_data: Dict[str, Any], user_nodes: List[Dict[str, Any]], 
                                 min_pattern_size: int = 3) -> List[List[str]]:
        """
        Identifies clusters that represent interesting patterns for meta-insight generation.
        Filters clusters based on quality metrics and pattern characteristics.
        """
        clusters, quality_metrics = self.detect_communities(graph_data, user_nodes)
        
        if not clusters:
            return []
        
        # Filter clusters based on quality
        # Prefer clusters with high intra-cluster density
        pattern_clusters = []
        
        G = self._build_nx_graph(graph_data, user_nodes)
        
        for cluster in clusters:
            if len(cluster) < min_pattern_size:
                continue
            
            # Calculate cluster-specific metrics
            subgraph = G.subgraph(cluster)
            if len(subgraph) < 2:
                continue
            
            density = nx.density(subgraph)
            
            # Include clusters with reasonable density (at least 0.2)
            if density >= 0.2:
                pattern_clusters.append(cluster)
        
        return pattern_clusters
