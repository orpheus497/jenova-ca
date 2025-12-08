##Script function and purpose: Graph Metrics Module for The JENOVA Cognitive Architecture
##This module provides comprehensive graph analysis metrics including clustering coefficient,
##modularity, path analysis, density metrics, and various centrality measures

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

##Class purpose: Calculates comprehensive graph metrics for cognitive graph analysis
class GraphMetrics:
    ##Function purpose: Initialize graph metrics calculator
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_config = config.get('cortex', {}).get('metrics', {})
        self.enabled = self.metrics_config.get('enabled', True)
    
    ##Function purpose: Convert cognitive graph JSON structure to NetworkX graph object
    def _build_nx_graph(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> nx.Graph:
        """
        Converts the cognitive graph JSON structure to a NetworkX graph.
        Filters by user if specified.
        """
        G = nx.Graph()
        
        # Add nodes
        for node_id, node_data in graph_data.get('nodes', {}).items():
            if user is None or node_data.get('user') == user:
                G.add_node(node_id, **node_data)
        
        # Add edges with relationship weights
        weights = self.config.get('cortex', {}).get('relationship_weights', {})
        for link in graph_data.get('links', []):
            source = link.get('source')
            target = link.get('target')
            relationship = link.get('relationship', 'related_to')
            
            # Only add edge if both nodes exist in filtered graph
            if source in G and target in G:
                weight = weights.get(relationship, 1.0)
                G.add_edge(source, target, relationship=relationship, weight=weight)
        
        return G
    
    ##Function purpose: Calculate clustering coefficient for each node and average
    def calculate_clustering_coefficient(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates local clustering coefficient for each node and average clustering coefficient.
        Returns dict with 'node_coefficients' and 'average_coefficient'.
        """
        if not self.metrics_config.get('calculate_clustering_coefficient', True):
            return {}
        
        G = self._build_nx_graph(graph_data, user)
        
        if len(G) == 0:
            return {'node_coefficients': {}, 'average_coefficient': 0.0}
        
        # Calculate local clustering coefficient for each node
        node_coefficients = nx.clustering(G)
        
        # Calculate average clustering coefficient
        avg_coefficient = sum(node_coefficients.values()) / len(node_coefficients) if node_coefficients else 0.0
        
        return {
            'node_coefficients': node_coefficients,
            'average_coefficient': round(avg_coefficient, 4)
        }
    
    ##Function purpose: Calculate modularity to measure community structure quality
    def calculate_modularity(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates modularity of the graph using community detection.
        Returns dict with 'modularity' score and 'communities' structure.
        """
        if not self.metrics_config.get('calculate_modularity', True):
            return {}
        
        G = self._build_nx_graph(graph_data, user)
        
        if len(G) == 0:
            return {'modularity': 0.0, 'communities': []}
        
        try:
            # Use greedy modularity communities algorithm (Louvain-like)
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
            
            # Convert communities from frozensets to lists for JSON serialization
            communities_list = [list(community) for community in communities]
            
            return {
                'modularity': round(modularity, 4),
                'communities': communities_list,
                'num_communities': len(communities_list)
            }
        except Exception:
            # Fallback if modularity calculation fails
            return {'modularity': 0.0, 'communities': [], 'num_communities': 0}
    
    ##Function purpose: Calculate average path length and diameter of the graph
    def calculate_path_metrics(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates average shortest path length and graph diameter.
        Only works for connected graphs.
        """
        G = self._build_nx_graph(graph_data, user)
        
        if len(G) == 0:
            return {'average_path_length': 0.0, 'diameter': 0, 'is_connected': False}
        
        # Check if graph is connected
        is_connected = nx.is_connected(G)
        
        if not is_connected:
            # For disconnected graphs, calculate metrics for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc).copy()
            
            if len(G_largest) > 1:
                avg_path_length = nx.average_shortest_path_length(G_largest)
                diameter = nx.diameter(G_largest)
            else:
                avg_path_length = 0.0
                diameter = 0
        else:
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        
        return {
            'average_path_length': round(avg_path_length, 4),
            'diameter': diameter,
            'is_connected': is_connected,
            'num_components': nx.number_connected_components(G)
        }
    
    ##Function purpose: Calculate graph density and related metrics
    def calculate_density_metrics(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates graph density, edge density, and node degree statistics.
        """
        G = self._build_nx_graph(graph_data, user)
        
        if len(G) == 0:
            return {
                'density': 0.0,
                'num_nodes': 0,
                'num_edges': 0,
                'average_degree': 0.0,
                'max_degree': 0,
                'min_degree': 0
            }
        
        density = nx.density(G)
        degrees = dict(G.degree())
        
        return {
            'density': round(density, 4),
            'num_nodes': len(G),
            'num_edges': G.number_of_edges(),
            'average_degree': round(sum(degrees.values()) / len(degrees), 2) if degrees else 0.0,
            'max_degree': max(degrees.values()) if degrees else 0,
            'min_degree': min(degrees.values()) if degrees else 0
        }
    
    ##Function purpose: Calculate betweenness centrality for each node
    def calculate_betweenness_centrality(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> Dict[str, float]:
        """
        Calculates betweenness centrality for each node.
        Measures how often a node appears on shortest paths between other nodes.
        """
        G = self._build_nx_graph(graph_data, user)
        
        if len(G) < 2:
            return {}
        
        try:
            betweenness = nx.betweenness_centrality(G)
            return {node_id: round(score, 4) for node_id, score in betweenness.items()}
        except Exception:
            return {}
    
    ##Function purpose: Calculate closeness centrality for each node
    def calculate_closeness_centrality(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> Dict[str, float]:
        """
        Calculates closeness centrality for each node.
        Measures average distance from a node to all other nodes.
        """
        G = self._build_nx_graph(graph_data, user)
        
        if len(G) < 2:
            return {}
        
        try:
            # For disconnected graphs, calculate for largest component
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            
            if len(G) < 2:
                return {}
            
            closeness = nx.closeness_centrality(G)
            return {node_id: round(score, 4) for node_id, score in closeness.items()}
        except Exception:
            return {}
    
    ##Function purpose: Calculate comprehensive metrics suite for the graph
    def calculate_all_metrics(self, graph_data: Dict[str, Any], user: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates all available graph metrics and returns comprehensive report.
        """
        if not self.enabled:
            return {}
        
        metrics = {
            'clustering': self.calculate_clustering_coefficient(graph_data, user),
            'modularity': self.calculate_modularity(graph_data, user),
            'path_metrics': self.calculate_path_metrics(graph_data, user),
            'density': self.calculate_density_metrics(graph_data, user),
            'betweenness_centrality': self.calculate_betweenness_centrality(graph_data, user),
            'closeness_centrality': self.calculate_closeness_centrality(graph_data, user)
        }
        
        return metrics
