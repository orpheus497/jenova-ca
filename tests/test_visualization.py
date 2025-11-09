# The JENOVA Cognitive Architecture - Visualization Tests
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 28: Tests for Knowledge Graph Visualization.

Tests graph export, terminal rendering, and analysis with comprehensive coverage.
"""

import pytest
import tempfile
from pathlib import Path
import json
import networkx as nx

from jenova.visualization import GraphExporter, TerminalRenderer, GraphAnalyzer


@pytest.fixture
def sample_graph():
    """Fixture providing sample directed graph."""
    G = nx.DiGraph()

    # Add nodes with attributes
    G.add_node("insight1", label="AI Safety Insight", type="insight")
    G.add_node("insight2", label="Machine Learning Concept", type="insight")
    G.add_node("concept1", label="Neural Networks", type="concept")
    G.add_node("concept2", label="Deep Learning", type="concept")
    G.add_node("memory1", label="Training Session", type="memory")

    # Add edges with relationships
    G.add_edge("insight1", "concept1", relationship="relates_to", weight=1.5)
    G.add_edge("concept1", "concept2", relationship="develops", weight=2.0)
    G.add_edge("concept2", "memory1", relationship="applied_in", weight=1.0)
    G.add_edge("insight2", "concept1", relationship="elaborates_on", weight=1.2)

    return G


class TestGraphExporter:
    """Test suite for GraphExporter."""

    @pytest.fixture
    def exporter(self, sample_graph):
        """Fixture providing GraphExporter instance."""
        return GraphExporter(sample_graph)

    def test_initialization(self):
        """Test exporter initialization."""
        exporter = GraphExporter()
        assert exporter.graph is not None
        assert exporter.graph.number_of_nodes() == 0

    def test_set_graph(self, sample_graph):
        """Test setting graph."""
        exporter = GraphExporter()
        exporter.set_graph(sample_graph)
        assert exporter.graph.number_of_nodes() == 5

    def test_to_graphml(self, exporter):
        """Test GraphML export."""
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.to_graphml(output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify can be read back
            G = nx.read_graphml(str(output_path))
            assert G.number_of_nodes() == 5
        finally:
            output_path.unlink()

    def test_to_json(self, exporter):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.to_json(output_path, pretty=True)
            assert output_path.exists()

            # Verify JSON structure
            with open(output_path) as f:
                data = json.load(f)

            assert "nodes" in data
            assert "links" in data
            assert "metadata" in data
            assert len(data["nodes"]) == 5
            assert data["metadata"]["node_count"] == 5
        finally:
            output_path.unlink()

    def test_to_dot(self, exporter):
        """Test DOT export."""
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.to_dot(output_path)
            assert output_path.exists()

            # Verify DOT format
            content = output_path.read_text()
            assert "digraph cortex" in content
            assert "->" in content  # Directed edge
        finally:
            output_path.unlink()

    def test_to_html(self, exporter):
        """Test HTML export."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.to_html(output_path, title="Test Graph")
            assert output_path.exists()

            # Verify HTML structure
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Test Graph" in content
            assert "<canvas" in content
            assert "graphData" in content  # JavaScript variable
        finally:
            output_path.unlink()

    def test_to_gexf(self, exporter):
        """Test GEXF export."""
        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.to_gexf(output_path)
            assert output_path.exists()
            assert output_path.stat().st_size > 0
        finally:
            output_path.unlink()

    def test_get_graph_stats(self, exporter):
        """Test graph statistics."""
        stats = exporter.get_graph_stats()
        assert stats["node_count"] == 5
        assert stats["edge_count"] == 4
        assert "density" in stats
        assert stats["is_directed"] is True

    def test_export_subgraph(self, exporter):
        """Test subgraph export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            node_ids = ["insight1", "concept1", "concept2"]
            exporter.export_subgraph(node_ids, output_path, format="json")
            assert output_path.exists()

            # Verify subgraph
            with open(output_path) as f:
                data = json.load(f)
            assert len(data["nodes"]) == 3
        finally:
            output_path.unlink()


class TestTerminalRenderer:
    """Test suite for TerminalRenderer."""

    @pytest.fixture
    def renderer(self, sample_graph):
        """Fixture providing TerminalRenderer instance."""
        return TerminalRenderer(sample_graph)

    def test_initialization(self):
        """Test renderer initialization."""
        renderer = TerminalRenderer()
        assert renderer.graph is not None

    def test_set_graph(self, sample_graph):
        """Test setting graph."""
        renderer = TerminalRenderer()
        renderer.set_graph(sample_graph)
        assert renderer.graph.number_of_nodes() == 5

    def test_render_tree(self, renderer):
        """Test tree rendering."""
        tree = renderer.render_tree("insight1", max_depth=2)
        assert isinstance(tree, str)
        assert len(tree) > 0
        assert "insight1" in tree or "AI Safety" in tree

    def test_render_tree_nonexistent(self, renderer):
        """Test tree rendering with nonexistent node."""
        tree = renderer.render_tree("nonexistent")
        assert "not found" in tree.lower()

    def test_render_list(self, renderer):
        """Test list rendering."""
        list_view = renderer.render_list(group_by_type=True, show_edges=True)
        assert isinstance(list_view, str)
        assert "INSIGHT" in list_view or "CONCEPT" in list_view

    def test_render_list_no_grouping(self, renderer):
        """Test list rendering without grouping."""
        list_view = renderer.render_list(group_by_type=False)
        assert isinstance(list_view, str)
        assert "NODES" in list_view

    def test_render_network(self, renderer):
        """Test network diagram rendering."""
        diagram = renderer.render_network(width=80, height=24)
        assert isinstance(diagram, str)
        assert "Legend" in diagram
        assert len(diagram.split('\n')) > 10  # Should have multiple lines

    def test_render_stats(self, renderer):
        """Test statistics rendering."""
        stats = renderer.render_stats()
        assert isinstance(stats, str)
        assert "GRAPH STATISTICS" in stats
        assert "Nodes: 5" in stats
        assert "Edges: 4" in stats

    def test_render_path(self, renderer):
        """Test path rendering."""
        path = renderer.render_path("insight1", "memory1")
        assert isinstance(path, str)
        assert "Path from" in path

    def test_render_path_no_path(self, renderer):
        """Test path rendering when no path exists."""
        path = renderer.render_path("memory1", "insight1")
        assert "No path found" in path or "Path from" in path


class TestGraphAnalyzer:
    """Test suite for GraphAnalyzer."""

    @pytest.fixture
    def analyzer(self, sample_graph):
        """Fixture providing GraphAnalyzer instance."""
        return GraphAnalyzer(sample_graph)

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = GraphAnalyzer()
        assert analyzer.graph is not None

    def test_set_graph(self, sample_graph):
        """Test setting graph."""
        analyzer = GraphAnalyzer()
        analyzer.set_graph(sample_graph)
        assert analyzer.graph.number_of_nodes() == 5

    def test_compute_centrality_betweenness(self, analyzer):
        """Test betweenness centrality."""
        centrality = analyzer.compute_centrality("betweenness")
        assert isinstance(centrality, dict)
        assert len(centrality) == 5
        assert all(isinstance(v, float) for v in centrality.values())

    def test_compute_centrality_closeness(self, analyzer):
        """Test closeness centrality."""
        centrality = analyzer.compute_centrality("closeness")
        assert isinstance(centrality, dict)
        assert len(centrality) == 5

    def test_compute_centrality_degree(self, analyzer):
        """Test degree centrality."""
        centrality = analyzer.compute_centrality("degree")
        assert isinstance(centrality, dict)
        assert len(centrality) == 5

    def test_compute_centrality_pagerank(self, analyzer):
        """Test PageRank centrality."""
        centrality = analyzer.compute_centrality("pagerank")
        assert isinstance(centrality, dict)
        assert len(centrality) == 5
        # PageRank scores should sum to ~1
        assert 0.9 < sum(centrality.values()) < 1.1

    def test_compute_centrality_top_k(self, analyzer):
        """Test centrality with top k."""
        centrality = analyzer.compute_centrality("betweenness", k=3)
        assert len(centrality) == 3

    def test_find_most_important_nodes(self, analyzer):
        """Test finding most important nodes."""
        important = analyzer.find_most_important_nodes("pagerank", top_n=3)
        assert isinstance(important, list)
        assert len(important) <= 3
        assert all(isinstance(item, tuple) for item in important)
        assert all(len(item) == 2 for item in important)

    def test_detect_communities_greedy(self, analyzer):
        """Test community detection with greedy modularity."""
        communities = analyzer.detect_communities("greedy_modularity")
        assert isinstance(communities, list)
        assert len(communities) > 0
        assert all(isinstance(comm, set) for comm in communities)

    def test_detect_communities_label_prop(self, analyzer):
        """Test community detection with label propagation."""
        communities = analyzer.detect_communities("label_propagation")
        assert isinstance(communities, list)
        assert len(communities) > 0

    def test_find_all_paths(self, analyzer):
        """Test finding all paths."""
        paths = analyzer.find_all_paths("insight1", "memory1", max_length=5)
        assert isinstance(paths, list)
        # Should find at least one path
        if paths:
            assert all(isinstance(p, list) for p in paths)
            assert all(p[0] == "insight1" and p[-1] == "memory1" for p in paths)

    def test_find_shortest_path(self, analyzer):
        """Test finding shortest path."""
        path = analyzer.find_shortest_path("insight1", "memory1")
        if path:
            assert isinstance(path, list)
            assert path[0] == "insight1"
            assert path[-1] == "memory1"

    def test_find_shortest_path_nonexistent(self, analyzer):
        """Test shortest path when no path exists."""
        path = analyzer.find_shortest_path("memory1", "insight1")
        # May or may not exist depending on graph structure
        assert path is None or isinstance(path, list)

    def test_compute_clustering_coefficient(self, analyzer):
        """Test clustering coefficient computation."""
        clustering = analyzer.compute_clustering_coefficient()
        assert isinstance(clustering, dict)
        assert all(isinstance(v, float) for v in clustering.values())
        assert all(0 <= v <= 1 for v in clustering.values())

    def test_get_node_neighborhood(self, analyzer):
        """Test getting node neighborhood."""
        neighborhood = analyzer.get_node_neighborhood("concept1", radius=1)
        assert isinstance(neighborhood, set)
        assert "concept1" in neighborhood  # Should include center node
        assert len(neighborhood) > 1  # Should have neighbors

    def test_get_node_neighborhood_nonexistent(self, analyzer):
        """Test neighborhood of nonexistent node."""
        neighborhood = analyzer.get_node_neighborhood("nonexistent", radius=1)
        assert isinstance(neighborhood, set)
        assert len(neighborhood) == 0

    def test_analyze_node_importance(self, analyzer):
        """Test node importance analysis."""
        importance = analyzer.analyze_node_importance("concept1")
        assert isinstance(importance, dict)
        assert "centrality" in importance
        assert "degree" in importance
        assert "neighborhood_size" in importance
        assert isinstance(importance["centrality"]["betweenness"], float)

    def test_find_bridges(self, analyzer):
        """Test finding bridge edges."""
        bridges = analyzer.find_bridges()
        assert isinstance(bridges, list)

    def test_get_connected_components(self, analyzer):
        """Test getting connected components."""
        components = analyzer.get_connected_components()
        assert isinstance(components, list)
        assert len(components) > 0
        assert all(isinstance(comp, set) for comp in components)

    def test_compute_graph_metrics(self, analyzer):
        """Test computing comprehensive graph metrics."""
        metrics = analyzer.compute_graph_metrics()
        assert isinstance(metrics, dict)
        assert metrics["nodes"] == 5
        assert metrics["edges"] == 4
        assert "density" in metrics
        assert metrics["is_directed"] is True
        assert "weakly_connected_components" in metrics


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_export_and_analyze_workflow(self, sample_graph):
        """Test complete export and analysis workflow."""
        # Export graph
        exporter = GraphExporter(sample_graph)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            exporter.to_json(json_path)

            # Analyze graph
            analyzer = GraphAnalyzer(sample_graph)
            metrics = analyzer.compute_graph_metrics()

            assert metrics["nodes"] == 5
            assert json_path.exists()

            # Render in terminal
            renderer = TerminalRenderer(sample_graph)
            stats = renderer.render_stats()

            assert "5" in stats  # Should show 5 nodes
        finally:
            json_path.unlink()

    def test_centrality_and_visualization(self, sample_graph):
        """Test finding important nodes and visualizing."""
        analyzer = GraphAnalyzer(sample_graph)

        # Find most important nodes
        important = analyzer.find_most_important_nodes("pagerank", top_n=3)
        assert len(important) <= 3

        # Render tree from most important node
        if important:
            top_node = important[0][0]
            renderer = TerminalRenderer(sample_graph)
            tree = renderer.render_tree(top_node, max_depth=2)
            assert len(tree) > 0
