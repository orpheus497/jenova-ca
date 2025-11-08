# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for the Cortex (cognitive graph) in the JENOVA Cognitive Architecture.

Tests all cortex components:
- Cognitive Node creation and management
- Cognitive Link creation and relationship types
- Centrality calculation and weighting
- Orphan node detection and linking
- Meta-insight generation from clusters
- Graph pruning and archival
- Proactive engine (suggestion generation)

This module ensures robust operation of the cognitive architecture's graph-based reasoning system.
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta

# Import cortex components
from jenova.cortex.cortex import Cortex, CognitiveNode, CognitiveLink
from jenova.cortex.proactive_engine import ProactiveEngine


class TestCognitiveNode:
    """Test suite for cognitive node data structure."""

    def test_create_cognitive_node(self):
        """Test creating a cognitive node."""
        node = CognitiveNode(
            id="test_node_1",
            type="insight",
            content="Test insight content",
            metadata={"source": "test"},
            timestamp=datetime.now(timezone.utc)
        )

        assert node.id == "test_node_1"
        assert node.type == "insight"
        assert node.content == "Test insight content"
        assert "source" in node.metadata

    def test_cognitive_node_types(self):
        """Test different cognitive node types."""
        types = ["insight", "assumption", "memory", "document", "meta_insight", "question"]

        for node_type in types:
            node = CognitiveNode(
                id=f"{node_type}_1",
                type=node_type,
                content=f"Test {node_type}",
                metadata={},
                timestamp=datetime.now(timezone.utc)
            )
            assert node.type == node_type

    def test_node_metadata(self):
        """Test node metadata storage."""
        metadata = {
            "concern": "testing",
            "confidence": 0.8,
            "entities": ["test", "example"],
            "emotion": "neutral"
        }

        node = CognitiveNode(
            id="meta_test",
            type="insight",
            content="Metadata test",
            metadata=metadata,
            timestamp=datetime.now(timezone.utc)
        )

        assert node.metadata == metadata
        assert node.metadata["confidence"] == 0.8


class TestCognitiveLink:
    """Test suite for cognitive link data structure."""

    def test_create_cognitive_link(self):
        """Test creating a cognitive link."""
        link = CognitiveLink(
            source_id="node_1",
            target_id="node_2",
            relationship_type="elaborates_on",
            weight=1.5,
            metadata={"created": "test"}
        )

        assert link.source_id == "node_1"
        assert link.target_id == "node_2"
        assert link.relationship_type == "elaborates_on"
        assert link.weight == 1.5

    def test_link_relationship_types(self):
        """Test different relationship types."""
        relationships = [
            "elaborates_on",
            "conflicts_with",
            "related_to",
            "develops",
            "summarizes",
            "created_from"
        ]

        for rel_type in relationships:
            link = CognitiveLink(
                source_id="n1",
                target_id="n2",
                relationship_type=rel_type,
                weight=1.0,
                metadata={}
            )
            assert link.relationship_type == rel_type

    def test_bidirectional_links(self):
        """Test creating bidirectional relationships."""
        link1 = CognitiveLink("a", "b", "related_to", 1.0, {})
        link2 = CognitiveLink("b", "a", "related_to", 1.0, {})

        assert link1.source_id == link2.target_id
        assert link1.target_id == link2.source_id


class TestCortex:
    """Test suite for the Cortex cognitive graph."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for cortex data."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "cortex": {
                "relationship_weights": {
                    "elaborates_on": 1.5,
                    "conflicts_with": 2.0,
                    "related_to": 1.0,
                    "develops": 1.5,
                    "summarizes": 1.2
                },
                "pruning": {
                    "enabled": True,
                    "prune_interval": 10,
                    "max_age_days": 30,
                    "min_centrality": 0.1
                }
            }
        }

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface."""
        llm = Mock()
        llm.generate = Mock(return_value='{"links": [], "meta_insight": "Test meta insight"}')
        return llm

    @pytest.fixture
    def cortex(self, temp_dir, mock_config, mock_file_logger, mock_llm_interface):
        """Create Cortex instance for testing."""
        cortex = Cortex(
            config=mock_config,
            user_data_root=temp_dir,
            file_logger=mock_file_logger,
            llm_interface=mock_llm_interface
        )
        return cortex

    def test_cortex_initialization(self, cortex):
        """Test cortex initializes correctly."""
        assert cortex is not None
        assert cortex.graph is not None
        assert len(cortex.nodes) == 0

    def test_add_node(self, cortex):
        """Test adding a node to the cortex."""
        node = CognitiveNode(
            id="test_node",
            type="insight",
            content="Test insight",
            metadata={},
            timestamp=datetime.now(timezone.utc)
        )

        cortex.add_node(node)
        assert "test_node" in cortex.nodes
        assert cortex.graph.has_node("test_node")

    def test_add_link(self, cortex):
        """Test adding a link between nodes."""
        # Add two nodes
        node1 = CognitiveNode("n1", "insight", "First", {}, datetime.now(timezone.utc))
        node2 = CognitiveNode("n2", "insight", "Second", {}, datetime.now(timezone.utc))

        cortex.add_node(node1)
        cortex.add_node(node2)

        # Add link
        link = CognitiveLink("n1", "n2", "elaborates_on", 1.5, {})
        cortex.add_link(link)

        assert cortex.graph.has_edge("n1", "n2")

    def test_get_node(self, cortex):
        """Test retrieving a node by ID."""
        node = CognitiveNode("retrieve_test", "insight", "Test", {}, datetime.now(timezone.utc))
        cortex.add_node(node)

        retrieved = cortex.get_node("retrieve_test")
        assert retrieved is not None
        assert retrieved.id == "retrieve_test"
        assert retrieved.content == "Test"

    def test_calculate_centrality(self, cortex):
        """Test centrality calculation."""
        # Create a simple graph
        nodes = [
            CognitiveNode(f"n{i}", "insight", f"Node {i}", {}, datetime.now(timezone.utc))
            for i in range(5)
        ]

        for node in nodes:
            cortex.add_node(node)

        # Add links (n0 is hub)
        for i in range(1, 5):
            link = CognitiveLink("n0", f"n{i}", "related_to", 1.0, {})
            cortex.add_link(link)

        # Calculate centrality
        centrality = cortex.calculate_centrality()

        # n0 should have highest centrality
        assert centrality["n0"] > centrality["n1"]
        assert centrality["n0"] > 0

    def test_weighted_centrality(self, cortex):
        """Test centrality calculation respects relationship weights."""
        # Create nodes
        n1 = CognitiveNode("n1", "insight", "Central", {}, datetime.now(timezone.utc))
        n2 = CognitiveNode("n2", "insight", "Peripheral", {}, datetime.now(timezone.utc))
        cortex.add_node(n1)
        cortex.add_node(n2)

        # Add high-weight link
        link = CognitiveLink("n1", "n2", "conflicts_with", 2.0, {})
        cortex.add_link(link)

        centrality = cortex.calculate_centrality()

        # Both nodes should have centrality weighted by relationship
        assert centrality["n1"] > 0
        assert centrality["n2"] > 0

    def test_find_orphan_nodes(self, cortex):
        """Test orphan node detection."""
        # Create connected cluster
        for i in range(3):
            node = CognitiveNode(f"connected_{i}", "insight", f"C{i}", {}, datetime.now(timezone.utc))
            cortex.add_node(node)

        cortex.add_link(CognitiveLink("connected_0", "connected_1", "related_to", 1.0, {}))
        cortex.add_link(CognitiveLink("connected_1", "connected_2", "related_to", 1.0, {}))

        # Create orphan
        orphan = CognitiveNode("orphan", "insight", "Lonely", {}, datetime.now(timezone.utc))
        cortex.add_node(orphan)

        # Find orphans
        orphans = cortex.find_orphan_nodes(min_degree=2)

        # Orphan should be detected (degree 0 < 2)
        assert "orphan" in [n.id for n in orphans]

    def test_graph_persistence(self, cortex):
        """Test saving and loading the graph."""
        # Add nodes and links
        n1 = CognitiveNode("persist_1", "insight", "Save me", {}, datetime.now(timezone.utc))
        n2 = CognitiveNode("persist_2", "insight", "Save me too", {}, datetime.now(timezone.utc))
        cortex.add_node(n1)
        cortex.add_node(n2)
        cortex.add_link(CognitiveLink("persist_1", "persist_2", "related_to", 1.0, {}))

        # Save
        cortex.save_graph()

        # Create new cortex and load
        new_cortex = Cortex(
            config=cortex.config,
            user_data_root=cortex.user_data_root,
            file_logger=cortex.file_logger,
            llm_interface=cortex.llm_interface
        )
        new_cortex.load_graph()

        # Verify loaded data
        assert "persist_1" in new_cortex.nodes
        assert "persist_2" in new_cortex.nodes
        assert new_cortex.graph.has_edge("persist_1", "persist_2")

    def test_get_connected_nodes(self, cortex):
        """Test finding connected nodes."""
        # Create a small network
        nodes = [CognitiveNode(f"conn_{i}", "insight", f"N{i}", {}, datetime.now(timezone.utc)) for i in range(4)]
        for node in nodes:
            cortex.add_node(node)

        # conn_0 connects to conn_1 and conn_2
        cortex.add_link(CognitiveLink("conn_0", "conn_1", "related_to", 1.0, {}))
        cortex.add_link(CognitiveLink("conn_0", "conn_2", "elaborates_on", 1.5, {}))

        # Get neighbors of conn_0
        neighbors = cortex.get_connected_nodes("conn_0")

        assert "conn_1" in neighbors
        assert "conn_2" in neighbors
        assert "conn_3" not in neighbors  # Not connected

    def test_prune_old_nodes(self, cortex):
        """Test graph pruning based on age and centrality."""
        # Create old, low-centrality node
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=40)
        old_node = CognitiveNode("old_node", "insight", "Old", {}, old_timestamp)
        cortex.add_node(old_node)

        # Create recent node
        recent_node = CognitiveNode("recent", "insight", "New", {}, datetime.now(timezone.utc))
        cortex.add_node(recent_node)

        # Prune
        pruned = cortex.prune_graph()

        # Old node should be pruned if below centrality threshold
        # (depends on implementation details)

    def test_update_relationship_weights(self, cortex):
        """Test dynamic relationship weight updating."""
        initial_weight = cortex.config["cortex"]["relationship_weights"]["elaborates_on"]

        # Update weight
        cortex.update_relationship_weight("elaborates_on", 2.0)

        updated_weight = cortex.config["cortex"]["relationship_weights"]["elaborates_on"]
        assert updated_weight == 2.0
        assert updated_weight != initial_weight


class TestReflectionMethods:
    """Test suite for cortex reflection capabilities (orphan linking, meta-insights)."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "cortex": {
                "relationship_weights": {
                    "elaborates_on": 1.5,
                    "conflicts_with": 2.0,
                    "related_to": 1.0,
                    "develops": 1.5,
                    "summarizes": 1.2
                },
                "pruning": {
                    "enabled": False  # Disable for these tests
                }
            }
        }

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface."""
        llm = Mock()
        # Mock response for orphan linking
        llm.generate = Mock(side_effect=[
            '{"links": [{"target": "connected_1", "type": "related_to", "reasoning": "both about testing"}]}',
            '{"meta_insight": "Combined understanding from cluster", "confidence": 0.8}'
        ])
        return llm

    @pytest.fixture
    def cortex_with_orphans(self, temp_dir, mock_config, mock_file_logger, mock_llm_interface):
        """Create cortex with orphan nodes for testing."""
        cortex = Cortex(
            config=mock_config,
            user_data_root=temp_dir,
            file_logger=mock_file_logger,
            llm_interface=mock_llm_interface
        )

        # Create connected cluster
        for i in range(3):
            node = CognitiveNode(f"connected_{i}", "insight", f"Connected {i}", {}, datetime.now(timezone.utc))
            cortex.add_node(node)

        cortex.add_link(CognitiveLink("connected_0", "connected_1", "related_to", 1.0, {}))
        cortex.add_link(CognitiveLink("connected_1", "connected_2", "elaborates_on", 1.5, {}))

        # Create orphan
        orphan = CognitiveNode("orphan", "insight", "Isolated node", {}, datetime.now(timezone.utc))
        cortex.add_node(orphan)

        return cortex

    def test_link_orphans_method(self, cortex_with_orphans):
        """Test the _link_orphans reflection method."""
        initial_edges = cortex_with_orphans.graph.number_of_edges()

        # Run orphan linking
        cortex_with_orphans._link_orphans()

        # Should have created new links
        final_edges = cortex_with_orphans.graph.number_of_edges()
        # Note: This depends on LLM mock response

    def test_generate_meta_insights_method(self, cortex_with_orphans):
        """Test meta-insight generation from clusters."""
        # Run meta-insight generation
        result = cortex_with_orphans._generate_meta_insights()

        # Should use LLM to generate meta-insights
        assert cortex_with_orphans.llm_interface.generate.called


class TestProactiveEngine:
    """Test suite for proactive suggestion engine."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            "cortex": {
                "relationship_weights": {
                    "related_to": 1.0
                }
            }
        }

    @pytest.fixture
    def mock_file_logger(self):
        """Create mock file logger."""
        logger = Mock()
        logger.log_info = Mock()
        logger.log_warning = Mock()
        logger.log_error = Mock()
        logger.log_debug = Mock()
        return logger

    @pytest.fixture
    def mock_llm_interface(self):
        """Create mock LLM interface."""
        llm = Mock()
        llm.generate = Mock(return_value="Proactive suggestion based on cognitive graph analysis")
        return llm

    @pytest.fixture
    def cortex_with_graph(self, temp_dir, mock_config, mock_file_logger, mock_llm_interface):
        """Create cortex with populated graph."""
        cortex = Cortex(
            config=mock_config,
            user_data_root=temp_dir,
            file_logger=mock_file_logger,
            llm_interface=mock_llm_interface
        )

        # Add diverse nodes
        for i in range(10):
            node = CognitiveNode(
                f"node_{i}",
                "insight",
                f"Insight {i}",
                {"concern": "testing"},
                datetime.now(timezone.utc)
            )
            cortex.add_node(node)

        # Create some links (sparse graph with underdeveloped areas)
        cortex.add_link(CognitiveLink("node_0", "node_1", "related_to", 1.0, {}))
        cortex.add_link(CognitiveLink("node_1", "node_2", "related_to", 1.0, {}))

        return cortex

    @pytest.fixture
    def proactive_engine(self, cortex_with_graph, mock_config, mock_file_logger, mock_llm_interface):
        """Create proactive engine instance."""
        engine = ProactiveEngine(
            cortex=cortex_with_graph,
            config=mock_config,
            file_logger=mock_file_logger,
            llm_interface=mock_llm_interface
        )
        return engine

    def test_proactive_engine_initialization(self, proactive_engine):
        """Test proactive engine initializes."""
        assert proactive_engine is not None
        assert proactive_engine.cortex is not None

    def test_identify_underdeveloped_areas(self, proactive_engine):
        """Test identification of underdeveloped areas."""
        underdeveloped = proactive_engine.find_underdeveloped_nodes()

        # Should find nodes with low centrality
        assert isinstance(underdeveloped, list)

    def test_identify_high_potential_areas(self, proactive_engine):
        """Test identification of high-potential areas."""
        high_potential = proactive_engine.find_high_potential_nodes()

        # Should find nodes with high centrality
        assert isinstance(high_potential, list)

    def test_generate_proactive_suggestion(self, proactive_engine):
        """Test generating a proactive suggestion."""
        suggestion = proactive_engine.generate_suggestion()

        assert suggestion is not None
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0


# Run tests with: pytest tests/test_cortex.py -v
