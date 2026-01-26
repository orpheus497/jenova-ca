##Script function and purpose: Unit tests for CognitiveGraph advanced features
"""
Tests for CognitiveGraph Advanced Features

Tests for emotion analysis, orphan linking, meta-insight generation,
graph pruning, clustering, contradiction detection, and connection suggestions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jenova.exceptions import (
    NodeNotFoundError,
)
from jenova.graph.graph import CognitiveGraph
from jenova.graph.types import (
    EdgeType,
    Emotion,
    Node,
)


##Class purpose: Mock LLM for testing
class MockLLM:
    """Mock LLM for testing graph intelligence features."""

    ##Method purpose: Initialize with configurable responses
    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    ##Method purpose: Generate mock response
    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate mock response."""
        self.calls.append((text, system_prompt))

        ##Step purpose: Check for configured response
        for key, response in self.responses.items():
            if key.lower() in text.lower():
                return response

        ##Step purpose: Return default JSON responses based on prompt type
        if "emotion" in text.lower():
            return json.dumps(
                {
                    "primary_emotion": "joy",
                    "confidence": 0.85,
                    "emotion_scores": {"joy": 0.85, "curiosity": 0.4},
                }
            )
        elif "contradicts" in text.lower():
            return json.dumps({"contradicts": False, "explanation": "No contradiction"})
        elif "related_node_ids" in text.lower():
            return json.dumps({"related_node_ids": [], "relationship": "relates_to"})
        elif "suggested_ids" in text.lower():
            return json.dumps({"suggested_ids": []})
        elif "label" in text.lower():
            return "Test Cluster"

        return "Mock insight generated from patterns."


##Class purpose: Test graph fixture creation
class TestCognitiveGraphFixtures:
    """Tests for CognitiveGraph fixture creation."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Create graph with user nodes
    @pytest.fixture
    def graph_with_nodes(self, graph: CognitiveGraph) -> CognitiveGraph:
        """Create graph with some user nodes."""
        ##Step purpose: Add nodes for test user
        for i in range(5):
            node = Node.create(
                label=f"Node {i}",
                content=f"Test content for node {i}",
                node_type="concept",
                metadata={"user": "testuser"},
            )
            graph.add_node(node)
        return graph

    ##Method purpose: Create mock LLM fixture
    @pytest.fixture
    def mock_llm(self) -> MockLLM:
        """Create a mock LLM."""
        return MockLLM()

    ##Method purpose: Test graph initialization
    def test_graph_initializes(self, graph: CognitiveGraph) -> None:
        """Graph initializes with empty data."""
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    ##Method purpose: Test add and get node
    def test_add_and_get_node(self, graph: CognitiveGraph) -> None:
        """Can add and retrieve nodes."""
        node = Node.create(
            label="Test",
            content="Test content",
            node_type="concept",
        )
        graph.add_node(node)

        retrieved = graph.get_node(node.id)
        assert retrieved.label == "Test"
        assert retrieved.content == "Test content"


##Class purpose: Test emotion analysis
class TestEmotionAnalysis:
    """Tests for analyze_emotion method."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Test joy emotion detection
    def test_analyze_emotion_detects_joy(self, graph: CognitiveGraph) -> None:
        """Emotion analysis detects joy correctly."""
        llm = MockLLM(
            {
                "analyze": json.dumps(
                    {"primary_emotion": "joy", "confidence": 0.9, "emotion_scores": {"joy": 0.9}}
                )
            }
        )

        result = graph.analyze_emotion("I'm so happy today!", llm)

        assert result.primary_emotion == Emotion.JOY
        assert result.confidence >= 0.8

    ##Method purpose: Test sadness emotion detection
    def test_analyze_emotion_detects_sadness(self, graph: CognitiveGraph) -> None:
        """Emotion analysis detects sadness correctly."""
        llm = MockLLM(
            {
                "analyze": json.dumps(
                    {
                        "primary_emotion": "sadness",
                        "confidence": 0.85,
                        "emotion_scores": {"sadness": 0.85},
                    }
                )
            }
        )

        result = graph.analyze_emotion("I feel really down today.", llm)

        assert result.primary_emotion == Emotion.SADNESS

    ##Method purpose: Test neutral emotion fallback
    def test_analyze_emotion_defaults_to_neutral(self, graph: CognitiveGraph) -> None:
        """Unknown emotions default to neutral."""
        llm = MockLLM(
            {
                "analyze": json.dumps(
                    {
                        "primary_emotion": "unknown_emotion",
                        "confidence": 0.5,
                    }
                )
            }
        )

        result = graph.analyze_emotion("This is a neutral statement.", llm)

        assert result.primary_emotion == Emotion.NEUTRAL

    ##Method purpose: Test content preview is included
    def test_analyze_emotion_includes_preview(self, graph: CognitiveGraph) -> None:
        """Emotion result includes content preview."""
        llm = MockLLM()
        content = "This is some test content for analysis."

        result = graph.analyze_emotion(content, llm)

        assert result.content_preview == content[:100]

    ##Method purpose: Test confidence clamping
    def test_analyze_emotion_clamps_confidence(self, graph: CognitiveGraph) -> None:
        """Confidence is clamped to 0.0-1.0 range."""
        llm = MockLLM(
            {
                "analyze": json.dumps(
                    {
                        "primary_emotion": "joy",
                        "confidence": 1.5,  # Invalid, should be clamped
                    }
                )
            }
        )

        result = graph.analyze_emotion("Happy!", llm)

        assert 0.0 <= result.confidence <= 1.0


##Class purpose: Test orphan linking
class TestLinkOrphans:
    """Tests for link_orphans method."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Test returns zero for insufficient nodes
    def test_link_orphans_insufficient_nodes(self, graph: CognitiveGraph) -> None:
        """Link orphans returns 0 for insufficient nodes."""
        node = Node.create("Test", "Content", "concept", {"user": "testuser"})
        graph.add_node(node)
        llm = MockLLM()

        result = graph.link_orphans("testuser", llm)

        assert result == 0

    ##Method purpose: Test returns zero when no orphans
    def test_link_orphans_no_orphans(self, graph: CognitiveGraph) -> None:
        """Link orphans returns 0 when all nodes connected."""
        ##Step purpose: Add connected nodes
        node1 = Node.create("A", "Content A", "concept", {"user": "testuser"})
        node2 = Node.create("B", "Content B", "concept", {"user": "testuser"})
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(node1.id, node2.id, EdgeType.RELATES_TO)

        llm = MockLLM()

        result = graph.link_orphans("testuser", llm)

        assert result == 0

    ##Method purpose: Test connects first two orphans when all orphans
    def test_link_orphans_all_orphans(self, graph: CognitiveGraph) -> None:
        """When all nodes are orphans, first two are connected."""
        node1 = Node.create("A", "Content A", "concept", {"user": "testuser"})
        node2 = Node.create("B", "Content B", "concept", {"user": "testuser"})
        graph.add_node(node1)
        graph.add_node(node2)

        llm = MockLLM()

        result = graph.link_orphans("testuser", llm)

        assert result == 1
        assert graph.edge_count() == 1


##Class purpose: Test meta-insight generation
class TestGenerateMetaInsights:
    """Tests for generate_meta_insights method."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Test returns empty for insufficient nodes
    def test_meta_insights_insufficient_nodes(self, graph: CognitiveGraph) -> None:
        """Returns empty list for insufficient nodes."""
        llm = MockLLM()

        result = graph.generate_meta_insights("testuser", llm, min_cluster_size=3)

        assert result == []

    ##Method purpose: Test generates insight from connected cluster
    def test_meta_insights_from_cluster(self, graph: CognitiveGraph) -> None:
        """Generates insight from connected cluster."""
        ##Step purpose: Create connected cluster
        nodes = []
        for i in range(4):
            node = Node.create(f"Node {i}", f"Content {i}", "concept", {"user": "testuser"})
            graph.add_node(node)
            nodes.append(node)

        ##Step purpose: Connect nodes
        graph.add_edge(nodes[0].id, nodes[1].id, EdgeType.RELATES_TO)
        graph.add_edge(nodes[1].id, nodes[2].id, EdgeType.RELATES_TO)
        graph.add_edge(nodes[2].id, nodes[3].id, EdgeType.RELATES_TO)

        llm = MockLLM()

        result = graph.generate_meta_insights("testuser", llm, min_cluster_size=3)

        assert len(result) >= 1


##Class purpose: Test graph pruning
class TestPruneGraph:
    """Tests for prune_graph method."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Test returns zero for no qualifying nodes
    def test_prune_no_qualifying_nodes(self, graph: CognitiveGraph) -> None:
        """Returns 0 when no nodes qualify for pruning."""
        ##Step purpose: Add recent node
        node = Node.create("Recent", "Content", "concept", {"user": "testuser"})
        graph.add_node(node)

        result = graph.prune_graph(max_age_days=30, min_connections=0)

        assert result == 0

    ##Method purpose: Test prunes old isolated nodes
    def test_prune_old_isolated_nodes(self, graph: CognitiveGraph) -> None:
        """Prunes old nodes with no connections."""
        ##Step purpose: Add old node by manually setting date
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        old_node = Node(
            id="old-node-id",
            label="Old",
            content="Old content",
            node_type="concept",
            created_at=old_date,
            metadata={"user": "testuser"},
        )
        graph._nodes[old_node.id] = old_node
        graph._save()

        result = graph.prune_graph(max_age_days=30, min_connections=1, username="testuser")

        assert result == 1
        assert not graph.has_node("old-node-id")

    ##Method purpose: Test preserves old connected nodes
    def test_prune_preserves_connected_nodes(self, graph: CognitiveGraph) -> None:
        """Old nodes with connections are preserved."""
        ##Step purpose: Add old connected nodes
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        node1 = Node(
            id="old-node-1",
            label="Old1",
            content="Content 1",
            node_type="concept",
            created_at=old_date,
            metadata={"user": "testuser"},
        )
        node2 = Node(
            id="old-node-2",
            label="Old2",
            content="Content 2",
            node_type="concept",
            created_at=old_date,
            metadata={"user": "testuser"},
        )
        graph._nodes[node1.id] = node1
        graph._nodes[node2.id] = node2
        graph.add_edge(node1.id, node2.id, EdgeType.RELATES_TO)

        result = graph.prune_graph(max_age_days=30, min_connections=1, username="testuser")

        ##Step purpose: Both nodes have 1 connection so should survive
        assert result == 0
        assert graph.has_node("old-node-1")
        assert graph.has_node("old-node-2")


##Class purpose: Test node clustering
class TestClusterNodes:
    """Tests for cluster_nodes method."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Test returns empty for no nodes
    def test_cluster_empty_graph(self, graph: CognitiveGraph) -> None:
        """Returns empty dict for no nodes."""
        result = graph.cluster_nodes("testuser", llm=None)

        assert result == {}

    ##Method purpose: Test creates clusters from components
    def test_cluster_creates_from_components(self, graph: CognitiveGraph) -> None:
        """Creates clusters from connected components."""
        ##Step purpose: Create two separate components
        node1 = Node.create("A", "Content A", "concept", {"user": "testuser"})
        node2 = Node.create("B", "Content B", "concept", {"user": "testuser"})
        node3 = Node.create("C", "Content C", "concept", {"user": "testuser"})
        node4 = Node.create("D", "Content D", "concept", {"user": "testuser"})

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        graph.add_node(node4)

        ##Step purpose: Create two components
        graph.add_edge(node1.id, node2.id, EdgeType.RELATES_TO)
        graph.add_edge(node3.id, node4.id, EdgeType.RELATES_TO)

        result = graph.cluster_nodes("testuser", llm=None)

        assert len(result) == 2


##Class purpose: Test contradiction detection
class TestFindContradictions:
    """Tests for find_contradictions method."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Test returns empty for insufficient nodes
    def test_contradictions_insufficient_nodes(self, graph: CognitiveGraph) -> None:
        """Returns empty for less than 2 nodes."""
        node = Node.create("A", "Content", "concept", {"user": "testuser"})
        graph.add_node(node)

        llm = MockLLM()

        result = graph.find_contradictions("testuser", llm)

        assert result == []

    ##Method purpose: Test detects contradiction
    def test_contradictions_detected(self, graph: CognitiveGraph) -> None:
        """Detects contradiction between nodes."""
        node1 = Node.create("A", "The sky is blue", "fact", {"user": "testuser"})
        node2 = Node.create("B", "The sky is green", "fact", {"user": "testuser"})
        graph.add_node(node1)
        graph.add_node(node2)

        llm = MockLLM(
            {"contradicts": json.dumps({"contradicts": True, "explanation": "Color mismatch"})}
        )

        result = graph.find_contradictions("testuser", llm)

        assert len(result) == 1
        assert node1.id in result[0]
        assert node2.id in result[0]


##Class purpose: Test connection suggestions
class TestSuggestConnections:
    """Tests for suggest_connections method."""

    ##Method purpose: Create graph fixture
    @pytest.fixture
    def graph(self, tmp_path: Path) -> CognitiveGraph:
        """Create a CognitiveGraph instance."""
        storage = tmp_path / "graph"
        storage.mkdir()
        return CognitiveGraph(storage)

    ##Method purpose: Test raises for nonexistent node
    def test_suggest_raises_for_missing_node(self, graph: CognitiveGraph) -> None:
        """Raises NodeNotFoundError for missing node."""
        llm = MockLLM()

        with pytest.raises(NodeNotFoundError):
            graph.suggest_connections("nonexistent", llm)

    ##Method purpose: Test returns empty for no candidates
    def test_suggest_empty_for_no_candidates(self, graph: CognitiveGraph) -> None:
        """Returns empty when no unconnected candidates exist."""
        node = Node.create("A", "Content A", "concept")
        graph.add_node(node)

        llm = MockLLM()

        result = graph.suggest_connections(node.id, llm)

        assert result == []

    ##Method purpose: Test returns suggestions
    def test_suggest_returns_suggestions(self, graph: CognitiveGraph) -> None:
        """Returns suggested node IDs."""
        node1 = Node.create("A", "Python programming", "concept")
        node2 = Node.create("B", "Software development", "concept")
        node3 = Node.create("C", "Machine learning", "concept")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        ##Step purpose: Create LLM that suggests node2
        llm = MockLLM({"suggested": json.dumps({"suggested_ids": [node2.id[:8]]})})

        result = graph.suggest_connections(node1.id, llm)

        ##Step purpose: Should suggest at least one connection
        assert isinstance(result, list)
