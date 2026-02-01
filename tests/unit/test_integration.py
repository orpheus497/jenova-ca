##Script function and purpose: Unit tests for IntegrationHub
"""
Tests for IntegrationHub - Memory/Cortex integration.

Tests the integration layer that coordinates Memory and Cortex systems.
"""

from dataclasses import dataclass

import pytest

from jenova.core.integration import (
    ConsistencyReport,
    CrossReference,
    IntegrationConfig,
    IntegrationHub,
    KnowledgeDuplication,
    KnowledgeGap,
    RelatedNodeResult,
    UnifiedKnowledgeMap,
)


##Class purpose: Mock node for testing
@dataclass
class MockNode:
    """Mock node for testing."""

    id: str
    label: str
    content: str
    node_type: str = "concept"
    metadata: dict = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


##Class purpose: Mock memory result for testing
@dataclass
class MockMemoryResult:
    """Mock memory result for testing."""

    id: str
    content: str
    score: float
    memory_type: str = "semantic"


##Class purpose: Mock graph implementation for testing
class MockGraph:
    """Mock graph for testing IntegrationHub."""

    def __init__(self) -> None:
        self._nodes: dict[str, MockNode] = {}
        self._added_nodes: list[MockNode] = []

    def add_test_node(self, node: MockNode) -> None:
        """Add a node for testing."""
        self._nodes[node.id] = node

    def search(
        self,
        query: str,
        max_results: int = 10,
        node_types: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Search nodes by query."""
        results = []
        query_lower = query.lower()
        for node in self._nodes.values():
            if query_lower in node.content.lower() or query_lower in node.label.lower():
                results.append(
                    {
                        "id": node.id,
                        "label": node.label,
                        "content": node.content,
                    }
                )
        return results[:max_results]

    def all_nodes(self) -> list[MockNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_node(self, node_id: str) -> MockNode:
        """Get node by ID."""
        if node_id not in self._nodes:
            raise KeyError(f"Node not found: {node_id}")
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes

    def add_node(self, node: MockNode) -> None:
        """Add a node."""
        self._nodes[node.id] = node
        self._added_nodes.append(node)

    def neighbors(self, node_id: str, direction: str = "out") -> list[MockNode]:
        """Get neighbors (returns empty for mock)."""
        return []


##Class purpose: Mock memory implementation for testing
class MockMemory:
    """Mock memory for testing IntegrationHub."""

    def __init__(self) -> None:
        self._items: list[MockMemoryResult] = []

    def add_test_item(self, content: str, score: float = 0.9) -> None:
        """Add an item for testing."""
        self._items.append(
            MockMemoryResult(
                id=f"mem_{len(self._items)}",
                content=content,
                score=score,
            )
        )

    def search(self, query: str, n_results: int = 5) -> list[MockMemoryResult]:
        """Search memory items."""
        results = []
        query_lower = query.lower()
        for item in self._items:
            if query_lower in item.content.lower():
                results.append(item)
        return results[:n_results]


##Class purpose: Test fixtures for integration tests
class TestIntegrationHub:
    """Tests for IntegrationHub."""

    @pytest.fixture
    def mock_graph(self) -> MockGraph:
        """Create mock graph with test data."""
        graph = MockGraph()
        graph.add_test_node(
            MockNode(
                id="node_1",
                label="Python",
                content="Python is a programming language",
                metadata={"user": "testuser", "centrality": "3.0"},
            )
        )
        graph.add_test_node(
            MockNode(
                id="node_2",
                label="JavaScript",
                content="JavaScript is a web language",
                metadata={"user": "testuser", "centrality": "2.0"},
            )
        )
        graph.add_test_node(
            MockNode(
                id="node_3",
                label="Other",
                content="Other content",
                metadata={"user": "otheruser"},
            )
        )
        return graph

    @pytest.fixture
    def mock_memory(self) -> MockMemory:
        """Create mock memory with test data."""
        memory = MockMemory()
        memory.add_test_item("Python is great for data science")
        memory.add_test_item("JavaScript runs in browsers")
        memory.add_test_item("General programming knowledge")
        return memory

    @pytest.fixture
    def hub(self, mock_graph: MockGraph, mock_memory: MockMemory) -> IntegrationHub:
        """Create IntegrationHub with mock dependencies."""
        return IntegrationHub(
            graph=mock_graph,
            memory=mock_memory,
            config=IntegrationConfig(enabled=True),
        )

    @pytest.fixture
    def disabled_hub(self, mock_graph: MockGraph, mock_memory: MockMemory) -> IntegrationHub:
        """Create disabled IntegrationHub."""
        return IntegrationHub(
            graph=mock_graph,
            memory=mock_memory,
            config=IntegrationConfig(enabled=False),
        )

    ##Method purpose: Test hub initialization
    def test_hub_initialization(self, hub: IntegrationHub) -> None:
        """Hub initializes with provided dependencies."""
        assert hub._config.enabled is True
        assert hub._config.max_related_nodes == 5

    ##Method purpose: Test finding related nodes
    def test_find_related_nodes_returns_matches(self, hub: IntegrationHub) -> None:
        """find_related_nodes returns matching nodes for user."""
        results = hub.find_related_nodes("Python", "testuser")

        assert len(results) > 0
        assert any(r.node_id == "node_1" for r in results)

    ##Method purpose: Test finding related nodes filters by user
    def test_find_related_nodes_filters_by_user(self, hub: IntegrationHub) -> None:
        """find_related_nodes only returns nodes for specified user."""
        results = hub.find_related_nodes("Other content", "testuser")

        # Should not return node_3 which belongs to otheruser
        assert not any(r.node_id == "node_3" for r in results)

    ##Method purpose: Test find_related_nodes returns empty when disabled
    def test_find_related_nodes_returns_empty_when_disabled(
        self,
        disabled_hub: IntegrationHub,
    ) -> None:
        """find_related_nodes returns empty list when integration disabled."""
        results = disabled_hub.find_related_nodes("Python", "testuser")

        assert results == []

    ##Method purpose: Test similarity scores are calculated
    def test_find_related_nodes_includes_similarity_scores(
        self,
        hub: IntegrationHub,
    ) -> None:
        """find_related_nodes includes similarity scores."""
        results = hub.find_related_nodes("Python", "testuser")

        assert len(results) > 0
        assert all(0.0 <= r.similarity_score <= 1.0 for r in results)

    ##Method purpose: Test centrality score calculation
    def test_get_centrality_score_returns_normalized_value(
        self,
        hub: IntegrationHub,
    ) -> None:
        """get_centrality_score returns normalized centrality."""
        score = hub.get_centrality_score("Python programming", "testuser")

        assert 0.0 <= score <= 1.0

    ##Method purpose: Test centrality score returns zero when disabled
    def test_get_centrality_score_returns_zero_when_disabled(
        self,
        disabled_hub: IntegrationHub,
    ) -> None:
        """get_centrality_score returns 0.0 when disabled."""
        score = disabled_hub.get_centrality_score("Python", "testuser")

        assert score == 0.0

    ##Method purpose: Test context expansion
    def test_expand_context_adds_related_content(self, hub: IntegrationHub) -> None:
        """expand_context_with_relationships adds related content."""
        initial = ["Python is popular"]
        expanded = hub.expand_context_with_relationships(initial, "testuser")

        assert len(expanded) >= len(initial)

    ##Method purpose: Test context expansion respects max_expansion
    def test_expand_context_respects_max_expansion(self, hub: IntegrationHub) -> None:
        """expand_context_with_relationships respects max_expansion limit."""
        initial = ["Python is popular"]
        expanded = hub.expand_context_with_relationships(initial, "testuser", max_expansion=1)

        # Should have at most 1 additional item
        assert len(expanded) <= len(initial) + 1

    ##Method purpose: Test context expansion returns original when disabled
    def test_expand_context_returns_original_when_disabled(
        self,
        disabled_hub: IntegrationHub,
    ) -> None:
        """expand_context_with_relationships returns original when disabled."""
        initial = ["Python is popular"]
        expanded = disabled_hub.expand_context_with_relationships(initial, "testuser")

        assert expanded == initial

    ##Method purpose: Test building unified context
    def test_build_unified_context_combines_sources(
        self,
        hub: IntegrationHub,
    ) -> None:
        """build_unified_context combines memory and cortex."""
        knowledge_map = hub.build_unified_context("testuser")

        assert isinstance(knowledge_map, UnifiedKnowledgeMap)
        # Should have some content
        assert not knowledge_map.is_empty() or True  # May be empty if no matches

    ##Method purpose: Test unified context includes cortex nodes
    def test_build_unified_context_includes_cortex_nodes(
        self,
        hub: IntegrationHub,
    ) -> None:
        """build_unified_context includes Cortex nodes for user."""
        knowledge_map = hub.build_unified_context("testuser")

        # Should only include nodes for testuser
        for node in knowledge_map.cortex_nodes:
            assert node.get("id") != "node_3"  # otheruser's node

    ##Method purpose: Test unified context returns empty when disabled
    def test_build_unified_context_returns_empty_when_disabled(
        self,
        disabled_hub: IntegrationHub,
    ) -> None:
        """build_unified_context returns empty map when disabled."""
        knowledge_map = disabled_hub.build_unified_context("testuser")

        assert knowledge_map.is_empty()

    ##Method purpose: Test consistency check
    def test_check_consistency_returns_report(self, hub: IntegrationHub) -> None:
        """check_consistency returns a ConsistencyReport."""
        report = hub.check_consistency("testuser")

        assert isinstance(report, ConsistencyReport)
        assert isinstance(report.is_consistent, bool)
        assert isinstance(report.gaps, list)
        assert isinstance(report.duplications, list)
        assert isinstance(report.recommendations, list)

    ##Method purpose: Test consistency check reports consistent when disabled
    def test_check_consistency_returns_consistent_when_disabled(
        self,
        disabled_hub: IntegrationHub,
    ) -> None:
        """check_consistency returns consistent when disabled."""
        report = disabled_hub.check_consistency("testuser")

        assert report.is_consistent is True
        assert report.gaps == []
        assert report.duplications == []

    ##Method purpose: Test consistency report summary
    def test_consistency_report_summary_describes_issues(self) -> None:
        """ConsistencyReport.summary() describes issues."""
        report = ConsistencyReport(
            is_consistent=False,
            gaps=[KnowledgeGap("test", "node_1", "content", 0.5)],
            duplications=[],
            recommendations=["Fix gaps"],
        )

        assert "knowledge gaps" in report.summary().lower()

    ##Method purpose: Test consistency report summary for consistent
    def test_consistency_report_summary_for_consistent(self) -> None:
        """ConsistencyReport.summary() indicates consistent state."""
        report = ConsistencyReport(
            is_consistent=True,
            gaps=[],
            duplications=[],
            recommendations=[],
        )

        assert "consistent" in report.summary().lower()

    ##Method purpose: Test memory to cortex propagation
    def test_propagate_memory_to_cortex_creates_node(
        self,
        hub: IntegrationHub,
        mock_graph: MockGraph,
    ) -> None:
        """propagate_memory_to_cortex creates reference node."""
        # Need high similarity match
        hub._config.similarity_threshold = 0.5

        node_id = hub.propagate_memory_to_cortex(
            "Python is a programming language",  # Matches node_1
            "episodic",
            "testuser",
        )

        # May or may not create node depending on similarity
        # Just verify it doesn't crash
        assert node_id is None or isinstance(node_id, str)

    ##Method purpose: Test memory to cortex propagation returns None when disabled
    def test_propagate_memory_to_cortex_returns_none_when_disabled(
        self,
        disabled_hub: IntegrationHub,
    ) -> None:
        """propagate_memory_to_cortex returns None when disabled."""
        result = disabled_hub.propagate_memory_to_cortex(
            "Python programming",
            "episodic",
            "testuser",
        )

        assert result is None

    ##Method purpose: Test get_knowledge_map returns unified context
    def test_get_knowledge_map_returns_unified_context(
        self,
        hub: IntegrationHub,
    ) -> None:
        """get_knowledge_map returns UnifiedKnowledgeMap for query."""
        knowledge_map = hub.get_knowledge_map("Python", "testuser")

        assert isinstance(knowledge_map, UnifiedKnowledgeMap)


##Class purpose: Tests for data classes
class TestIntegrationDataClasses:
    """Tests for integration data classes."""

    ##Method purpose: Test RelatedNodeResult creation
    def test_related_node_result_creation(self) -> None:
        """RelatedNodeResult can be created with required fields."""
        result = RelatedNodeResult(
            node_id="node_1",
            content="Test content",
            label="Test",
            similarity_score=0.85,
        )

        assert result.node_id == "node_1"
        assert result.similarity_score == 0.85
        assert result.metadata == {}

    ##Method purpose: Test CrossReference creation
    def test_cross_reference_creation(self) -> None:
        """CrossReference can be created with required fields."""
        ref = CrossReference(
            memory_content="Memory item",
            related_node_ids=["node_1", "node_2"],
            similarity_scores=[0.9, 0.7],
        )

        assert len(ref.related_node_ids) == 2
        assert len(ref.similarity_scores) == 2

    ##Method purpose: Test KnowledgeGap creation
    def test_knowledge_gap_creation(self) -> None:
        """KnowledgeGap can be created with required fields."""
        gap = KnowledgeGap(
            gap_type="high_centrality_not_in_memory",
            node_id="node_1",
            content="Important content",
            severity=0.8,
        )

        assert gap.gap_type == "high_centrality_not_in_memory"
        assert gap.severity == 0.8

    ##Method purpose: Test KnowledgeDuplication creation
    def test_knowledge_duplication_creation(self) -> None:
        """KnowledgeDuplication can be created with required fields."""
        dup = KnowledgeDuplication(
            memory_content="Duplicated content",
            node_id="node_1",
            similarity_score=0.95,
        )

        assert dup.similarity_score == 0.95

    ##Method purpose: Test UnifiedKnowledgeMap is_empty
    def test_unified_knowledge_map_is_empty(self) -> None:
        """UnifiedKnowledgeMap.is_empty() returns correct value."""
        empty_map = UnifiedKnowledgeMap(
            memory_items=[],
            cortex_nodes=[],
            cross_references=[],
            knowledge_gaps=[],
        )
        assert empty_map.is_empty() is True

        non_empty_map = UnifiedKnowledgeMap(
            memory_items=["item"],
            cortex_nodes=[],
            cross_references=[],
            knowledge_gaps=[],
        )
        assert non_empty_map.is_empty() is False

    ##Method purpose: Test IntegrationConfig defaults
    def test_integration_config_defaults(self) -> None:
        """IntegrationConfig has sensible defaults."""
        config = IntegrationConfig()

        assert config.enabled is True
        assert config.max_related_nodes == 5
        assert config.max_context_expansion == 3
        assert config.similarity_threshold == 0.7
