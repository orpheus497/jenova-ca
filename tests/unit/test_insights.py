##Script function and purpose: Unit tests for Insights module
"""
Insights Unit Tests

Tests for the InsightManager, ConcernManager, and related functionality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jenova.exceptions import ConcernError, InsightSaveError
from jenova.graph.types import Node
from jenova.insights import (
    Concern,
    ConcernManager,
    Insight,
    InsightManager,
)
from jenova.llm.types import GenerationParams


##Class purpose: Mock graph for testing
class MockGraph:
    """Mock cognitive graph for testing."""
    
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
    
    def add_node(self, node: Node) -> None:
        """Add a node to the mock graph."""
        self.nodes[node.id] = node
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self.nodes


##Class purpose: Mock LLM for testing
class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["test_topic"]
        self.call_count = 0
    
    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        params: GenerationParams | None = None,
    ) -> str:
        """Return mock response."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


##Class purpose: Mock memory search for testing
class MockMemorySearch:
    """Mock memory search for testing."""
    
    def __init__(self, results: list[tuple[str, float]] | None = None) -> None:
        self.results = results or []
    
    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[tuple[str, float]]:
        """Return mock search results."""
        return self.results[:n_results]


##Class purpose: Test suite for Insight type
class TestInsight:
    """Tests for Insight dataclass."""
    
    ##Method purpose: Test insight creation
    def test_insight_creation(self) -> None:
        """Insight can be created with required fields."""
        insight = Insight(
            content="User enjoys programming",
            username="testuser",
            topic="programming",
        )
        
        assert insight.content == "User enjoys programming"
        assert insight.username == "testuser"
        assert insight.topic == "programming"
        assert insight.cortex_id is None
    
    ##Method purpose: Test insight is frozen
    def test_insight_is_frozen(self) -> None:
        """Insight should be immutable."""
        insight = Insight(
            content="Test",
            username="testuser",
            topic="general",
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            insight.content = "Modified"  # type: ignore
    
    ##Method purpose: Test insight to_dict
    def test_insight_to_dict(self) -> None:
        """Insight serializes to dictionary."""
        insight = Insight(
            content="Test content",
            username="testuser",
            topic="testing",
            cortex_id="test-id-123",
            related_concerns=["coding", "qa"],
        )
        
        data = insight.to_dict()
        
        assert data["content"] == "Test content"
        assert data["user"] == "testuser"
        assert data["topic"] == "testing"
        assert data["cortex_id"] == "test-id-123"
        assert data["related_concerns"] == ["coding", "qa"]
    
    ##Method purpose: Test insight from_dict
    def test_insight_from_dict(self) -> None:
        """Insight deserializes from dictionary."""
        data = {
            "content": "Loaded content",
            "user": "loadeduser",
            "topic": "loaded_topic",
            "timestamp": "2024-01-01T12:00:00",
            "cortex_id": "loaded-id",
            "related_concerns": ["related"],
        }
        
        insight = Insight.from_dict(data)
        
        assert insight.content == "Loaded content"
        assert insight.username == "loadeduser"
        assert insight.topic == "loaded_topic"
        assert insight.cortex_id == "loaded-id"
        assert insight.related_concerns == ["related"]
    
    ##Method purpose: Test with_updates creates new instance
    def test_with_updates_creates_new_instance(self) -> None:
        """with_updates returns new Insight with changes."""
        original = Insight(
            content="Original",
            username="testuser",
            topic="original_topic",
        )
        
        updated = original.with_updates(content="Updated", topic="new_topic")
        
        assert original.content == "Original"
        assert original.topic == "original_topic"
        assert updated.content == "Updated"
        assert updated.topic == "new_topic"
        assert updated.username == original.username


##Class purpose: Test suite for Concern type
class TestConcern:
    """Tests for Concern dataclass."""
    
    ##Method purpose: Test concern creation
    def test_concern_creation(self) -> None:
        """Concern can be created with required fields."""
        concern = Concern(
            name="programming",
            description="Things about coding",
        )
        
        assert concern.name == "programming"
        assert concern.description == "Things about coding"
        assert concern.related_concerns == []
    
    ##Method purpose: Test concern to_dict
    def test_concern_to_dict(self) -> None:
        """Concern serializes to dictionary."""
        concern = Concern(
            name="testing",
            description="Quality assurance",
            related_concerns=["development", "ci"],
        )
        
        data = concern.to_dict()
        
        assert data["description"] == "Quality assurance"
        assert data["related_concerns"] == ["development", "ci"]
    
    ##Method purpose: Test concern from_dict
    def test_concern_from_dict(self) -> None:
        """Concern deserializes from dictionary."""
        data = {
            "description": "Loaded description",
            "related_concerns": ["related1", "related2"],
        }
        
        concern = Concern.from_dict("loaded_name", data)
        
        assert concern.name == "loaded_name"
        assert concern.description == "Loaded description"
        assert concern.related_concerns == ["related1", "related2"]


##Class purpose: Test suite for ConcernManager
class TestConcernManager:
    """Tests for ConcernManager."""
    
    ##Method purpose: Fixture for concern manager
    @pytest.fixture
    def manager(self, tmp_storage: Path) -> ConcernManager:
        """Create concern manager with mocks."""
        return ConcernManager(
            insights_root=tmp_storage / "insights",
            llm=MockLLM(),
        )
    
    ##Method purpose: Test get_all_concerns returns empty initially
    def test_get_all_concerns_returns_empty_initially(
        self,
        manager: ConcernManager,
    ) -> None:
        """get_all_concerns returns empty list for new manager."""
        concerns = manager.get_all_concerns()
        assert concerns == []
    
    ##Method purpose: Test add_concern creates concern
    def test_add_concern_creates_concern(self, manager: ConcernManager) -> None:
        """add_concern creates and stores concern."""
        concern = manager.add_concern(
            name="test_topic",
            description="Test description",
        )
        
        assert concern.name == "test_topic"
        assert "test_topic" in manager.get_all_concerns()
    
    ##Method purpose: Test add_concern sanitizes name
    def test_add_concern_sanitizes_name(self, manager: ConcernManager) -> None:
        """add_concern converts spaces to underscores and lowercases."""
        concern = manager.add_concern(
            name="Test Topic Name",
            description="Description",
        )
        
        assert concern.name == "test_topic_name"
    
    ##Method purpose: Test get_concern returns existing
    def test_get_concern_returns_existing(self, manager: ConcernManager) -> None:
        """get_concern returns concern by name."""
        manager.add_concern("existing", "Exists")
        
        concern = manager.get_concern("existing")
        
        assert concern is not None
        assert concern.description == "Exists"
    
    ##Method purpose: Test get_concern returns None for missing
    def test_get_concern_returns_none_for_missing(
        self,
        manager: ConcernManager,
    ) -> None:
        """get_concern returns None for non-existent concern."""
        concern = manager.get_concern("nonexistent")
        assert concern is None
    
    ##Method purpose: Test find_or_create_concern creates new when no topics
    def test_find_or_create_concern_creates_when_empty(
        self,
        tmp_storage: Path,
    ) -> None:
        """find_or_create_concern creates new when no existing topics."""
        manager = ConcernManager(
            insights_root=tmp_storage / "insights",
            llm=MockLLM(responses=["new_topic"]),
        )
        
        topic = manager.find_or_create_concern("Some insight content")
        
        assert topic == "new_topic"
        assert "new_topic" in manager.get_all_concerns()
    
    ##Method purpose: Test find_or_create_concern returns existing match
    def test_find_or_create_concern_returns_existing(
        self,
        tmp_storage: Path,
    ) -> None:
        """find_or_create_concern returns existing topic when matched."""
        manager = ConcernManager(
            insights_root=tmp_storage / "insights",
            llm=MockLLM(responses=["programming"]),
        )
        manager.add_concern("programming", "Code stuff")
        
        topic = manager.find_or_create_concern(
            "User writes Python code",
            existing_topics=["programming", "gaming"],
        )
        
        assert topic == "programming"
    
    ##Method purpose: Test link_concerns creates bidirectional link
    def test_link_concerns_creates_bidirectional_link(
        self,
        manager: ConcernManager,
    ) -> None:
        """link_concerns creates relationship in both directions."""
        manager.add_concern("topic_a", "First topic")
        manager.add_concern("topic_b", "Second topic")
        
        manager.link_concerns("topic_a", "topic_b")
        
        concern_a = manager.get_concern("topic_a")
        concern_b = manager.get_concern("topic_b")
        
        assert concern_a is not None
        assert concern_b is not None
        assert "topic_b" in concern_a.related_concerns
        assert "topic_a" in concern_b.related_concerns
    
    ##Method purpose: Test link_concerns raises for missing concern
    def test_link_concerns_raises_for_missing(
        self,
        manager: ConcernManager,
    ) -> None:
        """link_concerns raises ConcernError for non-existent concern."""
        manager.add_concern("existing", "Exists")
        
        with pytest.raises(ConcernError):
            manager.link_concerns("existing", "nonexistent")
    
    ##Method purpose: Test persistence across instances
    def test_persistence_across_instances(self, tmp_storage: Path) -> None:
        """Concerns persist across manager instances."""
        storage = tmp_storage / "insights"
        
        manager1 = ConcernManager(storage, MockLLM())
        manager1.add_concern("persistent", "Persists")
        
        manager2 = ConcernManager(storage, MockLLM())
        
        assert "persistent" in manager2.get_all_concerns()


##Class purpose: Test suite for InsightManager
class TestInsightManager:
    """Tests for InsightManager."""
    
    ##Method purpose: Fixture for insight manager
    @pytest.fixture
    def manager(self, tmp_storage: Path) -> InsightManager:
        """Create insight manager with mocks."""
        return InsightManager(
            insights_root=tmp_storage / "insights",
            graph=MockGraph(),
            llm=MockLLM(responses=["auto_topic"]),
        )
    
    ##Method purpose: Test save_insight creates insight
    def test_save_insight_creates_insight(self, manager: InsightManager) -> None:
        """save_insight creates and stores insight."""
        insight = manager.save_insight(
            content="User prefers Python",
            username="testuser",
            topic="programming",
        )
        
        assert insight.content == "User prefers Python"
        assert insight.username == "testuser"
        assert insight.topic == "programming"
        assert insight.cortex_id is not None
    
    ##Method purpose: Test save_insight auto-classifies topic
    def test_save_insight_auto_classifies_topic(
        self,
        tmp_storage: Path,
    ) -> None:
        """save_insight auto-classifies topic when not provided."""
        manager = InsightManager(
            insights_root=tmp_storage / "insights",
            graph=MockGraph(),
            llm=MockLLM(responses=["auto_topic"]),
        )
        
        insight = manager.save_insight(
            content="User likes coding",
            username="testuser",
        )
        
        assert insight.topic == "auto_topic"
    
    ##Method purpose: Test save_insight creates graph node
    def test_save_insight_creates_graph_node(self, tmp_storage: Path) -> None:
        """save_insight creates node in cognitive graph."""
        graph = MockGraph()
        manager = InsightManager(
            insights_root=tmp_storage / "insights",
            graph=graph,
            llm=MockLLM(),
        )
        
        insight = manager.save_insight("Test", "testuser", topic="test")
        
        assert insight.cortex_id is not None
        assert graph.has_node(insight.cortex_id)
    
    ##Method purpose: Test save_insight persists to file
    def test_save_insight_persists_to_file(
        self,
        tmp_storage: Path,
        manager: InsightManager,
    ) -> None:
        """save_insight creates JSON file on disk."""
        manager.save_insight(
            content="Persisted insight",
            username="testuser",
            topic="persistence",
        )
        
        ##Step purpose: Verify file exists
        topic_dir = tmp_storage / "insights" / "testuser" / "persistence"
        insight_files = list(topic_dir.glob("insight_*.json"))
        
        assert len(insight_files) == 1
        
        ##Step purpose: Verify content
        with open(insight_files[0]) as f:
            data = json.load(f)
        
        assert data["content"] == "Persisted insight"
    
    ##Method purpose: Test get_all_insights returns all for user
    def test_get_all_insights_returns_all_for_user(
        self,
        manager: InsightManager,
    ) -> None:
        """get_all_insights returns all insights for a user."""
        manager.save_insight("First", "testuser", topic="topic1")
        manager.save_insight("Second", "testuser", topic="topic2")
        manager.save_insight("Other user", "otheruser", topic="topic1")
        
        insights = manager.get_all_insights("testuser")
        
        assert len(insights) == 2
        contents = {i.content for i in insights}
        assert contents == {"First", "Second"}
    
    ##Method purpose: Test get_all_insights returns empty for unknown user
    def test_get_all_insights_returns_empty_for_unknown_user(
        self,
        manager: InsightManager,
    ) -> None:
        """get_all_insights returns empty list for unknown user."""
        insights = manager.get_all_insights("unknownuser")
        assert insights == []
    
    ##Method purpose: Test get_insights_by_topic filters correctly
    def test_get_insights_by_topic_filters_correctly(
        self,
        manager: InsightManager,
    ) -> None:
        """get_insights_by_topic returns only matching topic."""
        manager.save_insight("Programming insight", "testuser", topic="programming")
        manager.save_insight("Gaming insight", "testuser", topic="gaming")
        
        insights = manager.get_insights_by_topic("testuser", "programming")
        
        assert len(insights) == 1
        assert insights[0].content == "Programming insight"
    
    ##Method purpose: Test get_user_topics returns all topics
    def test_get_user_topics_returns_all_topics(
        self,
        manager: InsightManager,
    ) -> None:
        """get_user_topics returns all topics for user."""
        manager.save_insight("A", "testuser", topic="topic1")
        manager.save_insight("B", "testuser", topic="topic2")
        manager.save_insight("C", "testuser", topic="topic3")
        
        topics = manager.get_user_topics("testuser")
        
        assert set(topics) == {"topic1", "topic2", "topic3"}
    
    ##Method purpose: Test get_latest_insight_id returns most recent
    def test_get_latest_insight_id_returns_most_recent(
        self,
        manager: InsightManager,
    ) -> None:
        """get_latest_insight_id returns most recently saved insight ID."""
        manager.save_insight("First", "testuser", topic="t1")
        second = manager.save_insight("Second", "testuser", topic="t2")
        
        latest_id = manager.get_latest_insight_id("testuser")
        
        ##Step purpose: The latest should be the second one saved
        assert latest_id == second.cortex_id
    
    ##Method purpose: Test get_latest_insight_id returns None for unknown user
    def test_get_latest_insight_id_returns_none_for_unknown(
        self,
        manager: InsightManager,
    ) -> None:
        """get_latest_insight_id returns None for unknown user."""
        latest_id = manager.get_latest_insight_id("unknownuser")
        assert latest_id is None
    
    ##Method purpose: Test get_relevant_insights uses memory search
    def test_get_relevant_insights_uses_memory_search(
        self,
        tmp_storage: Path,
    ) -> None:
        """get_relevant_insights uses memory search when configured."""
        memory_search = MockMemorySearch(results=[
            ("Relevant insight 1", 0.1),
            ("Relevant insight 2", 0.2),
        ])
        
        manager = InsightManager(
            insights_root=tmp_storage / "insights",
            graph=MockGraph(),
            llm=MockLLM(),
            memory_search=memory_search,
        )
        
        results = manager.get_relevant_insights("query", "testuser", max_insights=2)
        
        assert len(results) == 2
        assert results[0] == "Relevant insight 1"
    
    ##Method purpose: Test get_relevant_insights returns empty without search
    def test_get_relevant_insights_returns_empty_without_search(
        self,
        manager: InsightManager,
    ) -> None:
        """get_relevant_insights returns empty without memory search."""
        results = manager.get_relevant_insights("query", "testuser")
        assert results == []
    
    ##Method purpose: Test concern_manager property returns manager
    def test_concern_manager_property(self, manager: InsightManager) -> None:
        """concern_manager property returns ConcernManager instance."""
        concern_mgr = manager.concern_manager
        
        assert isinstance(concern_mgr, ConcernManager)
