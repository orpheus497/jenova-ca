##Script function and purpose: Unit tests for Assumptions module
"""
Assumptions Unit Tests

Tests for the AssumptionManager, Assumption types, and related functionality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jenova.assumptions import (
    Assumption,
    AssumptionManager,
    AssumptionStatus,
    AssumptionStore,
)
from jenova.exceptions import AssumptionDuplicateError, AssumptionNotFoundError
from jenova.graph.types import Node
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
        self.responses = responses or ["true"]
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


##Class purpose: Test suite for Assumption type
class TestAssumption:
    """Tests for Assumption dataclass."""

    ##Method purpose: Test assumption creation
    def test_assumption_creation(self) -> None:
        """Assumption can be created with required fields."""
        assumption = Assumption(
            content="User prefers dark mode",
            username="testuser",
            status=AssumptionStatus.UNVERIFIED,
        )

        assert assumption.content == "User prefers dark mode"
        assert assumption.username == "testuser"
        assert assumption.status == AssumptionStatus.UNVERIFIED
        assert assumption.cortex_id is None

    ##Method purpose: Test assumption is frozen
    def test_assumption_is_frozen(self) -> None:
        """Assumption should be immutable."""
        assumption = Assumption(
            content="Test",
            username="testuser",
            status=AssumptionStatus.UNVERIFIED,
        )

        with pytest.raises(Exception):  # noqa: B017 FrozenInstanceError
            assumption.content = "Modified"  # type: ignore

    ##Method purpose: Test assumption to_dict
    def test_assumption_to_dict(self) -> None:
        """Assumption serializes to dictionary."""
        assumption = Assumption(
            content="Test content",
            username="testuser",
            status=AssumptionStatus.TRUE,
            cortex_id="test-id-123",
        )

        data = assumption.to_dict()

        assert data["content"] == "Test content"
        assert data["username"] == "testuser"
        assert data["status"] == "true"
        assert data["cortex_id"] == "test-id-123"

    ##Method purpose: Test assumption from_dict
    def test_assumption_from_dict(self) -> None:
        """Assumption deserializes from dictionary."""
        data = {
            "content": "Loaded content",
            "username": "loadeduser",
            "status": "false",
            "timestamp": "2024-01-01T12:00:00",
            "cortex_id": "loaded-id",
        }

        assumption = Assumption.from_dict(data)

        assert assumption.content == "Loaded content"
        assert assumption.username == "loadeduser"
        assert assumption.status == AssumptionStatus.FALSE
        assert assumption.cortex_id == "loaded-id"

    ##Method purpose: Test with_updates creates new instance
    def test_with_updates_creates_new_instance(self) -> None:
        """with_updates returns new Assumption with changes."""
        original = Assumption(
            content="Original",
            username="testuser",
            status=AssumptionStatus.UNVERIFIED,
        )

        updated = original.with_updates(content="Updated")

        assert original.content == "Original"
        assert updated.content == "Updated"
        assert updated.username == original.username


##Class purpose: Test suite for AssumptionStore
class TestAssumptionStore:
    """Tests for AssumptionStore container."""

    ##Method purpose: Test store creation
    def test_store_creation(self) -> None:
        """AssumptionStore can be created empty."""
        store = AssumptionStore()

        assert store.unverified == []
        assert store.verified_true == []
        assert store.verified_false == []

    ##Method purpose: Test find_by_content finds existing
    def test_find_by_content_finds_existing(self) -> None:
        """find_by_content locates assumption in any list."""
        assumption = Assumption(
            content="Test content",
            username="testuser",
            status=AssumptionStatus.TRUE,
        )

        store = AssumptionStore(verified_true=[assumption])

        result = store.find_by_content("Test content", "testuser")

        assert result is not None
        found_assumption, status = result
        assert found_assumption.content == "Test content"
        assert status == AssumptionStatus.TRUE

    ##Method purpose: Test find_by_content returns None for missing
    def test_find_by_content_returns_none_for_missing(self) -> None:
        """find_by_content returns None when not found."""
        store = AssumptionStore()

        result = store.find_by_content("Nonexistent", "testuser")

        assert result is None

    ##Method purpose: Test all_assumptions returns flat list
    def test_all_assumptions_returns_flat_list(self) -> None:
        """all_assumptions returns all assumptions from all lists."""
        store = AssumptionStore(
            unverified=[Assumption("A", "u", AssumptionStatus.UNVERIFIED)],
            verified_true=[Assumption("B", "u", AssumptionStatus.TRUE)],
            verified_false=[Assumption("C", "u", AssumptionStatus.FALSE)],
        )

        all_items = store.all_assumptions()

        assert len(all_items) == 3
        contents = {a.content for a in all_items}
        assert contents == {"A", "B", "C"}


##Class purpose: Test suite for AssumptionManager
class TestAssumptionManager:
    """Tests for AssumptionManager."""

    ##Method purpose: Fixture for assumption manager
    @pytest.fixture
    def manager(self, tmp_storage: Path) -> AssumptionManager:
        """Create assumption manager with mocks."""
        return AssumptionManager(
            storage_path=tmp_storage / "assumptions",
            graph=MockGraph(),
            llm=MockLLM(),
        )

    ##Method purpose: Test add_assumption creates new assumption
    def test_add_assumption_creates_new(self, manager: AssumptionManager) -> None:
        """add_assumption creates and stores new assumption."""
        cortex_id = manager.add_assumption(
            content="User prefers dark mode",
            username="testuser",
        )

        assert cortex_id is not None
        store = manager.get_all_assumptions()
        assert len(store.unverified) == 1
        assert store.unverified[0].content == "User prefers dark mode"

    ##Method purpose: Test add_assumption rejects duplicates
    def test_add_assumption_rejects_duplicates(self, manager: AssumptionManager) -> None:
        """add_assumption raises error for duplicate content."""
        manager.add_assumption("Duplicate", "testuser")

        with pytest.raises(AssumptionDuplicateError):
            manager.add_assumption("Duplicate", "testuser")

    ##Method purpose: Test add_assumption links to graph
    def test_add_assumption_links_to_graph(self, tmp_storage: Path) -> None:
        """add_assumption creates node in cognitive graph."""
        graph = MockGraph()
        manager = AssumptionManager(
            storage_path=tmp_storage / "assumptions",
            graph=graph,
            llm=MockLLM(),
        )

        cortex_id = manager.add_assumption("Test", "testuser")

        assert graph.has_node(cortex_id)

    ##Method purpose: Test get_assumption_to_verify returns first unverified
    def test_get_assumption_to_verify_returns_unverified(
        self,
        tmp_storage: Path,
    ) -> None:
        """get_assumption_to_verify returns first unverified with question."""
        manager = AssumptionManager(
            storage_path=tmp_storage / "assumptions",
            graph=MockGraph(),
            llm=MockLLM(responses=["Is this true?"]),
        )

        manager.add_assumption("First unverified", "testuser")
        manager.add_assumption("Second unverified", "testuser")

        result = manager.get_assumption_to_verify("testuser")

        assert result is not None
        assumption, question = result
        assert assumption.content == "First unverified"
        assert question == "Is this true?"

    ##Method purpose: Test get_assumption_to_verify returns None when empty
    def test_get_assumption_to_verify_returns_none_when_empty(
        self,
        manager: AssumptionManager,
    ) -> None:
        """get_assumption_to_verify returns None when no unverified."""
        result = manager.get_assumption_to_verify("testuser")
        assert result is None

    ##Method purpose: Test resolve_assumption moves to true
    def test_resolve_assumption_moves_to_true(
        self,
        tmp_storage: Path,
    ) -> None:
        """resolve_assumption moves assumption to verified_true."""
        manager = AssumptionManager(
            storage_path=tmp_storage / "assumptions",
            graph=MockGraph(),
            llm=MockLLM(responses=["question?", "true"]),
        )

        manager.add_assumption("Test assumption", "testuser")
        assumption = manager.get_all_assumptions().unverified[0]

        status = manager.resolve_assumption(assumption, "Yes it's true", "testuser")

        assert status == AssumptionStatus.TRUE
        store = manager.get_all_assumptions()
        assert len(store.unverified) == 0
        assert len(store.verified_true) == 1

    ##Method purpose: Test resolve_assumption creates insight node when confirmed
    def test_resolve_assumption_creates_insight_when_true(
        self,
        tmp_storage: Path,
    ) -> None:
        """resolve_assumption creates insight node in graph when confirmed."""
        graph = MockGraph()
        manager = AssumptionManager(
            storage_path=tmp_storage / "assumptions",
            graph=graph,
            llm=MockLLM(responses=["question?", "true"]),
        )

        manager.add_assumption("User prefers dark mode", "testuser")
        assumption = manager.get_all_assumptions().unverified[0]

        ##Step purpose: Count nodes before resolution
        nodes_before = len(graph.nodes)

        manager.resolve_assumption(assumption, "Yes, I do prefer dark mode", "testuser")

        ##Step purpose: Verify insight node was created
        nodes_after = len(graph.nodes)
        assert nodes_after == nodes_before + 1

        ##Step purpose: Find the insight node
        insight_nodes = [n for n in graph.nodes.values() if n.node_type == "insight"]
        assert len(insight_nodes) == 1
        assert "dark mode" in insight_nodes[0].content

    ##Method purpose: Test resolve_assumption moves to false
    def test_resolve_assumption_moves_to_false(
        self,
        tmp_storage: Path,
    ) -> None:
        """resolve_assumption moves assumption to verified_false."""
        manager = AssumptionManager(
            storage_path=tmp_storage / "assumptions",
            graph=MockGraph(),
            llm=MockLLM(responses=["question?", "false"]),
        )

        manager.add_assumption("Test assumption", "testuser")
        assumption = manager.get_all_assumptions().unverified[0]

        status = manager.resolve_assumption(assumption, "No it's false", "testuser")

        assert status == AssumptionStatus.FALSE
        store = manager.get_all_assumptions()
        assert len(store.unverified) == 0
        assert len(store.verified_false) == 1

    ##Method purpose: Test resolve_assumption raises for not found
    def test_resolve_assumption_raises_for_not_found(
        self,
        manager: AssumptionManager,
    ) -> None:
        """resolve_assumption raises error for non-existent assumption."""
        fake_assumption = Assumption(
            content="Nonexistent",
            username="testuser",
            status=AssumptionStatus.UNVERIFIED,
        )

        with pytest.raises(AssumptionNotFoundError):
            manager.resolve_assumption(fake_assumption, "response", "testuser")

    ##Method purpose: Test update_assumption modifies content
    def test_update_assumption_modifies_content(
        self,
        manager: AssumptionManager,
    ) -> None:
        """update_assumption changes assumption content."""
        manager.add_assumption("Original content", "testuser")

        updated = manager.update_assumption(
            old_content="Original content",
            new_content="Updated content",
            username="testuser",
        )

        assert updated.content == "Updated content"
        store = manager.get_all_assumptions()
        assert store.unverified[0].content == "Updated content"

    ##Method purpose: Test update_assumption raises for not found
    def test_update_assumption_raises_for_not_found(
        self,
        manager: AssumptionManager,
    ) -> None:
        """update_assumption raises error for non-existent assumption."""
        with pytest.raises(AssumptionNotFoundError):
            manager.update_assumption("Nonexistent", "New", "testuser")

    ##Method purpose: Test unverified_count returns correct count
    def test_unverified_count_returns_correct_count(
        self,
        manager: AssumptionManager,
    ) -> None:
        """unverified_count returns count for specific user."""
        manager.add_assumption("First", "testuser")
        manager.add_assumption("Second", "testuser")
        manager.add_assumption("Other user", "otheruser")

        assert manager.unverified_count("testuser") == 2
        assert manager.unverified_count("otheruser") == 1
        assert manager.unverified_count("nobody") == 0

    ##Method purpose: Test persistence across instances
    def test_persistence_across_instances(self, tmp_storage: Path) -> None:
        """Assumptions persist across manager instances."""
        storage = tmp_storage / "assumptions"
        graph = MockGraph()
        llm = MockLLM()

        ##Step purpose: Create and add with first instance
        manager1 = AssumptionManager(storage, graph, llm)
        manager1.add_assumption("Persistent assumption", "testuser")

        ##Step purpose: Create new instance and verify
        manager2 = AssumptionManager(storage, MockGraph(), MockLLM())
        store = manager2.get_all_assumptions()

        assert len(store.unverified) == 1
        assert store.unverified[0].content == "Persistent assumption"
