##Script function and purpose: Unit tests for Memory class
"""
Memory Unit Tests

Tests for the unified Memory class.
"""

from __future__ import annotations

import pytest

from jenova.memory import Memory, MemoryResult, MemoryType


##Class purpose: Test suite for Memory class
class TestMemory:
    """Tests for Memory class."""

    ##Method purpose: Test that adding content stores it retrievably
    def test_add_stores_content(self, episodic_memory: Memory) -> None:
        """Adding content stores it retrievably."""
        ##Step purpose: Add content
        content = "Python is a programming language"
        memory_id = episodic_memory.add(content, metadata={"source": "test"})

        ##Step purpose: Verify stored
        assert memory_id is not None
        result = episodic_memory.get(memory_id)
        assert result is not None
        assert result.content == content

    ##Method purpose: Test that search finds relevant content
    def test_search_finds_content(self, episodic_memory: Memory) -> None:
        """Search returns relevant content."""
        ##Step purpose: Add test content
        episodic_memory.add("Python is a programming language")
        episodic_memory.add("JavaScript runs in browsers")

        ##Step purpose: Search and verify
        results = episodic_memory.search("Python")
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    ##Method purpose: Test that search returns empty for no matches
    def test_search_returns_empty_for_no_matches(self, episodic_memory: Memory) -> None:
        """Search returns empty list when nothing matches."""
        ##Step purpose: Search empty memory
        results = episodic_memory.search("nonexistent query")
        assert results == []

    ##Method purpose: Test that search respects n_results limit
    def test_search_respects_n_results(self, episodic_memory: Memory) -> None:
        """Search limits results to n_results."""
        ##Step purpose: Add multiple documents
        for i in range(10):
            episodic_memory.add(f"Document {i} about Python programming")

        ##Step purpose: Search with limit
        results = episodic_memory.search("Python", n_results=3)
        assert len(results) <= 3

    ##Method purpose: Test that delete removes content
    def test_delete_removes_content(self, episodic_memory: Memory) -> None:
        """Delete removes content from memory."""
        ##Step purpose: Add and delete
        memory_id = episodic_memory.add("Test content")
        deleted = episodic_memory.delete(memory_id)

        ##Step purpose: Verify deleted
        assert deleted is True
        assert episodic_memory.get(memory_id) is None

    ##Method purpose: Test that delete returns false for missing ID
    def test_delete_returns_false_for_missing(self, episodic_memory: Memory) -> None:
        """Delete returns False for non-existent ID."""
        result = episodic_memory.delete("nonexistent-id")
        assert result is False

    ##Method purpose: Test that count returns correct number
    def test_count_returns_correct_number(self, episodic_memory: Memory) -> None:
        """Count returns number of memories."""
        ##Step purpose: Add some content
        episodic_memory.add("First")
        episodic_memory.add("Second")
        episodic_memory.add("Third")

        ##Step purpose: Verify count
        assert episodic_memory.count() == 3

    ##Method purpose: Test that clear removes all content
    def test_clear_removes_all(self, episodic_memory: Memory) -> None:
        """Clear removes all memories."""
        ##Step purpose: Add content
        episodic_memory.add("First")
        episodic_memory.add("Second")

        ##Step purpose: Clear and verify
        episodic_memory.clear()
        assert episodic_memory.count() == 0


##Class purpose: Test suite for MemoryResult
class TestMemoryResult:
    """Tests for MemoryResult dataclass."""

    ##Method purpose: Test MemoryResult is immutable
    def test_memory_result_is_immutable(self) -> None:
        """MemoryResult should be frozen (immutable)."""
        result = MemoryResult(
            id="test-id",
            content="Test content",
            score=0.95,
            memory_type=MemoryType.SEMANTIC,
            metadata={"key": "value"},
        )

        ##Step purpose: Verify immutability
        with pytest.raises(Exception):  # noqa: B017 FrozenInstanceError
            result.content = "New content"  # type: ignore
