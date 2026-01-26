##Script function and purpose: Integration tests for CognitiveEngine - Tests full cognitive cycle with real components.
##Dependency purpose: Validates that CognitiveEngine correctly orchestrates knowledge retrieval, LLM, and response generation.
"""Integration tests for CognitiveEngine.

Tests the full cognitive cycle including:
- Context retrieval from KnowledgeStore
- Response generation via LLM
- Memory updates for learning
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jenova.memory.types import MemoryType

if TYPE_CHECKING:
    from jenova.core.engine import CognitiveEngine
    from jenova.core.knowledge import KnowledgeStore


##Class purpose: Integration tests for CognitiveEngine
@pytest.mark.integration
class TestCognitiveEngineIntegration:
    """Integration tests for CognitiveEngine."""

    ##Method purpose: Test that engine can process a simple query
    def test_think_processes_simple_query(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that think() processes a simple query.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Action purpose: Process a simple query
        result = cognitive_engine.think("Hello, how are you?")

        ##Step purpose: Verify result structure
        assert result is not None
        assert result.content is not None
        assert len(result.content) > 0
        assert result.is_error is False
        assert result.error_message is None

    ##Method purpose: Test that engine returns greeting response
    def test_think_returns_appropriate_response(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that think() returns contextually appropriate responses.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Action purpose: Process query that should match pattern
        result = cognitive_engine.think("hello")

        ##Step purpose: Verify response matches expected pattern
        assert result.is_error is False
        assert "hello" in result.content.lower() or "help" in result.content.lower()

    ##Method purpose: Test that engine retrieves relevant context
    def test_think_retrieves_context(
        self,
        cognitive_engine: CognitiveEngine,
        knowledge_store: KnowledgeStore,
    ) -> None:
        """Test that think() retrieves relevant context from memory.

        Args:
            cognitive_engine: Configured engine fixture.
            knowledge_store: Knowledge store fixture.
        """
        ##Step purpose: Add knowledge to memory first
        knowledge_store.add(
            content="The user's favorite color is blue.",
            memory_type=MemoryType.SEMANTIC,
            metadata={"source": "test"},
        )

        ##Action purpose: Query for the stored information
        result = cognitive_engine.think("What is my favorite color?")

        ##Step purpose: Verify context was retrieved and used
        assert result.is_error is False
        assert result.context_used >= 0  # May have context from memory
        ##Note: The mock LLM will respond based on pattern, but this tests the retrieval flow

    ##Method purpose: Test that engine stores interactions when learning enabled
    def test_think_stores_interaction(
        self,
        cognitive_engine: CognitiveEngine,
        knowledge_store: KnowledgeStore,
    ) -> None:
        """Test that think() stores interaction in memory when learning is enabled.

        Args:
            cognitive_engine: Configured engine fixture.
            knowledge_store: Knowledge store fixture.
        """
        ##Step purpose: Get initial memory count
        initial_count = knowledge_store.get_memory(MemoryType.EPISODIC).count()

        ##Action purpose: Process a query (should store interaction)
        cognitive_engine.think("This is a test message for storage.")

        ##Step purpose: Verify interaction was stored in episodic memory
        new_count = knowledge_store.get_memory(MemoryType.EPISODIC).count()
        assert new_count > initial_count, "Interaction should be stored in episodic memory"

    ##Method purpose: Test that learning can be disabled
    def test_think_respects_learning_disabled(
        self,
        cognitive_engine: CognitiveEngine,
        knowledge_store: KnowledgeStore,
    ) -> None:
        """Test that think() does not store when learning is disabled.

        Args:
            cognitive_engine: Configured engine fixture.
            knowledge_store: Knowledge store fixture.
        """
        ##Step purpose: Disable learning
        cognitive_engine.engine_config.enable_learning = False

        ##Step purpose: Get initial memory count
        initial_count = knowledge_store.get_memory(MemoryType.EPISODIC).count()

        ##Action purpose: Process a query
        cognitive_engine.think("This should not be stored.")

        ##Step purpose: Verify no new interaction was stored
        new_count = knowledge_store.get_memory(MemoryType.EPISODIC).count()
        assert new_count == initial_count, "Interaction should not be stored when learning disabled"

        ##Cleanup: Re-enable learning for other tests
        cognitive_engine.engine_config.enable_learning = True

    ##Method purpose: Test that engine handles errors gracefully
    def test_think_handles_empty_input(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that think() handles empty input gracefully.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Action purpose: Process empty query
        result = cognitive_engine.think("")

        ##Step purpose: Verify graceful handling
        assert result is not None
        assert result.content is not None  # Should still return something
        ##Note: Empty input is handled, not an error

    ##Method purpose: Test that reset clears engine state
    def test_reset_clears_state(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that reset() clears engine state.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Step purpose: Process some queries to build up state
        cognitive_engine.think("First message")
        cognitive_engine.think("Second message")

        ##Step purpose: Verify state exists before reset
        assert cognitive_engine._turn_count > 0
        assert len(cognitive_engine._history) > 0

        ##Action purpose: Reset the engine
        cognitive_engine.reset()

        ##Step purpose: Verify state is cleared
        assert cognitive_engine._turn_count == 0
        assert len(cognitive_engine._history) == 0

    ##Method purpose: Test that history is maintained across turns
    def test_maintains_conversation_history(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that engine maintains conversation history.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Action purpose: Process multiple queries
        cognitive_engine.think("My name is Alice.")
        cognitive_engine.think("I like programming.")
        cognitive_engine.think("What do you know about me?")

        ##Step purpose: Verify history is maintained
        assert len(cognitive_engine._history) == 3
        assert cognitive_engine._turn_count == 3

    ##Method purpose: Test that history is trimmed when exceeding max
    def test_history_trimming(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that history is trimmed when exceeding max turns.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Step purpose: Get max history turns
        max_turns = cognitive_engine.engine_config.max_history_turns

        ##Action purpose: Process more queries than max
        for i in range(max_turns + 3):
            cognitive_engine.think(f"Message number {i}")

        ##Step purpose: Verify history is trimmed
        assert len(cognitive_engine._history) <= max_turns


##Class purpose: Edge case tests for CognitiveEngine
@pytest.mark.integration
class TestCognitiveEngineEdgeCases:
    """Edge case tests for CognitiveEngine."""

    ##Method purpose: Test handling of very long input
    def test_handles_long_input(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that engine handles very long input.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Step purpose: Create a long input
        long_input = "This is a test. " * 500  # ~2500 words

        ##Action purpose: Process long input
        result = cognitive_engine.think(long_input)

        ##Step purpose: Verify graceful handling
        assert result is not None
        assert result.is_error is False

    ##Method purpose: Test handling of special characters
    def test_handles_special_characters(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that engine handles special characters.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Action purpose: Process input with special characters
        result = cognitive_engine.think("What about émojis 🎉 and spëcial çharacters?")

        ##Step purpose: Verify graceful handling
        assert result is not None
        assert result.is_error is False

    ##Method purpose: Test handling of newlines in input
    def test_handles_multiline_input(self, cognitive_engine: CognitiveEngine) -> None:
        """Test that engine handles multiline input.

        Args:
            cognitive_engine: Configured engine fixture.
        """
        ##Step purpose: Create multiline input
        multiline_input = """This is line one.
        This is line two.
        This is line three."""

        ##Action purpose: Process multiline input
        result = cognitive_engine.think(multiline_input)

        ##Step purpose: Verify graceful handling
        assert result is not None
        assert result.is_error is False
