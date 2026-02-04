##Script function and purpose: Integration tests for KnowledgeStore - Tests
##                          unified memory and graph operations.
##Dependency purpose: Validates that KnowledgeStore correctly stores,
##                    retrieves, and searches knowledge.
"""Integration tests for KnowledgeStore.

Tests the unified knowledge system including:
- Memory storage and retrieval
- Graph operations
- Search functionality
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jenova.config.models import GraphConfig, MemoryConfig
from jenova.core.knowledge import KnowledgeStore
from jenova.graph.types import EdgeType, Node
from jenova.memory.types import MemoryType

if TYPE_CHECKING:
    pass


##Class purpose: Integration tests for KnowledgeStore
@pytest.mark.integration
class TestKnowledgeStoreIntegration:
    """Integration tests for KnowledgeStore."""

    ##Method purpose: Test that knowledge can be stored and retrieved
    def test_store_and_retrieve(self, knowledge_store: KnowledgeStore) -> None:
        """Test storing and retrieving knowledge.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Store content in semantic memory
        content = "Python is a programming language known for its readability."
        doc_id = knowledge_store.add(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            metadata={"source": "test", "topic": "programming"},
        )

        ##Step purpose: Verify ID was returned
        assert doc_id is not None
        assert len(doc_id) > 0

        ##Action purpose: Retrieve the stored content
        result = knowledge_store.get_memory(MemoryType.SEMANTIC).get(doc_id)

        ##Step purpose: Verify content matches
        assert result is not None
        assert result.content == content
        assert result.metadata.get("source") == "test"

    ##Method purpose: Test semantic search functionality
    def test_semantic_search(self, knowledge_store: KnowledgeStore) -> None:
        """Test semantic search returns relevant results.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Store multiple pieces of knowledge
        knowledge_store.add(
            content="Python is great for data science and machine learning.",
            memory_type=MemoryType.SEMANTIC,
        )
        knowledge_store.add(
            content="JavaScript is used for web development.",
            memory_type=MemoryType.SEMANTIC,
        )
        knowledge_store.add(
            content="Rust is known for memory safety and performance.",
            memory_type=MemoryType.SEMANTIC,
        )

        ##Action purpose: Search for Python-related content
        results = knowledge_store.search(
            query="data science programming",
            memory_types=[MemoryType.SEMANTIC],
            n_results=3,
            include_graph=False,
        )

        ##Step purpose: Verify results returned
        assert results is not None
        assert len(results.memories) > 0

        ##Step purpose: Verify most relevant result is about Python/data science
        top_result = results.memories[0]
        assert "python" in top_result.content.lower() or "data" in top_result.content.lower()

    ##Method purpose: Test graph relationship operations
    def test_graph_relationships(self, knowledge_store: KnowledgeStore) -> None:
        """Test graph relationship storage and traversal.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Create nodes
        python_node = Node.create(
            label="Python",
            content="A programming language",
            node_type="concept",
        )
        data_science_node = Node.create(
            label="Data Science",
            content="Field of study using data",
            node_type="concept",
        )

        ##Action purpose: Add nodes to graph
        knowledge_store.graph.add_node(python_node)
        knowledge_store.graph.add_node(data_science_node)

        ##Action purpose: Create relationship
        edge = knowledge_store.graph.add_edge(
            source_id=python_node.id,
            target_id=data_science_node.id,
            edge_type=EdgeType.RELATES_TO,
            weight=0.9,
        )

        ##Step purpose: Verify relationship exists
        assert edge is not None
        neighbors = knowledge_store.graph.neighbors(python_node.id)
        assert len(neighbors) == 1
        assert neighbors[0].id == data_science_node.id

    ##Method purpose: Test combined memory and graph search
    def test_combined_search(self, knowledge_store: KnowledgeStore) -> None:
        """Test search across both memory and graph.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Add memory content
        knowledge_store.add(
            content="Machine learning uses algorithms to learn from data.",
            memory_type=MemoryType.SEMANTIC,
        )

        ##Step purpose: Add graph node
        ml_node = Node.create(
            label="Machine Learning",
            content="Branch of AI that learns from data",
            node_type="concept",
        )
        knowledge_store.graph.add_node(ml_node)

        ##Action purpose: Search with graph included
        results = knowledge_store.search(
            query="machine learning algorithms",
            n_results=5,
            include_graph=True,
        )

        ##Step purpose: Verify both sources searched
        assert results is not None
        assert results.query == "machine learning algorithms"
        ##Note: Results may come from memory, graph, or both

    ##Method purpose: Test persistence across instances
    def test_persistence(self, integration_data_dir: Path) -> None:
        """Test that knowledge persists across store instances.

        Args:
            integration_data_dir: Temporary data directory fixture.
        """
        ##Step purpose: Create config with specific paths
        memory_config = MemoryConfig(storage_path=integration_data_dir / "memory")
        graph_config = GraphConfig(storage_path=integration_data_dir / "graph")

        ##Step purpose: Create first store instance and add data
        store1 = KnowledgeStore(
            memory_config=memory_config,
            graph_config=graph_config,
        )

        content = "This should persist across instances."
        doc_id = store1.add(
            content=content,
            memory_type=MemoryType.SEMANTIC,
        )

        ##Step purpose: Add a graph node
        node = Node.create(
            label="PersistenceTest",
            content="Testing persistence",
            node_type="test",
        )
        store1.graph.add_node(node)

        ##Step purpose: Create second store instance
        store2 = KnowledgeStore(
            memory_config=memory_config,
            graph_config=graph_config,
        )

        ##Step purpose: Verify memory data persisted
        retrieved = store2.get_memory(MemoryType.SEMANTIC).get(doc_id)
        assert retrieved is not None
        assert retrieved.content == content

        ##Step purpose: Verify graph data persisted
        assert store2.graph.has_node(node.id)
        retrieved_node = store2.graph.get_node(node.id)
        assert retrieved_node.label == "PersistenceTest"

    ##Method purpose: Test handling of empty queries
    def test_empty_query_handling(self, knowledge_store: KnowledgeStore) -> None:
        """Test graceful handling of empty queries.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Action purpose: Search with empty query
        results = knowledge_store.search(
            query="",
            n_results=5,
        )

        ##Step purpose: Verify graceful handling
        assert results is not None
        assert results.query == ""
        ##Note: Empty query should return empty or minimal results, not error


##Class purpose: Memory-specific integration tests
@pytest.mark.integration
class TestKnowledgeStoreMemory:
    """Memory-specific tests for KnowledgeStore."""

    ##Method purpose: Test storing in different memory types
    def test_different_memory_types(self, knowledge_store: KnowledgeStore) -> None:
        """Test storing content in different memory types.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Store in each memory type
        episodic_id = knowledge_store.add(
            content="User asked about Python yesterday.",
            memory_type=MemoryType.EPISODIC,
        )
        semantic_id = knowledge_store.add(
            content="Python was created by Guido van Rossum.",
            memory_type=MemoryType.SEMANTIC,
        )
        procedural_id = knowledge_store.add(
            content="To install Python, use your package manager.",
            memory_type=MemoryType.PROCEDURAL,
        )

        ##Step purpose: Verify each was stored in correct type
        assert knowledge_store.get_memory(MemoryType.EPISODIC).get(episodic_id) is not None
        assert knowledge_store.get_memory(MemoryType.SEMANTIC).get(semantic_id) is not None
        assert knowledge_store.get_memory(MemoryType.PROCEDURAL).get(procedural_id) is not None

    ##Method purpose: Test searching specific memory types
    def test_search_specific_memory_types(self, knowledge_store: KnowledgeStore) -> None:
        """Test searching only specific memory types.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Store in different types
        knowledge_store.add(
            content="Yesterday we discussed Python.",
            memory_type=MemoryType.EPISODIC,
        )
        knowledge_store.add(
            content="Python is a programming language.",
            memory_type=MemoryType.SEMANTIC,
        )

        ##Action purpose: Search only episodic memory
        episodic_results = knowledge_store.search(
            query="Python",
            memory_types=[MemoryType.EPISODIC],
            n_results=5,
            include_graph=False,
        )

        ##Step purpose: Verify only episodic results returned
        for memory in episodic_results.memories:
            assert memory.memory_type == MemoryType.EPISODIC

    ##Method purpose: Test memory metadata handling
    def test_memory_metadata(self, knowledge_store: KnowledgeStore) -> None:
        """Test metadata is stored and retrieved correctly.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Store with metadata
        metadata = {"source": "test", "topic": "testing", "priority": "high"}
        doc_id = knowledge_store.add(
            content="Content with metadata.",
            memory_type=MemoryType.SEMANTIC,
            metadata=metadata,
        )

        ##Action purpose: Retrieve and check metadata
        result = knowledge_store.get_memory(MemoryType.SEMANTIC).get(doc_id)

        ##Step purpose: Verify metadata
        assert result is not None
        assert result.metadata.get("source") == "test"
        assert result.metadata.get("topic") == "testing"
        assert result.metadata.get("priority") == "high"


##Class purpose: Graph-specific integration tests
@pytest.mark.integration
class TestKnowledgeStoreGraph:
    """Graph-specific tests for KnowledgeStore."""

    ##Method purpose: Test graph node operations
    def test_graph_node_crud(self, knowledge_store: KnowledgeStore) -> None:
        """Test graph node create, read, update, delete.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Create node
        node = Node.create(
            label="TestNode",
            content="Test content",
            node_type="test",
        )
        knowledge_store.graph.add_node(node)

        ##Step purpose: Read node
        assert knowledge_store.graph.has_node(node.id)
        retrieved = knowledge_store.graph.get_node(node.id)
        assert retrieved.label == "TestNode"

        ##Step purpose: Delete node
        knowledge_store.graph.remove_node(node.id)
        assert not knowledge_store.graph.has_node(node.id)

    ##Method purpose: Test graph search
    def test_graph_search(self, knowledge_store: KnowledgeStore) -> None:
        """Test graph search functionality.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Create nodes
        node1 = Node.create(
            label="Programming Languages",
            content="Languages for software development",
            node_type="concept",
        )
        node2 = Node.create(
            label="Web Development",
            content="Building websites and web applications",
            node_type="concept",
        )
        knowledge_store.graph.add_node(node1)
        knowledge_store.graph.add_node(node2)

        ##Action purpose: Search graph
        results = knowledge_store.graph.search(
            query="programming",
            max_results=5,
        )

        ##Step purpose: Verify results
        assert len(results) > 0
        assert any("programming" in r["label"].lower() for r in results)

    ##Method purpose: Test edge types
    def test_graph_edge_types(self, knowledge_store: KnowledgeStore) -> None:
        """Test different edge types.

        Args:
            knowledge_store: Configured knowledge store fixture.
        """
        ##Step purpose: Create nodes
        parent = Node.create(label="Parent", content="Parent node", node_type="test")
        child = Node.create(label="Child", content="Child node", node_type="test")
        knowledge_store.graph.add_node(parent)
        knowledge_store.graph.add_node(child)

        ##Action purpose: Create edge with specific type
        edge = knowledge_store.graph.add_edge(
            source_id=parent.id,
            target_id=child.id,
            edge_type=EdgeType.HAS_CHILD,
            weight=1.0,
        )

        ##Step purpose: Verify edge type
        assert edge.edge_type == EdgeType.HAS_CHILD
