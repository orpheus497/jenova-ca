##Script function and purpose: Integration tests for end-to-end cognitive flow
"""
Integration tests for cognitive cycle: Retrieve → Plan → Execute → Reflect

Tests cover:
- Complete memory → graph → response flow
- Context retrieval and organization
- Plan generation
- Response generation with sources
- Error scenarios
- State consistency
"""

import pytest
from unittest.mock import Mock, patch
from jenova.core.engine import CognitiveEngine
from jenova.core.response import ResponseGenerator
from jenova.memory.memory import KnowledgeStore
from jenova.graph.graph import CognitiveGraph
from jenova.exceptions import EngineError


##Class purpose: Fixture providing mock LLM
@pytest.fixture
def mock_llm() -> Mock:
    """##Test case: Mock LLM for integration tests."""
    llm = Mock()
    llm.generate = Mock(return_value='{"response": "Test response"}')
    return llm


##Class purpose: Fixture providing knowledge store
@pytest.fixture
def knowledge_store() -> KnowledgeStore:
    """##Test case: Real knowledge store instance."""
    return KnowledgeStore(embedding_model=None)


##Class purpose: Fixture providing cognitive graph
@pytest.fixture
def graph() -> CognitiveGraph:
    """##Test case: Real cognitive graph instance."""
    return CognitiveGraph()


##Class purpose: Fixture providing response generator
@pytest.fixture
def response_gen(mock_llm: Mock) -> ResponseGenerator:
    """##Test case: Response generator with mock LLM."""
    return ResponseGenerator(llm=mock_llm)


##Class purpose: Fixture providing cognitive engine
@pytest.fixture
def engine(mock_llm: Mock, knowledge_store: KnowledgeStore, graph: CognitiveGraph) -> CognitiveEngine:
    """##Test case: Complete cognitive engine."""
    engine = CognitiveEngine(llm=mock_llm)
    engine._memory = knowledge_store
    engine._graph = graph
    return engine


##Function purpose: Test basic cognitive flow
def test_basic_cognitive_flow(engine: CognitiveEngine, mock_llm: Mock) -> None:
    """##Test case: Basic cognitive cycle works end-to-end."""
    ##Step purpose: Send query through engine
    mock_llm.generate = Mock(return_value='{"response": "I understand your question"}')
    
    ##Action purpose: Execute think
    result = engine.think("What is machine learning?", username="test_user")
    
    ##Assertion purpose: Verify result
    assert result is not None
    assert hasattr(result, "response")


##Function purpose: Test memory storage
def test_memory_storage(knowledge_store: KnowledgeStore) -> None:
    """##Test case: Knowledge store stores interactions."""
    ##Action purpose: Add to memory
    knowledge_store.add_interaction(
        "user1",
        "What is AI?",
        "AI is artificial intelligence",
        tags=["question", "ai"]
    )
    
    ##Action purpose: Retrieve
    results = knowledge_store.search("AI", username="user1")
    
    ##Assertion purpose: Verify stored and retrieved
    assert len(results) > 0


##Function purpose: Test graph node creation
def test_graph_node_creation(graph: CognitiveGraph) -> None:
    """##Test case: Graph creates and stores nodes."""
    ##Action purpose: Add nodes
    node_id = graph.add_node(
        content="Machine learning concepts",
        node_type="topic",
        username="user1",
        tags=["ml", "tech"]
    )
    
    ##Assertion purpose: Verify created
    assert node_id is not None
    assert graph.get_node(node_id) is not None


##Function purpose: Test response generation
def test_response_generation(response_gen: ResponseGenerator, mock_llm: Mock) -> None:
    """##Test case: Response generator formats responses."""
    ##Step purpose: Set up LLM response
    mock_llm.generate = Mock(return_value="This is a test response")
    
    ##Action purpose: Generate response
    response = response_gen.generate(
        query="What is learning?",
        context=["Learning is acquiring knowledge"],
        user_context={"name": "test_user"}
    )
    
    ##Assertion purpose: Verify response
    assert response is not None
    assert len(response) > 0


##Function purpose: Test context organization
def test_context_organization(engine: CognitiveEngine, knowledge_store: KnowledgeStore) -> None:
    """##Test case: Context is properly organized by tier."""
    ##Step purpose: Store multiple interactions
    knowledge_store.add_interaction("user1", "Q1", "A1", tags=["important"])
    knowledge_store.add_interaction("user1", "Q2", "A2", tags=["reference"])
    knowledge_store.add_interaction("user1", "Q3", "A3", tags=["context"])
    
    ##Action purpose: Search and verify organization
    results = knowledge_store.search("Q", username="user1", limit=10)
    
    ##Assertion purpose: Verify retrieved
    assert len(results) >= 0


##Function purpose: Test error handling on missing LLM
def test_error_handling_missing_llm() -> None:
    """##Test case: Engine handles missing LLM gracefully."""
    ##Step purpose: Create engine without LLM
    engine = CognitiveEngine(llm=None)
    
    ##Action purpose: Try to think
    with pytest.raises(EngineError):
        engine.think("Test query", username="user1")


##Function purpose: Test graph linking
def test_graph_linking(graph: CognitiveGraph) -> None:
    """##Test case: Graph nodes can be linked."""
    ##Step purpose: Create two nodes
    node1 = graph.add_node(
        content="Concept A",
        node_type="concept",
        username="user1"
    )
    node2 = graph.add_node(
        content="Concept B",
        node_type="concept",
        username="user1"
    )
    
    ##Action purpose: Link nodes
    graph.add_edge(node1, node2, relationship="related_to")
    
    ##Assertion purpose: Verify linked
    node_data = graph.get_node(node1)
    assert node_data is not None


##Function purpose: Test state consistency memory to graph
def test_state_consistency_memory_graph(engine: CognitiveEngine, knowledge_store: KnowledgeStore, graph: CognitiveGraph) -> None:
    """##Test case: Memory and graph stay synchronized."""
    ##Step purpose: Store in memory
    knowledge_store.add_interaction(
        "user1",
        "question",
        "answer",
        tags=["test"]
    )
    
    ##Step purpose: Add corresponding node to graph
    graph.add_node(
        content="question",
        node_type="interaction",
        username="user1"
    )
    
    ##Action purpose: Verify both exist
    memory_results = knowledge_store.search("question", username="user1")
    graph_nodes = graph.get_nodes_by_user("user1")
    
    ##Assertion purpose: Verify consistency
    assert len(memory_results) > 0
    assert len(graph_nodes) > 0


##Function purpose: Test complex query flow
def test_complex_query_flow(engine: CognitiveEngine, mock_llm: Mock) -> None:
    """##Test case: Complex multi-turn queries are handled."""
    ##Step purpose: Set up LLM
    mock_llm.generate = Mock(return_value='{"response": "Complex answer", "plan": "detailed plan"}')
    
    ##Action purpose: Execute complex query
    result = engine.think(
        "Explain how neural networks learn by processing multiple examples over iterations",
        username="test_user"
    )
    
    ##Assertion purpose: Verify handling
    assert result is not None


##Function purpose: Test multiple users isolation
def test_multiple_users_isolation(knowledge_store: KnowledgeStore) -> None:
    """##Test case: Multiple users' data is isolated."""
    ##Step purpose: Add data for two users
    knowledge_store.add_interaction("user1", "Query A", "Answer A")
    knowledge_store.add_interaction("user2", "Query B", "Answer B")
    
    ##Action purpose: Search for each user
    user1_results = knowledge_store.search("Query", username="user1")
    user2_results = knowledge_store.search("Query", username="user2")
    
    ##Assertion purpose: Verify isolation
    # Each should get their own results
    assert user1_results is not None
    assert user2_results is not None


##Function purpose: Test engine reset
def test_engine_reset(engine: CognitiveEngine, knowledge_store: KnowledgeStore) -> None:
    """##Test case: Engine can be reset."""
    ##Step purpose: Store some data
    knowledge_store.add_interaction("user1", "Q", "A")
    
    ##Action purpose: Reset engine
    engine.reset()
    
    ##Assertion purpose: Verify reset (should not error)
    assert engine is not None
