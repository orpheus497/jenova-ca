##Script function and purpose: Unit tests for proactive suggestion engine
"""
Test suite for ProactiveEngine - Autonomous suggestion generation based on cognitive state.

Tests cover:
- Suggestion generation (5 categories)
- Cooldown management
- Engagement tracking
- Priority filtering
- Error handling and exception safety
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from jenova.exceptions import GraphError, ProactiveError
from jenova.graph.proactive import (
    EngagementTracker,
    GraphProtocol,
    LLMProtocol,
    ProactiveConfig,
    ProactiveEngine,
    Suggestion,
    SuggestionCategory,
)


##Class purpose: Fixture providing mock graph
@pytest.fixture
def mock_graph() -> Mock:
    """##Test case: Mock graph implementation."""
    graph = Mock(spec=GraphProtocol)
    graph.get_nodes_by_user = Mock(return_value=[])
    graph.search = Mock(return_value=[])
    return graph


##Class purpose: Fixture providing mock LLM
@pytest.fixture
def mock_llm() -> Mock:
    """##Test case: Mock LLM implementation."""
    llm = Mock(spec=LLMProtocol)
    llm.generate = Mock(return_value="Test suggestion")
    return llm


##Class purpose: Fixture providing default config
@pytest.fixture
def default_config() -> ProactiveConfig:
    """##Test case: Default proactive configuration."""
    return ProactiveConfig(
        cooldown_minutes=15,
        max_suggestions_per_session=10,
        priority_threshold=0.3,
    )


##Class purpose: Fixture providing proactive engine
@pytest.fixture
def engine(default_config: ProactiveConfig, mock_graph: Mock, mock_llm: Mock) -> ProactiveEngine:
    """##Test case: Proactive engine with dependencies."""
    return ProactiveEngine(default_config, mock_graph, mock_llm)


##Function purpose: Test engagement tracker initialization
def test_engagement_tracker_init() -> None:
    """##Test case: EngagementTracker initializes correctly."""
    ##Step purpose: Create tracker
    tracker = EngagementTracker()

    ##Assertion purpose: Verify initial state
    assert tracker.suggestions_shown == 0
    assert tracker.suggestions_accepted == 0
    assert len(tracker.category_shown) == 0
    assert len(tracker.category_accepted) == 0


##Function purpose: Test engagement tracker recording shown
def test_engagement_tracker_record_shown() -> None:
    """##Test case: EngagementTracker records suggestions shown."""
    ##Step purpose: Create tracker and record
    tracker = EngagementTracker()
    tracker.record_shown(SuggestionCategory.EXPLORE)

    ##Assertion purpose: Verify counts
    assert tracker.suggestions_shown == 1
    assert tracker.category_shown[SuggestionCategory.EXPLORE] == 1


##Function purpose: Test engagement tracker recording accepted
def test_engagement_tracker_record_accepted() -> None:
    """##Test case: EngagementTracker records suggestions accepted."""
    ##Step purpose: Create tracker and record
    tracker = EngagementTracker()
    tracker.record_shown(SuggestionCategory.VERIFY)
    tracker.record_accepted(SuggestionCategory.VERIFY)

    ##Assertion purpose: Verify counts
    assert tracker.suggestions_accepted == 1
    assert tracker.category_accepted[SuggestionCategory.VERIFY] == 1


##Function purpose: Test acceptance rate calculation
def test_engagement_tracker_acceptance_rate() -> None:
    """##Test case: EngagementTracker calculates acceptance rate."""
    ##Step purpose: Create tracker with data
    tracker = EngagementTracker()
    tracker.record_shown(SuggestionCategory.DEVELOP)
    tracker.record_shown(SuggestionCategory.DEVELOP)
    tracker.record_accepted(SuggestionCategory.DEVELOP)

    ##Assertion purpose: Verify rate (1 / 2 = 0.5)
    rate = tracker.get_acceptance_rate(SuggestionCategory.DEVELOP)
    assert rate == 0.5


##Function purpose: Test acceptance rate with no shows
def test_engagement_tracker_acceptance_rate_no_shows() -> None:
    """##Test case: Acceptance rate defaults to 0.5 when never shown."""
    ##Step purpose: Create tracker
    tracker = EngagementTracker()

    ##Assertion purpose: Verify default rate
    rate = tracker.get_acceptance_rate(SuggestionCategory.CONNECT)
    assert rate == 0.5


##Function purpose: Test suggestion dataclass immutability
def test_suggestion_frozen() -> None:
    """##Test case: Suggestion is frozen (immutable)."""
    ##Step purpose: Create suggestion
    suggestion = Suggestion(
        category=SuggestionCategory.EXPLORE,
        content="Test content",
    )

    ##Assertion purpose: Verify frozen
    with pytest.raises(AttributeError):
        suggestion.priority = 0.8


##Function purpose: Test engine initialization
def test_engine_initialization(
    default_config: ProactiveConfig, mock_graph: Mock, mock_llm: Mock
) -> None:
    """##Test case: ProactiveEngine initializes correctly."""
    ##Step purpose: Create engine
    engine = ProactiveEngine(default_config, mock_graph, mock_llm)

    ##Assertion purpose: Verify state
    assert engine._config == default_config
    assert engine._graph is mock_graph
    assert engine._llm is mock_llm
    assert engine._session_suggestion_count == 0


##Function purpose: Test set graph
def test_set_graph(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: Can set graph after initialization."""
    ##Step purpose: Create new graph and set
    new_graph = Mock(spec=GraphProtocol)
    engine.set_graph(new_graph)

    ##Assertion purpose: Verify graph set
    assert engine._graph is new_graph


##Function purpose: Test set LLM
def test_set_llm(engine: ProactiveEngine, mock_llm: Mock) -> None:
    """##Test case: Can set LLM after initialization."""
    ##Step purpose: Create new LLM and set
    new_llm = Mock(spec=LLMProtocol)
    engine.set_llm(new_llm)

    ##Assertion purpose: Verify LLM set
    assert engine._llm is new_llm


##Function purpose: Test cooldown tracking
def test_is_on_cooldown_no_previous() -> None:
    """##Test case: First suggestion has no cooldown."""
    ##Step purpose: Create engine
    config = ProactiveConfig(cooldown_minutes=15)
    engine = ProactiveEngine(config)

    ##Assertion purpose: Verify no cooldown on first call
    assert not engine._is_on_cooldown(SuggestionCategory.EXPLORE)


##Function purpose: Test cooldown enforcement
def test_is_on_cooldown_within_cooldown() -> None:
    """##Test case: Suggestion within cooldown is blocked."""
    ##Step purpose: Create engine and simulate suggestion
    config = ProactiveConfig(cooldown_minutes=15)
    engine = ProactiveEngine(config)

    ##Action purpose: Simulate suggestion just made
    engine._last_suggestion_time[SuggestionCategory.EXPLORE] = datetime.now()

    ##Assertion purpose: Verify on cooldown
    assert engine._is_on_cooldown(SuggestionCategory.EXPLORE)


##Function purpose: Test cooldown expiration
def test_is_on_cooldown_expired() -> None:
    """##Test case: Suggestion after cooldown is allowed."""
    ##Step purpose: Create engine and set old suggestion time
    config = ProactiveConfig(cooldown_minutes=15)
    engine = ProactiveEngine(config)

    ##Action purpose: Set suggestion time to past (way outside cooldown)
    old_time = datetime.now() - timedelta(hours=1)
    engine._last_suggestion_time[SuggestionCategory.EXPLORE] = old_time

    ##Assertion purpose: Verify not on cooldown
    assert not engine._is_on_cooldown(SuggestionCategory.EXPLORE)


##Function purpose: Test session limit enforcement
def test_get_suggestion_session_limit(engine: ProactiveEngine) -> None:
    """##Test case: get_suggestion respects session limit."""
    ##Step purpose: Set session count to limit
    engine._session_suggestion_count = engine._config.max_suggestions_per_session

    ##Action purpose: Try to get suggestion
    result = engine.get_suggestion("user1")

    ##Assertion purpose: Verify None returned
    assert result is None


##Function purpose: Test get suggestion with all categories on cooldown
def test_get_suggestion_all_cooldown(engine: ProactiveEngine) -> None:
    """##Test case: get_suggestion returns None if all on cooldown."""
    ##Step purpose: Put all categories on cooldown
    now = datetime.now()
    for category in SuggestionCategory:
        engine._last_suggestion_time[category] = now

    ##Action purpose: Try to get suggestion
    result = engine.get_suggestion("user1")

    ##Assertion purpose: Verify None returned
    assert result is None


##Function purpose: Test explore suggestion generation
def test_generate_explore_suggestion_no_nodes(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: Explore suggestion generated for new user."""
    ##Step purpose: Set up mock to return no nodes
    mock_graph.get_nodes_by_user = Mock(return_value=[])

    ##Action purpose: Generate suggestion
    result = engine._generate_explore_suggestion("user1")

    ##Assertion purpose: Verify suggestion
    assert result is not None
    assert result.category == SuggestionCategory.EXPLORE
    assert "explore" in result.content.lower()


##Function purpose: Test explore suggestion with nodes
def test_generate_explore_suggestion_with_nodes(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: Explore suggestion identifies underexplored topics."""
    ##Step purpose: Set up mock with topic data
    nodes = [
        {"type": "question", "id": "1"},
        {"type": "question", "id": "2"},
        {"type": "question", "id": "3"},
        {"type": "insight", "id": "4"},  # Only 1 insight
    ]
    mock_graph.get_nodes_by_user = Mock(return_value=nodes)

    ##Action purpose: Generate suggestion
    result = engine._generate_explore_suggestion("user1")

    ##Assertion purpose: Verify suggestion targets less explored type
    assert result is not None
    assert result.category == SuggestionCategory.EXPLORE


##Function purpose: Test verify suggestion generation
def test_generate_verify_suggestion(engine: ProactiveEngine) -> None:
    """##Test case: Verify suggestion generated."""
    ##Action purpose: Generate suggestion
    result = engine._generate_verify_suggestion("user1")

    ##Assertion purpose: Verify suggestion
    assert result is not None
    assert result.category == SuggestionCategory.VERIFY


##Function purpose: Test develop suggestion generation
def test_generate_develop_suggestion(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: Develop suggestion generated from insights."""
    ##Step purpose: Set up mock with insight nodes
    nodes = [
        {"type": "insight", "id": "1", "content": "Important insight about learning"},
        {"type": "insight", "id": "2", "content": "Another insight"},
    ]
    mock_graph.get_nodes_by_user = Mock(return_value=nodes)

    ##Action purpose: Generate suggestion
    result = engine._generate_develop_suggestion("user1")

    ##Assertion purpose: Verify suggestion
    assert result is not None
    assert result.category == SuggestionCategory.DEVELOP


##Function purpose: Test develop suggestion with no insights
def test_generate_develop_suggestion_no_insights(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: Develop suggestion returns None if no insights."""
    ##Step purpose: Set up mock with no insight nodes
    nodes = [
        {"type": "question", "id": "1"},
        {"type": "reference", "id": "2"},
    ]
    mock_graph.get_nodes_by_user = Mock(return_value=nodes)

    ##Action purpose: Generate suggestion
    result = engine._generate_develop_suggestion("user1")

    ##Assertion purpose: Verify None
    assert result is None


##Function purpose: Test connect suggestion generation
def test_generate_connect_suggestion(engine: ProactiveEngine) -> None:
    """##Test case: Connect suggestion generated."""
    ##Action purpose: Generate suggestion
    result = engine._generate_connect_suggestion("user1")

    ##Assertion purpose: Verify suggestion
    assert result is not None
    assert result.category == SuggestionCategory.CONNECT


##Function purpose: Test reflect suggestion generation
def test_generate_reflect_suggestion(engine: ProactiveEngine) -> None:
    """##Test case: Reflect suggestion generated."""
    ##Action purpose: Generate suggestion
    result = engine._generate_reflect_suggestion("user1")

    ##Assertion purpose: Verify suggestion
    assert result is not None
    assert result.category == SuggestionCategory.REFLECT


##Function purpose: Test generate suggestion routing
def test_generate_suggestion_routes_to_correct_generator(
    engine: ProactiveEngine, mock_graph: Mock
) -> None:
    """##Test case: _generate_suggestion routes to correct category generator."""
    ##Step purpose: Mock graph
    mock_graph.get_nodes_by_user = Mock(return_value=[])

    ##Action purpose: Generate explore suggestion
    result = engine._generate_suggestion(SuggestionCategory.EXPLORE, "user1")

    ##Assertion purpose: Verify correct type returned
    assert result.category == SuggestionCategory.EXPLORE


##Function purpose: Test generate suggestion with unknown category
def test_generate_suggestion_unknown_category(engine: ProactiveEngine) -> None:
    """##Test case: Unknown category raises ProactiveError."""
    ##Step purpose: Create invalid category (use mock)
    invalid_category = Mock()
    invalid_category.__class__ = SuggestionCategory

    ##Action purpose: Try to generate suggestion
    with pytest.raises(ProactiveError):
        engine._generate_suggestion(invalid_category, "user1")


##Function purpose: Test generate suggestion handles graph errors
def test_generate_suggestion_graph_error(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: GraphError is caught and re-raised as ProactiveError."""
    ##Step purpose: Set up mock to raise error
    mock_graph.get_nodes_by_user = Mock(side_effect=GraphError("Graph error"))

    ##Action purpose: Try to generate suggestion
    with pytest.raises(ProactiveError):
        engine._generate_suggestion(SuggestionCategory.EXPLORE, "user1")


##Function purpose: Test get suggestion full flow
def test_get_suggestion_full_flow(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: get_suggestion executes complete flow."""
    ##Step purpose: Set up mock
    mock_graph.get_nodes_by_user = Mock(return_value=[])

    ##Action purpose: Get suggestion
    result = engine.get_suggestion("user1")

    ##Assertion purpose: Verify suggestion returned
    assert result is not None
    assert isinstance(result, Suggestion)
    assert engine._session_suggestion_count == 1


##Function purpose: Test suggestion below threshold filtered
def test_get_suggestion_below_threshold(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: Suggestions below priority threshold are filtered."""
    ##Step purpose: Set threshold high and generate low-priority suggestion
    engine._config.priority_threshold = 0.9
    mock_graph.get_nodes_by_user = Mock(return_value=[])

    ##Action purpose: Get suggestion (reflect has 0.3 priority, below 0.9)
    # We need to force REFLECT category
    engine._last_category = None
    engine._last_suggestion_time = {}
    for _ in range(4):  # Try a few times to maybe get REFLECT
        result = engine.get_suggestion("user1")
        if result and result.category == SuggestionCategory.REFLECT and result.priority < 0.9:
            assert result is None or result.priority >= 0.9
            break


##Function purpose: Test record acceptance
def test_record_acceptance(engine: ProactiveEngine) -> None:
    """##Test case: record_acceptance updates engagement."""
    ##Step purpose: Record acceptance
    engine.record_acceptance(SuggestionCategory.EXPLORE)

    ##Assertion purpose: Verify tracked
    assert engine._engagement.suggestions_accepted == 1


##Function purpose: Test reset session
def test_reset_session(engine: ProactiveEngine, mock_graph: Mock) -> None:
    """##Test case: reset_session clears session state."""
    ##Step purpose: Generate suggestion and reset
    mock_graph.get_nodes_by_user = Mock(return_value=[])
    engine.get_suggestion("user1")

    ##Action purpose: Reset
    engine.reset_session()

    ##Assertion purpose: Verify reset
    assert engine._session_suggestion_count == 0
    assert engine._last_category is None


##Function purpose: Test get status
def test_get_status(engine: ProactiveEngine) -> None:
    """##Test case: get_status returns complete status dict."""
    ##Step purpose: Get status
    status = engine.get_status()

    ##Assertion purpose: Verify structure
    assert "session_count" in status
    assert "max_per_session" in status
    assert "last_category" in status
    assert "cooldowns" in status
    assert "engagement" in status


##Function purpose: Test rotation avoiding last category
def test_get_next_category_rotation(engine: ProactiveEngine) -> None:
    """##Test case: Category rotation avoids repeating last category."""
    ##Step purpose: Set last category and get next
    engine._last_category = SuggestionCategory.EXPLORE
    engine._config.rotation_enabled = True

    ##Action purpose: Get next category multiple times
    results = []
    for _ in range(3):
        cat = engine._get_next_category()
        if cat:
            results.append(cat)

    ##Assertion purpose: Verify mostly different categories
    # At least some should be different from EXPLORE
    different = [c for c in results if c != SuggestionCategory.EXPLORE]
    assert len(different) > 0
