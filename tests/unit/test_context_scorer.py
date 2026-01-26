##Script function and purpose: Unit tests for ContextScorer - multi-factor relevance scoring
"""
Tests for ContextScorer

Tests context scoring including semantic similarity, entity overlap,
keyword matching, and query type alignment.
"""

import pytest

from jenova.core.context_scorer import (
    ContextScorer,
    ContextScorerConfig,
    ScoredContext,
    ScoringBreakdown,
    ScoringWeights,
)
from jenova.core.query_analyzer import (
    AnalyzedQuery,
    QueryComplexity,
    QueryIntent,
    QueryType,
)


##Class purpose: Test ScoringWeights normalization
class TestScoringWeights:
    """Tests for ScoringWeights dataclass."""

    ##Method purpose: Test normalization sums to 1.0
    def test_normalize_sums_to_one(self) -> None:
        """Normalized weights sum to 1.0."""
        weights = ScoringWeights(
            semantic_similarity=0.8,
            entity_overlap=0.6,
            keyword_match=0.4,
            query_type_match=0.2,
        )

        normalized = weights.normalize()
        total = (
            normalized.semantic_similarity
            + normalized.entity_overlap
            + normalized.keyword_match
            + normalized.query_type_match
        )

        assert abs(total - 1.0) < 0.001

    ##Method purpose: Test default weights are valid
    def test_default_weights_sum_to_one(self) -> None:
        """Default weights sum to 1.0."""
        weights = ScoringWeights()
        total = (
            weights.semantic_similarity
            + weights.entity_overlap
            + weights.keyword_match
            + weights.query_type_match
        )

        assert abs(total - 1.0) < 0.001

    ##Method purpose: Test zero weights normalization
    def test_zero_weights_normalization(self) -> None:
        """Zero weights return defaults on normalization."""
        weights = ScoringWeights(
            semantic_similarity=0,
            entity_overlap=0,
            keyword_match=0,
            query_type_match=0,
        )

        normalized = weights.normalize()

        # Should return default weights
        assert normalized.semantic_similarity == 0.4


##Class purpose: Test ContextScorer scoring behavior
class TestContextScorerScoring:
    """Tests for ContextScorer scoring functionality."""

    ##Method purpose: Create scorer fixture
    @pytest.fixture
    def scorer(self) -> ContextScorer:
        """Create a ContextScorer instance."""
        return ContextScorer()

    ##Method purpose: Create sample analysis fixture
    @pytest.fixture
    def sample_analysis(self) -> AnalyzedQuery:
        """Create a sample AnalyzedQuery for testing."""
        return AnalyzedQuery(
            original_query="How do I learn Python programming?",
            intent=QueryIntent.QUESTION,
            intent_confidence=0.9,
            complexity=QueryComplexity.SIMPLE,
            complexity_confidence=0.8,
            query_type=QueryType.PROCEDURAL,
            type_confidence=0.8,
            entities=["Python"],
            keywords=["learn", "python", "programming"],
            topics=[],
            entity_links=[],
            reformulations=[],
            overall_confidence=0.8,
        )

    ##Method purpose: Test scoring returns ScoredContext
    def test_score_returns_scored_context(
        self,
        scorer: ContextScorer,
        sample_analysis: AnalyzedQuery,
    ) -> None:
        """Score method returns ScoredContext."""
        context_items = [
            "Python is a programming language.",
            "JavaScript is used for web development.",
        ]

        result = scorer.score(
            context_items,
            sample_analysis.original_query,
            sample_analysis,
        )

        assert isinstance(result, ScoredContext)
        assert len(result.items) == 2

    ##Method purpose: Test relevant items score higher
    def test_relevant_items_score_higher(
        self,
        scorer: ContextScorer,
        sample_analysis: AnalyzedQuery,
    ) -> None:
        """Items with matching content score higher."""
        context_items = [
            "Python programming tutorial for beginners.",
            "Cooking recipes from around the world.",
        ]

        result = scorer.score(
            context_items,
            sample_analysis.original_query,
            sample_analysis,
        )

        # Python item should score higher
        assert result.items[0].content == "Python programming tutorial for beginners."
        assert result.items[0].total_score > result.items[1].total_score

    ##Method purpose: Test scores are in valid range
    def test_scores_in_valid_range(
        self,
        scorer: ContextScorer,
        sample_analysis: AnalyzedQuery,
    ) -> None:
        """All scores are between 0.0 and 1.0."""
        context_items = ["Test content one.", "Test content two."]

        result = scorer.score(
            context_items,
            sample_analysis.original_query,
            sample_analysis,
        )

        for item in result.items:
            assert 0.0 <= item.total_score <= 1.0
            assert 0.0 <= item.semantic_score <= 1.0
            assert 0.0 <= item.entity_score <= 1.0
            assert 0.0 <= item.keyword_score <= 1.0
            assert 0.0 <= item.type_score <= 1.0

    ##Method purpose: Test empty context returns empty result
    def test_empty_context_returns_empty(
        self,
        scorer: ContextScorer,
        sample_analysis: AnalyzedQuery,
    ) -> None:
        """Empty context list returns empty ScoredContext."""
        result = scorer.score(
            [],
            sample_analysis.original_query,
            sample_analysis,
        )

        assert len(result.items) == 0


##Class purpose: Test individual scoring factors
class TestScoringFactors:
    """Tests for individual scoring factors."""

    ##Method purpose: Create scorer fixture
    @pytest.fixture
    def scorer(self) -> ContextScorer:
        """Create a ContextScorer instance."""
        return ContextScorer()

    ##Method purpose: Test entity overlap scoring
    def test_entity_overlap_scoring(self, scorer: ContextScorer) -> None:
        """Entity overlap increases entity score."""
        analysis = AnalyzedQuery(
            original_query="Tell me about Python",
            entities=["Python", "programming"],
        )

        context_with_entity = "Python is a great language."
        context_without = "The weather is nice today."

        result = scorer.score(
            [context_with_entity, context_without],
            analysis.original_query,
            analysis,
        )

        # Item with entity match should have higher entity score
        python_item = next(i for i in result.items if "Python" in i.content)
        weather_item = next(i for i in result.items if "weather" in i.content)

        assert python_item.entity_score > weather_item.entity_score

    ##Method purpose: Test keyword match scoring
    def test_keyword_match_scoring(self, scorer: ContextScorer) -> None:
        """Keyword matches increase keyword score."""
        analysis = AnalyzedQuery(
            original_query="machine learning algorithms",
            keywords=["machine", "learning", "algorithms"],
        )

        context_with_keywords = "Machine learning algorithms are powerful."
        context_without = "The sunset was beautiful."

        result = scorer.score(
            [context_with_keywords, context_without],
            analysis.original_query,
            analysis,
        )

        ml_item = next(i for i in result.items if "learning" in i.content)
        sunset_item = next(i for i in result.items if "sunset" in i.content)

        assert ml_item.keyword_score > sunset_item.keyword_score

    ##Method purpose: Test query type match for procedural
    def test_procedural_type_match(self, scorer: ContextScorer) -> None:
        """Procedural content scores higher for procedural queries."""
        analysis = AnalyzedQuery(
            original_query="How to install Python",
            query_type=QueryType.PROCEDURAL,
        )

        procedural_content = "Step 1: Download Python. Step 2: Install."
        factual_content = "Python was created in 1991."

        result = scorer.score(
            [procedural_content, factual_content],
            analysis.original_query,
            analysis,
        )

        step_item = next(i for i in result.items if "Step" in i.content)
        created_item = next(i for i in result.items if "created" in i.content)

        assert step_item.type_score >= created_item.type_score


##Class purpose: Test ScoredContext methods
class TestScoredContextMethods:
    """Tests for ScoredContext helper methods."""

    ##Method purpose: Create scored context fixture
    @pytest.fixture
    def scored_context(self) -> ScoredContext:
        """Create a ScoredContext with test data."""
        items = [
            ScoringBreakdown(content="High score item", total_score=0.9),
            ScoringBreakdown(content="Medium score item", total_score=0.6),
            ScoringBreakdown(content="Low score item", total_score=0.3),
        ]
        return ScoredContext(items=items, query="test query")

    ##Method purpose: Test top method
    def test_top_returns_n_items(self, scored_context: ScoredContext) -> None:
        """Top method returns correct number of items."""
        top_2 = scored_context.top(2)

        assert len(top_2) == 2
        assert top_2[0].total_score == 0.9
        assert top_2[1].total_score == 0.6

    ##Method purpose: Test above_threshold method
    def test_above_threshold_filters_correctly(
        self,
        scored_context: ScoredContext,
    ) -> None:
        """Above threshold filters items correctly."""
        above_05 = scored_context.above_threshold(0.5)

        assert len(above_05) == 2
        for item in above_05:
            assert item.total_score >= 0.5

    ##Method purpose: Test as_strings method
    def test_as_strings_returns_content(
        self,
        scored_context: ScoredContext,
    ) -> None:
        """As strings returns content strings."""
        strings = scored_context.as_strings()

        assert len(strings) == 3
        assert "High score item" in strings

    ##Method purpose: Test as_strings with limit
    def test_as_strings_respects_limit(
        self,
        scored_context: ScoredContext,
    ) -> None:
        """As strings respects optional limit."""
        strings = scored_context.as_strings(n=2)

        assert len(strings) == 2


##Class purpose: Test scorer configuration
class TestScorerConfiguration:
    """Tests for scorer configuration."""

    ##Method purpose: Test disabled scorer returns default scores
    def test_disabled_scorer_returns_default_scores(self) -> None:
        """Disabled scorer returns 1.0 for all items."""
        config = ContextScorerConfig(enabled=False)
        scorer = ContextScorer(config=config)

        analysis = AnalyzedQuery(original_query="test")
        result = scorer.score(["item1", "item2"], "test", analysis)

        for item in result.items:
            assert item.total_score == 1.0

    ##Method purpose: Test custom weights are used
    def test_custom_weights_used(self) -> None:
        """Custom weights affect scoring."""
        # Heavy weight on keyword match
        config = ContextScorerConfig(
            weights=ScoringWeights(
                semantic_similarity=0.0,
                entity_overlap=0.0,
                keyword_match=1.0,
                query_type_match=0.0,
            ),
        )
        scorer = ContextScorer(config=config)

        analysis = AnalyzedQuery(
            original_query="python",
            keywords=["python"],
        )

        result = scorer.score(
            ["python programming", "java development"],
            "python",
            analysis,
        )

        # With only keyword weight, python item should dominate
        python_item = next(i for i in result.items if "python" in i.content)
        java_item = next(i for i in result.items if "java" in i.content)

        assert python_item.total_score > java_item.total_score


##Class purpose: Test simple scoring interface
class TestSimpleScoring:
    """Tests for simple scoring interface."""

    ##Method purpose: Create scorer fixture
    @pytest.fixture
    def scorer(self) -> ContextScorer:
        """Create a ContextScorer instance."""
        return ContextScorer()

    ##Method purpose: Test score_simple returns tuples
    def test_score_simple_returns_tuples(self, scorer: ContextScorer) -> None:
        """Score simple returns list of tuples."""
        result = scorer.score_simple(["item1", "item2"], "query")

        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    ##Method purpose: Test score_simple sorts by score
    def test_score_simple_sorts_by_score(self, scorer: ContextScorer) -> None:
        """Score simple returns items sorted by score."""
        result = scorer.score_simple(
            ["Python programming", "Cooking recipes"],
            "Python",
        )

        # Should be sorted descending
        if len(result) >= 2:
            assert result[0][1] >= result[1][1]
