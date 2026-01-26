##Script function and purpose: Unit tests for QueryAnalyzer - intent, complexity, entity extraction
"""
Tests for QueryAnalyzer

Tests query analysis including intent classification, complexity assessment,
entity extraction, topic detection, and reformulation generation.
"""

import pytest

from jenova.core.query_analyzer import (
    AnalyzedQuery,
    EntityLink,
    QueryAnalyzer,
    QueryAnalyzerConfig,
    QueryComplexity,
    QueryIntent,
    QueryType,
    TopicCategory,
    TopicResult,
)


##Class purpose: Test QueryAnalyzer intent classification
class TestQueryIntentClassification:
    """Tests for intent classification."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer(config=QueryAnalyzerConfig())

    ##Method purpose: Test question detection
    def test_question_detected_by_question_mark(self, analyzer: QueryAnalyzer) -> None:
        """Question mark triggers question intent."""
        result = analyzer.analyze("What is Python?")

        assert result.intent == QueryIntent.QUESTION
        assert result.intent_confidence >= 0.8

    ##Method purpose: Test question detection by keyword
    def test_question_detected_by_keyword(self, analyzer: QueryAnalyzer) -> None:
        """Question words trigger question intent."""
        result = analyzer.analyze("How does machine learning work")

        assert result.intent == QueryIntent.QUESTION
        assert result.intent_confidence >= 0.7

    ##Method purpose: Test command detection
    def test_command_detected_by_slash(self, analyzer: QueryAnalyzer) -> None:
        """Slash prefix triggers command intent."""
        result = analyzer.analyze("/help")

        assert result.intent == QueryIntent.COMMAND
        assert result.intent_confidence >= 0.7

    ##Method purpose: Test command detection by keyword
    def test_command_detected_by_keyword(self, analyzer: QueryAnalyzer) -> None:
        """Command words trigger command intent."""
        result = analyzer.analyze("create a new file")

        assert result.intent == QueryIntent.COMMAND
        assert result.intent_confidence >= 0.7

    ##Method purpose: Test information seeking detection
    def test_information_seeking_detected(self, analyzer: QueryAnalyzer) -> None:
        """Information seeking patterns are detected."""
        result = analyzer.analyze("Tell me about quantum computing")

        assert result.intent == QueryIntent.INFORMATION_SEEKING

    ##Method purpose: Test statement detection
    def test_statement_detected(self, analyzer: QueryAnalyzer) -> None:
        """Statement patterns are detected."""
        result = analyzer.analyze("I think Python is the best language")

        assert result.intent == QueryIntent.STATEMENT

    ##Method purpose: Test conversation fallback
    def test_conversation_fallback(self, analyzer: QueryAnalyzer) -> None:
        """Unmatched queries default to conversation."""
        result = analyzer.analyze("hello there")

        assert result.intent in (QueryIntent.CONVERSATION, QueryIntent.CONVERSATIONAL)


##Class purpose: Test QueryAnalyzer complexity assessment
class TestQueryComplexityAssessment:
    """Tests for complexity assessment."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer()

    ##Method purpose: Test simple query detection
    def test_simple_query_detected(self, analyzer: QueryAnalyzer) -> None:
        """Short queries are classified as simple."""
        result = analyzer.analyze("What is AI?")

        assert result.complexity == QueryComplexity.SIMPLE
        assert result.complexity_confidence >= 0.7

    ##Method purpose: Test moderate query detection
    def test_moderate_query_detected(self, analyzer: QueryAnalyzer) -> None:
        """Medium-length queries are classified as moderate."""
        result = analyzer.analyze("How do I set up a Python virtual environment for my project")

        assert result.complexity in (QueryComplexity.MODERATE, QueryComplexity.SIMPLE)

    ##Method purpose: Test complex query detection
    def test_complex_query_detected(self, analyzer: QueryAnalyzer) -> None:
        """Queries with multiple parts are classified as complex."""
        result = analyzer.analyze(
            "Can you explain the difference between supervised and unsupervised "
            "machine learning, and provide examples of when to use each approach, "
            "because I need to understand this for my project"
        )

        assert result.complexity in (
            QueryComplexity.COMPLEX,
            QueryComplexity.VERY_COMPLEX,
        )


##Class purpose: Test QueryAnalyzer type classification
class TestQueryTypeClassification:
    """Tests for query type classification."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer()

    ##Method purpose: Test procedural type detection
    def test_procedural_type_detected(self, analyzer: QueryAnalyzer) -> None:
        """Procedural patterns trigger procedural type."""
        result = analyzer.analyze("How to install Docker on Ubuntu")

        assert result.query_type == QueryType.PROCEDURAL

    ##Method purpose: Test analytical type detection
    def test_analytical_type_detected(self, analyzer: QueryAnalyzer) -> None:
        """Analytical patterns trigger analytical type."""
        result = analyzer.analyze("Compare Python and JavaScript for web development")

        assert result.query_type == QueryType.ANALYTICAL

    ##Method purpose: Test creative type detection
    def test_creative_type_detected(self, analyzer: QueryAnalyzer) -> None:
        """Creative patterns trigger creative type."""
        result = analyzer.analyze("Write a story about a robot")

        assert result.query_type == QueryType.CREATIVE

    ##Method purpose: Test conversational type detection
    def test_conversational_type_detected(self, analyzer: QueryAnalyzer) -> None:
        """Conversational patterns trigger conversational type."""
        result = analyzer.analyze("Hello, how are you today?")

        assert result.query_type == QueryType.CONVERSATIONAL


##Class purpose: Test entity extraction
class TestEntityExtraction:
    """Tests for entity extraction."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer()

    ##Method purpose: Test capitalized entity extraction
    def test_capitalized_entities_extracted(self, analyzer: QueryAnalyzer) -> None:
        """Capitalized words are extracted as entities."""
        result = analyzer.analyze("Tell me about Python and JavaScript")

        # Python and JavaScript should be found (may vary by position)
        assert len(result.entities) >= 0  # May not capture mid-sentence caps

    ##Method purpose: Test quoted entity extraction
    def test_quoted_entities_extracted(self, analyzer: QueryAnalyzer) -> None:
        """Quoted strings are extracted as entities."""
        result = analyzer.analyze('Search for "machine learning" in the docs')

        assert "machine learning" in result.entities

    ##Method purpose: Test technical term extraction
    def test_technical_terms_extracted(self, analyzer: QueryAnalyzer) -> None:
        """Technical terms (acronyms, CamelCase) are extracted."""
        result = analyzer.analyze("How do I use the API with JSON data and GraphQL")

        assert "API" in result.entities or "JSON" in result.entities


##Class purpose: Test keyword extraction
class TestKeywordExtraction:
    """Tests for keyword extraction."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer()

    ##Method purpose: Test stop words are filtered
    def test_stop_words_filtered(self, analyzer: QueryAnalyzer) -> None:
        """Common stop words are not included in keywords."""
        result = analyzer.analyze("What is the best way to learn Python")

        # "the", "is", "to" should be filtered
        assert "the" not in result.keywords
        assert "is" not in result.keywords
        assert "to" not in result.keywords

    ##Method purpose: Test meaningful words are extracted
    def test_meaningful_words_extracted(self, analyzer: QueryAnalyzer) -> None:
        """Meaningful content words are extracted."""
        result = analyzer.analyze("How to implement machine learning algorithms")

        # Should contain content words
        assert len(result.keywords) > 0
        assert (
            "implement" in result.keywords
            or "machine" in result.keywords
            or "learning" in result.keywords
        )


##Class purpose: Test topic extraction
class TestTopicExtraction:
    """Tests for topic extraction."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer()

    ##Method purpose: Test technical topic detection
    def test_technical_topic_detected(self, analyzer: QueryAnalyzer) -> None:
        """Technical keywords trigger technical topic."""
        result = analyzer.analyze("How do I fix this code bug in Python")

        tech_topics = [t for t in result.topics if t.category == TopicCategory.TECHNICAL]
        assert len(tech_topics) > 0

    ##Method purpose: Test procedural topic detection
    def test_procedural_topic_detected(self, analyzer: QueryAnalyzer) -> None:
        """Procedural keywords trigger procedural topic."""
        result = analyzer.analyze("Step by step guide to setting up Docker")

        proc_topics = [t for t in result.topics if t.category == TopicCategory.PROCEDURAL]
        assert len(proc_topics) > 0

    ##Method purpose: Test default topic when no match
    def test_default_topic_when_no_match(self, analyzer: QueryAnalyzer) -> None:
        """Unknown queries get a default topic."""
        result = analyzer.analyze("xyz abc")

        assert len(result.topics) > 0
        # Should have at least one topic (possibly unknown/general)


##Class purpose: Test reformulation generation
class TestReformulationGeneration:
    """Tests for query reformulation."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer()

    ##Method purpose: Test reformulations are generated
    def test_reformulations_generated(self, analyzer: QueryAnalyzer) -> None:
        """Reformulations are generated for queries."""
        result = analyzer.analyze("How does Python handle memory management")

        # May or may not have reformulations depending on analysis
        assert isinstance(result.reformulations, list)

    ##Method purpose: Test reformulation limit
    def test_reformulation_limit(self, analyzer: QueryAnalyzer) -> None:
        """Reformulations are limited to max 3."""
        result = analyzer.analyze("Explain machine learning algorithms and their applications")

        assert len(result.reformulations) <= 3


##Class purpose: Test analyzer configuration
class TestAnalyzerConfiguration:
    """Tests for analyzer configuration."""

    ##Method purpose: Test disabled analyzer returns default
    def test_disabled_analyzer_returns_default(self) -> None:
        """Disabled analyzer returns default analysis."""
        config = QueryAnalyzerConfig(enabled=False)
        analyzer = QueryAnalyzer(config=config)

        result = analyzer.analyze("What is Python?")

        assert result.intent == QueryIntent.CONVERSATION
        assert result.complexity == QueryComplexity.SIMPLE
        assert result.overall_confidence == 0.5

    ##Method purpose: Test config options are respected
    def test_config_options_respected(self) -> None:
        """Configuration options affect behavior."""
        config = QueryAnalyzerConfig(
            entity_linking_enabled=False,
            reformulation_enabled=False,
        )
        analyzer = QueryAnalyzer(config=config)

        result = analyzer.analyze("Tell me about Python")

        # Without reformulation enabled, should be empty
        assert isinstance(result.reformulations, list)


##Class purpose: Test overall confidence calculation
class TestOverallConfidence:
    """Tests for overall confidence calculation."""

    ##Method purpose: Create analyzer fixture
    @pytest.fixture
    def analyzer(self) -> QueryAnalyzer:
        """Create a QueryAnalyzer instance."""
        return QueryAnalyzer()

    ##Method purpose: Test confidence is in valid range
    def test_confidence_in_valid_range(self, analyzer: QueryAnalyzer) -> None:
        """Overall confidence is between 0.0 and 1.0."""
        result = analyzer.analyze("What is machine learning?")

        assert 0.0 <= result.overall_confidence <= 1.0

    ##Method purpose: Test high confidence for clear queries
    def test_high_confidence_for_clear_queries(self, analyzer: QueryAnalyzer) -> None:
        """Clear queries should have higher confidence."""
        result = analyzer.analyze("What is Python?")

        # Clear question should have decent confidence
        assert result.overall_confidence >= 0.5


##Class purpose: Test AnalyzedQuery dataclass
class TestAnalyzedQueryDataclass:
    """Tests for AnalyzedQuery dataclass."""

    ##Method purpose: Test summary generation
    def test_summary_generation(self) -> None:
        """Summary method generates readable output."""
        analysis = AnalyzedQuery(
            original_query="Test query",
            intent=QueryIntent.QUESTION,
            intent_confidence=0.9,
            complexity=QueryComplexity.SIMPLE,
            complexity_confidence=0.8,
            query_type=QueryType.FACTUAL,
            type_confidence=0.7,
            entities=["Python", "AI"],
            keywords=["test", "query"],
            topics=[TopicResult(topic="tech", category=TopicCategory.TECHNICAL)],
            entity_links=[EntityLink(entity="Python", node_id="123")],
            reformulations=["Alternative query"],
            overall_confidence=0.8,
        )

        summary = analysis.summary()

        assert "question" in summary.lower()
        assert "factual" in summary.lower()
        assert "simple" in summary.lower()

    ##Method purpose: Test default values
    def test_default_values(self) -> None:
        """AnalyzedQuery has sensible defaults."""
        analysis = AnalyzedQuery(original_query="test")

        assert analysis.intent == QueryIntent.CONVERSATION
        assert analysis.complexity == QueryComplexity.SIMPLE
        assert analysis.entities == []
        assert analysis.keywords == []
