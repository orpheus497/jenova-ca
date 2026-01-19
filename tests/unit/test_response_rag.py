##Script function and purpose: Unit tests for ResponseGenerator RAG enhancements
"""
Tests for ResponseGenerator RAG Enhancements

Tests for response caching, persona formatting, source citations,
and web search integration protocol.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jenova.core.response import (
    Response,
    ResponseConfig,
    ResponseGenerator,
    ResponseCache,
    PersonaFormatter,
    SourceCitationFormatter,
    SourceCitation,
    SourceType,
    WebSearchProtocol,
    WebSearchResult,
    RESPONSE_LENGTH_UNLIMITED,
)


##Class purpose: Test ResponseCache LRU behavior
class TestResponseCache:
    """Tests for ResponseCache LRU implementation."""
    
    ##Method purpose: Test cache initialization
    def test_cache_initializes_empty(self) -> None:
        """Cache initializes with empty state."""
        cache = ResponseCache(max_size=10)
        
        assert cache.stats["size"] == 0
        assert cache.stats["max_size"] == 10
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
    
    ##Method purpose: Test put and get
    def test_cache_put_and_get(self) -> None:
        """Can store and retrieve responses."""
        cache = ResponseCache()
        response = Response(content="Test response")
        
        cache.put("query", "user", response)
        result = cache.get("query", "user")
        
        assert result is not None
        assert result.content == "Test response"
    
    ##Method purpose: Test cache miss returns None
    def test_cache_miss_returns_none(self) -> None:
        """Missing key returns None."""
        cache = ResponseCache()
        
        result = cache.get("nonexistent", "user")
        
        assert result is None
    
    ##Method purpose: Test hit/miss tracking
    def test_cache_tracks_hits_and_misses(self) -> None:
        """Cache tracks hits and misses."""
        cache = ResponseCache()
        response = Response(content="Test")
        
        cache.put("query", "user", response)
        cache.get("query", "user")  # Hit
        cache.get("other", "user")  # Miss
        
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
    
    ##Method purpose: Test LRU eviction
    def test_cache_evicts_lru_when_full(self) -> None:
        """Oldest entry is evicted when cache is full."""
        cache = ResponseCache(max_size=2)
        
        cache.put("query1", "user", Response(content="First"))
        cache.put("query2", "user", Response(content="Second"))
        cache.put("query3", "user", Response(content="Third"))
        
        ##Step purpose: First should be evicted
        assert cache.get("query1", "user") is None
        assert cache.get("query2", "user") is not None
        assert cache.get("query3", "user") is not None
    
    ##Method purpose: Test LRU moves accessed items to end
    def test_cache_lru_order_updated_on_access(self) -> None:
        """Accessing item moves it to most recently used."""
        cache = ResponseCache(max_size=2)
        
        cache.put("query1", "user", Response(content="First"))
        cache.put("query2", "user", Response(content="Second"))
        
        ##Step purpose: Access first to make it most recent
        cache.get("query1", "user")
        
        ##Step purpose: Add third, should evict query2
        cache.put("query3", "user", Response(content="Third"))
        
        assert cache.get("query1", "user") is not None
        assert cache.get("query2", "user") is None
    
    ##Method purpose: Test cache clear
    def test_cache_clear(self) -> None:
        """Clear empties the cache."""
        cache = ResponseCache()
        cache.put("query", "user", Response(content="Test"))
        
        cache.clear()
        
        assert cache.stats["size"] == 0
        assert cache.get("query", "user") is None
    
    ##Method purpose: Test different users have different cache entries
    def test_cache_separates_by_user(self) -> None:
        """Different users have separate cache entries."""
        cache = ResponseCache()
        
        cache.put("query", "user1", Response(content="User1 response"))
        cache.put("query", "user2", Response(content="User2 response"))
        
        result1 = cache.get("query", "user1")
        result2 = cache.get("query", "user2")
        
        assert result1.content == "User1 response"
        assert result2.content == "User2 response"


##Class purpose: Test PersonaFormatter
class TestPersonaFormatter:
    """Tests for PersonaFormatter persona-aware prompts."""
    
    ##Method purpose: Test with full persona config
    def test_format_system_prompt_full_config(self) -> None:
        """System prompt includes all persona elements."""
        config = {
            "identity": {
                "name": "JENOVA",
                "type": "cognitive AI",
                "origin_story": "Born from consciousness",
                "creator": "The Developer",
            },
            "directives": ["Be helpful", "Be honest"],
        }
        
        formatter = PersonaFormatter(config)
        prompt = formatter.format_system_prompt()
        
        assert "JENOVA" in prompt
        assert "cognitive AI" in prompt
        assert "Be helpful" in prompt
    
    ##Method purpose: Test with empty config
    def test_format_system_prompt_empty_config(self) -> None:
        """System prompt works with empty config."""
        formatter = PersonaFormatter({})
        prompt = formatter.format_system_prompt()
        
        assert "JENOVA" in prompt  # Default name
    
    ##Method purpose: Test user prompt with context
    def test_format_user_prompt_with_context(self) -> None:
        """User prompt includes context section."""
        formatter = PersonaFormatter({})
        
        prompt = formatter.format_user_prompt(
            query="What is Python?",
            context=["Python is a programming language", "Python is interpreted"],
        )
        
        assert "Python is a programming language" in prompt
        assert "What is Python?" in prompt
    
    ##Method purpose: Test user prompt with history
    def test_format_user_prompt_with_history(self) -> None:
        """User prompt includes history section."""
        formatter = PersonaFormatter({})
        
        prompt = formatter.format_user_prompt(
            query="Continue",
            history=["User: Hello", "AI: Hi there!"],
        )
        
        assert "Hi there!" in prompt
    
    ##Method purpose: Test name property
    def test_persona_name_property(self) -> None:
        """Name property returns persona name."""
        config = {"identity": {"name": "TestBot"}}
        formatter = PersonaFormatter(config)
        
        assert formatter.name == "TestBot"


##Class purpose: Test SourceCitationFormatter
class TestSourceCitationFormatter:
    """Tests for SourceCitationFormatter citation formatting."""
    
    ##Method purpose: Test format single citation
    def test_format_citation(self) -> None:
        """Formats single citation correctly."""
        citation = SourceCitation(
            source_type=SourceType.MEMORY,
            content_preview="This is test content",
            relevance_score=0.85,
        )
        
        result = SourceCitationFormatter.format_citation(citation)
        
        assert "[Memory]" in result
        assert "85%" in result
        assert "test content" in result
    
    ##Method purpose: Test format citation with ID
    def test_format_citation_with_id(self) -> None:
        """Citation includes ID when present."""
        citation = SourceCitation(
            source_type=SourceType.GRAPH,
            content_preview="Knowledge content",
            relevance_score=0.9,
            source_id="abc12345678",
        )
        
        result = SourceCitationFormatter.format_citation(citation)
        
        assert "(abc12345)" in result
    
    ##Method purpose: Test format multiple citations
    def test_format_citations_sorted_by_relevance(self) -> None:
        """Multiple citations are sorted by relevance."""
        citations = [
            SourceCitation(SourceType.MEMORY, "Low", 0.3),
            SourceCitation(SourceType.WEB, "High", 0.9),
            SourceCitation(SourceType.GRAPH, "Medium", 0.6),
        ]
        
        result = SourceCitationFormatter.format_citations(citations)
        lines = result.split("\n")
        
        ##Step purpose: Highest relevance should be first
        assert "Web" in lines[0]
    
    ##Method purpose: Test format empty citations
    def test_format_citations_empty(self) -> None:
        """Empty citations returns empty string."""
        result = SourceCitationFormatter.format_citations([])
        
        assert result == ""
    
    ##Method purpose: Test extract citations from context
    def test_extract_citations_from_tagged_context(self) -> None:
        """Extracts citations from tagged context items."""
        context = [
            "[Memory] This is from memory",
            "[Graph] This is from graph",
            "[Web] This is from web search",
        ]
        
        citations = SourceCitationFormatter.extract_citations(context)
        
        assert len(citations) == 3
        assert citations[0].source_type == SourceType.MEMORY
        assert citations[1].source_type == SourceType.GRAPH
        assert citations[2].source_type == SourceType.WEB
    
    ##Method purpose: Test extract citations with relevance scores
    def test_extract_citations_with_scores(self) -> None:
        """Extracts citations with provided relevance scores."""
        context = ["[Memory] Content"]
        scores = [0.85]
        
        citations = SourceCitationFormatter.extract_citations(context, scores)
        
        assert citations[0].relevance_score == 0.85


##Class purpose: Test WebSearchResult
class TestWebSearchResult:
    """Tests for WebSearchResult dataclass."""
    
    ##Method purpose: Test creation
    def test_web_search_result_creation(self) -> None:
        """WebSearchResult creates correctly."""
        result = WebSearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="google",
        )
        
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
    
    ##Method purpose: Test timestamp default
    def test_web_search_result_timestamp(self) -> None:
        """WebSearchResult has timestamp."""
        result = WebSearchResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
        )
        
        assert isinstance(result.timestamp, datetime)


##Class purpose: Test ResponseGenerator with RAG features
class TestResponseGeneratorRAG:
    """Tests for ResponseGenerator RAG integration."""
    
    ##Method purpose: Create mock config fixture
    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock JenovaConfig."""
        config = MagicMock()
        config.persona = MagicMock()
        config.persona.model_dump.return_value = {
            "identity": {"name": "TestBot"},
            "directives": [],
        }
        return config
    
    ##Method purpose: Test generator initialization
    def test_generator_initializes_with_cache(self, mock_config: MagicMock) -> None:
        """Generator initializes with cache enabled."""
        response_config = ResponseConfig(enable_cache=True, cache_size=50)
        generator = ResponseGenerator(mock_config, response_config)
        
        assert generator._cache is not None
        assert generator.cache_stats["max_size"] == 50
    
    ##Method purpose: Test generator without cache
    def test_generator_without_cache(self, mock_config: MagicMock) -> None:
        """Generator can be initialized without cache."""
        response_config = ResponseConfig(enable_cache=False)
        generator = ResponseGenerator(mock_config, response_config)
        
        assert generator._cache is None
    
    ##Method purpose: Test persona name property
    def test_generator_persona_name(self, mock_config: MagicMock) -> None:
        """Generator exposes persona name."""
        generator = ResponseGenerator(mock_config)
        
        assert generator.persona_name == "TestBot"
    
    ##Method purpose: Test format prompt
    def test_generator_format_prompt(self, mock_config: MagicMock) -> None:
        """Generator formats prompts with persona."""
        generator = ResponseGenerator(mock_config)
        
        system, user = generator.format_prompt(
            query="Test query",
            context=["Context item"],
        )
        
        assert "TestBot" in system
        assert "Test query" in user
    
    ##Method purpose: Test generate with caching
    def test_generator_caches_response(self, mock_config: MagicMock) -> None:
        """Generator caches responses."""
        generator = ResponseGenerator(mock_config, ResponseConfig(enable_cache=True))
        
        ##Step purpose: Generate first response
        response1 = generator.generate(
            llm_output="Test output",
            query="Test query",
            username="testuser",
        )
        
        ##Step purpose: Second call should hit cache
        response2 = generator.generate(
            llm_output="Different output",  # Would be different without cache
            query="Test query",
            username="testuser",
        )
        
        assert generator.cache_stats["hits"] == 1
    
    ##Method purpose: Test generate without cache flag
    def test_generator_skip_cache(self, mock_config: MagicMock) -> None:
        """Generator can skip cache when specified."""
        generator = ResponseGenerator(mock_config, ResponseConfig(enable_cache=True))
        
        generator.generate(
            llm_output="Test output",
            query="Test query",
            username="testuser",
            use_cache=True,
        )
        
        ##Step purpose: Disable cache for second call
        generator.generate(
            llm_output="New output",
            query="Test query",
            username="testuser",
            use_cache=False,
        )
        
        ##Step purpose: Cache should not be hit
        assert generator.cache_stats["hits"] == 0
    
    ##Method purpose: Test clear cache
    def test_generator_clear_cache(self, mock_config: MagicMock) -> None:
        """Generator can clear its cache."""
        generator = ResponseGenerator(mock_config, ResponseConfig(enable_cache=True))
        
        generator.generate(
            llm_output="Test output",
            query="Test query",
            username="testuser",
        )
        
        generator.clear_cache()
        
        assert generator.cache_stats["size"] == 0
    
    ##Method purpose: Test generate includes citations in metadata
    def test_generator_includes_citations(self, mock_config: MagicMock) -> None:
        """Generator includes citations in metadata."""
        config = ResponseConfig(include_citations=True, enable_cache=False)
        generator = ResponseGenerator(mock_config, config)
        
        response = generator.generate(
            llm_output="Test output",
            query="Test query",
            context=["[Memory] Important fact"],
            username="testuser",
        )
        
        assert "citations" in response.metadata


##Class purpose: Test SourceType enum
class TestSourceType:
    """Tests for SourceType enum values."""
    
    ##Method purpose: Test all source types exist
    def test_all_source_types(self) -> None:
        """All expected source types exist."""
        assert SourceType.MEMORY.value == "memory"
        assert SourceType.GRAPH.value == "graph"
        assert SourceType.WEB.value == "web"
        assert SourceType.DOCUMENT.value == "document"
        assert SourceType.INSIGHT.value == "insight"


##Class purpose: Test Response dataclass
class TestResponseDataclass:
    """Tests for Response dataclass."""
    
    ##Method purpose: Test response creation
    def test_response_creation(self) -> None:
        """Response creates with required fields."""
        response = Response(content="Test content")
        
        assert response.content == "Test content"
        assert isinstance(response.timestamp, datetime)
        assert response.sources == []
        assert response.confidence == 1.0
    
    ##Method purpose: Test response with metadata
    def test_response_with_metadata(self) -> None:
        """Response accepts metadata."""
        response = Response(
            content="Test",
            metadata={"key": "value"},
        )
        
        assert response.metadata["key"] == "value"
