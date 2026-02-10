##Script function and purpose: ResponseGenerator - Formats and structures LLM output into coherent responses.
##Dependency purpose: Takes raw LLM output and context, produces formatted responses with metadata.
"""ResponseGenerator formats LLM output into structured responses.

This module provides response formatting and post-processing:
- Formats raw LLM output
- Adds response metadata
- Handles response validation
- LRU response caching for performance
- Persona-aware prompt formatting
- Enhanced source citations
- Web search integration protocol
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from jenova.config.models import JenovaConfig

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Structured response with metadata
@dataclass
class Response:
    """A structured response from JENOVA.

    Attributes:
        content: The response content.
        timestamp: When the response was generated.
        sources: List of source references used.
        confidence: Confidence score (0.0 to 1.0).
        metadata: Additional response metadata.
    """

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


##Step purpose: Define constants for response generation limits
RESPONSE_LENGTH_UNLIMITED: int = 0  # 0 means no length limit on responses
MAX_CONTEXT_ITEMS: int = 10  # Maximum context items to include in prompt
MAX_WEB_RESULTS: int = 5  # Maximum web search results to include
MAX_CITATIONS: int = 5  # Maximum citations to display


##Class purpose: Enum for source types in citations
class SourceType(Enum):
    """Types of sources for citation formatting."""

    MEMORY = "memory"
    """From personal memory store."""

    GRAPH = "graph"
    """From cognitive graph."""

    WEB = "web"
    """From web search."""

    DOCUMENT = "document"
    """From uploaded document."""

    INSIGHT = "insight"
    """From generated insight."""


##Class purpose: Structured source citation
@dataclass
class SourceCitation:
    """A citation to a source used in response generation."""

    source_type: SourceType
    """Type of source."""

    content_preview: str
    """Preview of source content."""

    relevance_score: float = 0.0
    """How relevant this source was (0.0 to 1.0)."""

    source_id: str = ""
    """Optional unique source identifier."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional source metadata."""


##Class purpose: Web search result container
@dataclass
class WebSearchResult:
    """A web search result for RAG integration."""

    title: str
    """Page title."""

    url: str
    """Page URL."""

    snippet: str
    """Search result snippet."""

    source: str = ""
    """Search engine or API source."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When the result was retrieved."""


##Class purpose: Protocol for web search integration
class WebSearchProtocol(Protocol):
    """Protocol for web search integration.

    Implement this protocol to integrate custom web search providers.
    The ResponseGenerator can optionally use web search for RAG.
    """

    ##Method purpose: Search the web for relevant content
    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """
        Search the web for relevant content.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of WebSearchResult objects
        """
        ...

    ##Method purpose: Check if search is available
    def is_available(self) -> bool:
        """Check if web search is available."""
        ...


##Class purpose: LRU cache for response caching
class ResponseCache:
    """LRU cache for caching generated responses.

    Uses OrderedDict for O(1) operations while maintaining LRU order.
    Thread-safe via internal lock. All public methods are synchronized.
    """

    ##Method purpose: Initialize cache with max size
    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize response cache.

        Args:
            max_size: Maximum number of cached responses
        """
        ##Step purpose: Store configuration
        self._max_size = max_size
        self._cache: OrderedDict[str, Response] = OrderedDict()
        self._hits = 0
        self._misses = 0
        ##Sec: Add lock for thread-safe operations (P1-002 Daedelus audit)
        self._lock = threading.Lock()

    ##Method purpose: Generate cache key from query and context
    @staticmethod
    def _make_key(query: str, username: str, context_hash: str = "") -> str:
        """
        Generate cache key from query components.

        Args:
            query: User query
            username: Username
            context_hash: Optional hash of context

        Returns:
            Cache key string
        """
        ##Step purpose: Combine components and hash
        combined = f"{username}:{query}:{context_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    ##Method purpose: Get cached response if available
    def get(
        self,
        query: str,
        username: str,
        context_hash: str = "",
    ) -> Response | None:
        """
        Get cached response if available.

        Thread-safe: Uses internal lock for synchronization.

        Args:
            query: User query
            username: Username
            context_hash: Optional context hash

        Returns:
            Cached Response or None
        """
        key = self._make_key(query, username, context_hash)

        ##Sec: Acquire lock for thread-safe cache access (P1-002)
        with self._lock:
            ##Condition purpose: Check cache hit
            if key in self._cache:
                ##Step purpose: Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1

                logger.debug(
                    "cache_hit",
                    query_preview=query[:50],
                    cache_size=len(self._cache),
                )

                return self._cache[key]

            self._misses += 1
            return None

    ##Method purpose: Store response in cache
    def put(
        self,
        query: str,
        username: str,
        response: Response,
        context_hash: str = "",
    ) -> None:
        """
        Store response in cache.

        Thread-safe: Uses internal lock for synchronization.

        Args:
            query: User query
            username: Username
            response: Response to cache
            context_hash: Optional context hash
        """
        key = self._make_key(query, username, context_hash)

        ##Sec: Acquire lock for thread-safe cache modification (P1-002)
        with self._lock:
            ##Condition purpose: Evict oldest if at capacity
            if len(self._cache) >= self._max_size:
                ##Step purpose: Remove oldest (first) item
                self._cache.popitem(last=False)
                logger.debug("cache_eviction", cache_size=self._max_size)

            ##Step purpose: Add new entry
            self._cache[key] = response

    ##Method purpose: Clear entire cache
    def clear(self) -> None:
        """Clear all cached responses. Thread-safe."""
        ##Sec: Acquire lock for thread-safe cache clearing (P1-002)
        with self._lock:
            self._cache.clear()
            logger.debug("cache_cleared")

    ##Method purpose: Get cache statistics
    @property
    def stats(self) -> dict[str, int]:
        """Get cache statistics. Thread-safe."""
        ##Sec: Acquire lock for thread-safe stats access (P1-002)
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": int(hit_rate * 100),
            }

    ##Method purpose: Check if key exists
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache. Thread-safe."""
        ##Sec: Acquire lock for thread-safe containment check (P1-002)
        with self._lock:
            return key in self._cache


##Class purpose: Formats prompts with persona context
class PersonaFormatter:
    """Formats prompts with persona-aware context.

    Injects persona identity, directives, and style into prompts
    for consistent personality in responses.
    """

    ##Method purpose: Initialize with persona config
    def __init__(self, persona_config: dict[str, object]) -> None:
        """
        Initialize persona formatter.

        Args:
            persona_config: Persona configuration dict with identity/directives
        """
        ##Step purpose: Extract persona components with type safety
        identity_raw = persona_config.get("identity", {})

        ##Fix: Cast to dict after type check for mypy compliance (BUG-RESPONSE-001)
        if isinstance(identity_raw, dict):
            identity: dict[str, object] = identity_raw
            self._name = str(identity.get("name", "JENOVA"))
            self._type = str(identity.get("type", "AI assistant"))
            self._origin = str(identity.get("origin_story", ""))
            self._creator = str(identity.get("creator", ""))
        else:
            self._name = "JENOVA"
            self._type = "AI assistant"
            self._origin = ""
            self._creator = ""

        directives = persona_config.get("directives", [])
        self._directives = directives if isinstance(directives, list) else []

    ##Method purpose: Format system prompt with persona
    def format_system_prompt(self) -> str:
        """
        Generate persona-aware system prompt.

        Returns:
            Formatted system prompt string
        """
        ##Step purpose: Build directive section
        directive_text = ""
        if self._directives:
            directive_text = "\n\nYour core directives:\n" + "\n".join(
                f"- {d}" for d in self._directives
            )

        ##Step purpose: Build origin section
        origin_text = ""
        if self._origin:
            origin_text = f"\n\nYour origin: {self._origin}"

        return f"""You are {self._name}, a {self._type}.{origin_text}{directive_text}

Respond naturally and helpfully while maintaining your identity."""

    ##Method purpose: Format user message with context
    def format_user_prompt(
        self,
        query: str,
        context: list[str] | None = None,
        history: list[str] | None = None,
        web_results: list[WebSearchResult] | None = None,
    ) -> str:
        """
        Format user message with context.

        Args:
            query: User query
            context: Retrieved context items
            history: Conversation history
            web_results: Optional web search results

        Returns:
            Formatted user prompt string
        """
        sections: list[str] = []

        ##Condition purpose: Add context section
        if context:
            context_text = "\n".join(f"- {c}" for c in context[:MAX_CONTEXT_ITEMS])
            sections.append(f"## Retrieved Context\n{context_text}")

        ##Condition purpose: Add web results section
        if web_results:
            web_text = "\n".join(
                f"- [{r.title}]({r.url}): {r.snippet[:100]}" for r in web_results[:MAX_WEB_RESULTS]
            )
            sections.append(f"## Web Search Results\n{web_text}")

        ##Condition purpose: Add history section
        if history:
            history_text = "\n".join(history[-5:])
            sections.append(f"## Recent Conversation\n{history_text}")

        ##Step purpose: Combine sections
        context_block = "\n\n".join(sections) if sections else ""

        ##Condition purpose: Include context if present
        if context_block:
            return f"""{context_block}

## User Query
{query}"""

        return query

    ##Method purpose: Get persona name
    @property
    def name(self) -> str:
        """Get persona name."""
        return self._name


##Class purpose: Formats source citations for responses
class SourceCitationFormatter:
    """Formats source citations for inclusion in responses.

    Creates consistent, readable citations from various source types.
    """

    ##Method purpose: Format a single citation
    @staticmethod
    def format_citation(citation: SourceCitation) -> str:
        """
        Format a single citation for display.

        Args:
            citation: Citation to format

        Returns:
            Formatted citation string
        """
        ##Step purpose: Build type indicator
        type_labels = {
            SourceType.MEMORY: "Memory",
            SourceType.GRAPH: "Knowledge",
            SourceType.WEB: "Web",
            SourceType.DOCUMENT: "Document",
            SourceType.INSIGHT: "Insight",
        }
        type_label = type_labels.get(citation.source_type, "Source")

        ##Step purpose: Format with relevance
        relevance_pct = int(citation.relevance_score * 100)

        ##Condition purpose: Include source ID if present
        id_part = f" ({citation.source_id[:8]})" if citation.source_id else ""

        return f"[{type_label}{id_part}] {citation.content_preview[:80]}... ({relevance_pct}% relevant)"

    ##Method purpose: Format multiple citations
    @staticmethod
    def format_citations(citations: list[SourceCitation]) -> str:
        """
        Format multiple citations as a block.

        Args:
            citations: Citations to format

        Returns:
            Formatted citations block
        """
        ##Condition purpose: Handle empty case
        if not citations:
            return ""

        ##Step purpose: Sort by relevance and format
        sorted_citations = sorted(citations, key=lambda c: c.relevance_score, reverse=True)

        formatted = [
            SourceCitationFormatter.format_citation(c) for c in sorted_citations[:MAX_CITATIONS]
        ]

        return "\n".join(formatted)

    ##Method purpose: Extract citations from context items
    @staticmethod
    def extract_citations(
        context: list[str],
        relevance_scores: list[float] | None = None,
    ) -> list[SourceCitation]:
        """
        Extract citations from context items.

        Args:
            context: Context items with optional type tags
            relevance_scores: Optional scores per item

        Returns:
            List of SourceCitation objects
        """
        citations: list[SourceCitation] = []

        ##Step purpose: Pattern to match source type tags
        tag_pattern = re.compile(r"^\[(\w+)\]\s*(.+)$", re.DOTALL)

        ##Step purpose: Type mapping
        type_map = {
            "memory": SourceType.MEMORY,
            "episodic": SourceType.MEMORY,
            "semantic": SourceType.MEMORY,
            "procedural": SourceType.MEMORY,
            "graph": SourceType.GRAPH,
            "knowledge": SourceType.GRAPH,
            "web": SourceType.WEB,
            "search": SourceType.WEB,
            "document": SourceType.DOCUMENT,
            "doc": SourceType.DOCUMENT,
            "insight": SourceType.INSIGHT,
        }

        ##Loop purpose: Extract from each context item
        for i, item in enumerate(context):
            match = tag_pattern.match(item)

            ##Condition purpose: Check for tagged format
            if match:
                tag = match.group(1).lower()
                content = match.group(2).strip()
                source_type = type_map.get(tag, SourceType.MEMORY)
            else:
                content = item
                source_type = SourceType.MEMORY

            ##Step purpose: Get relevance score
            score = relevance_scores[i] if relevance_scores and i < len(relevance_scores) else 0.5

            citations.append(
                SourceCitation(
                    source_type=source_type,
                    content_preview=content[:100],
                    relevance_score=score,
                )
            )

        return citations


##Class purpose: Configuration for response generation
@dataclass
class ResponseConfig:
    """Configuration for ResponseGenerator.

    Attributes:
        include_sources: Whether to include source references.
        max_length: Maximum response length. Use RESPONSE_LENGTH_UNLIMITED (0) for no limit.
        format_style: Response formatting style.
        enable_cache: Whether to enable response caching.
        cache_size: Maximum cache size.
        include_citations: Whether to include formatted citations.
    """

    include_sources: bool = True
    max_length: int = RESPONSE_LENGTH_UNLIMITED
    format_style: str = "default"
    enable_cache: bool = True
    cache_size: int = 100
    include_citations: bool = True


##Class purpose: Formats and structures LLM output into responses
class ResponseGenerator:
    """Generates structured responses from LLM output.

    The ResponseGenerator handles:
    - Raw LLM output formatting
    - Source attribution
    - Response validation
    - Metadata enrichment
    - LRU response caching
    - Persona-aware prompt formatting

    Attributes:
        config: Response configuration.
    """

    ##Fix: Add explicit type annotation for optional cache (BUG-RESPONSE-002)
    _cache: ResponseCache | None

    ##Method purpose: Initialize with configuration
    def __init__(
        self,
        config: JenovaConfig,
        response_config: ResponseConfig | None = None,
        web_search: WebSearchProtocol | None = None,
    ) -> None:
        """Initialize the ResponseGenerator.

        Args:
            config: JENOVA configuration.
            response_config: Optional response-specific configuration.
            web_search: Optional web search provider.
        """
        ##Step purpose: Store configuration
        self._config = config
        self._response_config = response_config or ResponseConfig()
        self._web_search = web_search

        ##Step purpose: Initialize cache if enabled
        if self._response_config.enable_cache:
            self._cache = ResponseCache(max_size=self._response_config.cache_size)
        else:
            self._cache = None

        ##Step purpose: Initialize persona formatter
        persona_dict = getattr(config, "persona", None)
        if persona_dict and hasattr(persona_dict, "model_dump"):
            self._persona = PersonaFormatter(persona_dict.model_dump())
        elif isinstance(persona_dict, dict):
            self._persona = PersonaFormatter(persona_dict)
        else:
            self._persona = PersonaFormatter({})

        ##Step purpose: Initialize citation formatter
        self._citation_formatter = SourceCitationFormatter()

        ##Action purpose: Log initialization
        logger.debug(
            "response_generator_initialized",
            include_sources=self._response_config.include_sources,
            format_style=self._response_config.format_style,
            cache_enabled=self._cache is not None,
        )

    ##Method purpose: Get cache statistics
    @property
    def cache_stats(self) -> dict[str, int | bool]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics. When cache is disabled,
            returns {"enabled": False}. When enabled, returns stats dict
            with int values.
        """
        if self._cache:
            return self._cache.stats
        return {"enabled": False}

    ##Method purpose: Get persona name
    @property
    def persona_name(self) -> str:
        """Get the persona name."""
        return self._persona.name

    ##Method purpose: Clear response cache
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache:
            self._cache.clear()

    ##Method purpose: Format prompt with persona context
    def format_prompt(
        self,
        query: str,
        context: list[str] | None = None,
        history: list[str] | None = None,
        include_web_search: bool = False,
    ) -> tuple[str, str]:
        """
        Format a complete prompt with persona and context.

        Args:
            query: User query
            context: Retrieved context items
            history: Conversation history
            include_web_search: Whether to perform web search

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        ##Step purpose: Get web results if enabled and available
        web_results: list[WebSearchResult] | None = None
        if include_web_search and self._web_search and self._web_search.is_available():
            ##Error purpose: Handle web search errors gracefully
            try:
                web_results = self._web_search.search(query, max_results=MAX_WEB_RESULTS)
            except Exception as e:
                logger.warning("web_search_failed", error=str(e))

        system_prompt = self._persona.format_system_prompt()
        user_prompt = self._persona.format_user_prompt(
            query=query,
            context=context,
            history=history,
            web_results=web_results,
        )

        return system_prompt, user_prompt

    ##Method purpose: Generate a formatted response from LLM output
    def generate(
        self,
        llm_output: str,
        context: list[str] | None = None,
        query: str | None = None,
        username: str = "",
        use_cache: bool = True,
        relevance_scores: list[float] | None = None,
    ) -> Response:
        """Generate a structured response from LLM output.

        Args:
            llm_output: Raw output from the LLM.
            context: Optional list of context items used.
            query: Optional original query for reference.
            username: Username for cache key.
            use_cache: Whether to use/store in cache.
            relevance_scores: Optional relevance scores for context items.

        Returns:
            Structured Response object.

        Raises:
            ResponseError: If response generation fails.
        """
        ##Step purpose: Compute context hash once (reused for get and put)
        context_hash = ""
        if use_cache and self._cache and query and context:
            context_hash = hashlib.sha256(str(context).encode()).hexdigest()[:16]

        ##Condition purpose: Check cache for existing response
        if use_cache and self._cache and query:
            cached = self._cache.get(query, username, context_hash)
            if cached:
                return cached

        ##Step purpose: Format the content according to style
        formatted_content = self._format_content(llm_output)

        ##Step purpose: Validate the response
        is_valid = self._validate_response(formatted_content)

        ##Condition purpose: Handle invalid responses
        if not is_valid:
            logger.warning("invalid_response_generated", content_length=len(formatted_content))
            formatted_content = self._get_fallback_response()

        ##Step purpose: Extract sources if configured
        sources: list[str] = []
        if self._response_config.include_sources and context:
            sources = self._extract_sources(context)

        ##Step purpose: Extract and format citations if configured
        citations_text = ""
        if self._response_config.include_citations and context:
            citations = self._citation_formatter.extract_citations(context, relevance_scores)
            citations_text = self._citation_formatter.format_citations(citations)

        ##Step purpose: Calculate confidence based on response quality
        confidence = self._calculate_confidence(formatted_content, context)

        ##Step purpose: Build metadata
        metadata: dict[str, str] = {}
        if query:
            metadata["query"] = query[:100]
        metadata["format_style"] = self._response_config.format_style
        metadata["persona"] = self._persona.name
        if citations_text:
            metadata["citations"] = citations_text

        ##Action purpose: Log response generation
        logger.debug(
            "response_generated",
            content_length=len(formatted_content),
            sources_count=len(sources),
            confidence=confidence,
        )

        response = Response(
            content=formatted_content,
            sources=sources,
            confidence=confidence,
            metadata=metadata,
        )

        ##Condition purpose: Store in cache if enabled (reuse pre-computed hash)
        if use_cache and self._cache and query:
            self._cache.put(query, username, response, context_hash)

        return response

    ##Method purpose: Generate response with optional web search integration
    def generate_with_rag(
        self,
        llm_output: str,
        query: str,
        context: list[str] | None = None,
        username: str = "",
        history: list[str] | None = None,
        include_web: bool = False,
    ) -> Response:
        """
        Generate response with full RAG features.

        Combines caching, persona formatting, citations, and optional web search.

        Args:
            llm_output: Raw LLM output
            query: User query
            context: Retrieved context items
            username: Username
            history: Conversation history
            include_web: Whether to include web search

        Returns:
            Structured Response with RAG enhancements
        """
        ##Step purpose: Get web results if requested
        web_context: list[str] = []
        if include_web and self._web_search and self._web_search.is_available():
            ##Error purpose: Handle web search errors gracefully
            try:
                results = self._web_search.search(query, max_results=MAX_WEB_RESULTS)
                web_context = [f"[Web] {r.title}: {r.snippet}" for r in results]
            except Exception as e:
                logger.warning("web_search_failed", error=str(e))

        ##Step purpose: Combine context sources
        full_context = (context or []) + web_context

        ##Step purpose: Generate with full context
        return self.generate(
            llm_output=llm_output,
            context=full_context,
            query=query,
            username=username,
            use_cache=True,
        )

    ##Method purpose: Validate response content
    def _validate_response(self, content: str) -> bool:
        """Validate response content.

        Args:
            content: The response content to validate.

        Returns:
            True if valid, False otherwise.
        """
        ##Condition purpose: Check for empty content
        if not content or not content.strip():
            return False

        ##Condition purpose: Check minimum length
        if len(content.strip()) < 5:
            return False

        ##Condition purpose: Check for error patterns
        error_patterns = [
            r"^error:",
            r"^exception:",
            r"failed to generate",
            r"^\s*null\s*$",
            r"^\s*undefined\s*$",
        ]
        content_lower = content.lower()
        ##Loop purpose: Check each error pattern
        return all(not re.search(pattern, content_lower) for pattern in error_patterns)

    ##Method purpose: Format response according to style
    def _format_content(self, content: str) -> str:
        """Format content according to configured style.

        Args:
            content: Raw content to format.

        Returns:
            Formatted content string.
        """
        ##Step purpose: Strip whitespace
        formatted = content.strip()

        ##Condition purpose: Apply max length if configured (skip if RESPONSE_LENGTH_UNLIMITED)
        if (
            self._response_config.max_length != RESPONSE_LENGTH_UNLIMITED
            and len(formatted) > self._response_config.max_length
        ):
            ##Step purpose: Truncate at sentence boundary if possible
            truncated = formatted[: self._response_config.max_length]
            last_period = truncated.rfind(".")
            if last_period > self._response_config.max_length * 0.7:
                formatted = truncated[: last_period + 1]
            else:
                formatted = truncated + "..."

        ##Condition purpose: Apply style-specific formatting
        if self._response_config.format_style == "minimal":
            ##Step purpose: Remove extra whitespace for minimal style
            formatted = " ".join(formatted.split())

        return formatted

    ##Method purpose: Extract source references from context
    def _extract_sources(self, context: list[str]) -> list[str]:
        """Extract source references from context items.

        Args:
            context: List of context items.

        Returns:
            List of source reference strings.
        """
        sources: list[str] = []

        ##Step purpose: Pattern to match source type tags
        source_pattern = re.compile(r"^\[(\w+)\]")

        ##Loop purpose: Extract source type from each context item
        for item in context:
            match = source_pattern.match(item)
            if match:
                source_type = match.group(1)
                ##Condition purpose: Avoid duplicate source types
                if source_type not in sources:
                    sources.append(source_type)

        return sources

    ##Method purpose: Calculate confidence score for response
    def _calculate_confidence(
        self,
        content: str,
        context: list[str] | None,
    ) -> float:
        """Calculate confidence score based on response quality.

        Args:
            content: The response content.
            context: Context items used.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        confidence = 1.0

        ##Condition purpose: Lower confidence if no context was used
        if not context:
            confidence *= 0.8

        ##Condition purpose: Lower confidence for very short responses
        if len(content) < 50:
            confidence *= 0.9

        ##Condition purpose: Lower confidence if response seems uncertain
        uncertainty_phrases = ["i'm not sure", "i think", "maybe", "possibly"]
        content_lower = content.lower()
        ##Loop purpose: Check for uncertainty markers
        for phrase in uncertainty_phrases:
            if phrase in content_lower:
                confidence *= 0.85
                break

        return round(confidence, 2)

    ##Method purpose: Get fallback response for invalid LLM output
    def _get_fallback_response(self) -> str:
        """Get a fallback response when LLM output is invalid."""
        return "I apologize, but I wasn't able to generate a proper response. Could you please rephrase your question?"
