##Script function and purpose: Query Analysis for user intent, entity extraction, and complexity scoring
"""
Query Analyzer

Analyzes user queries to extract structured information for enhanced comprehension.
Provides intent classification, entity extraction, complexity scoring, topic modeling,
and query reformulation capabilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import structlog

from jenova.utils.cache import TTLCache
from jenova.utils.json_safe import JSONSizeError, extract_json_from_response, safe_json_loads

if TYPE_CHECKING:
    from jenova.graph import CognitiveGraph

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Enum defining query intent types for classification
class QueryIntent(Enum):
    """Classification of user query intent."""

    QUESTION = "question"
    COMMAND = "command"
    CONVERSATION = "conversation"
    INFORMATION_SEEKING = "information_seeking"
    CLARIFICATION = "clarification"
    STATEMENT = "statement"


##Class purpose: Enum defining query complexity levels for planning adaptation
class QueryComplexity(Enum):
    """Complexity level of a user query."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


##Class purpose: Enum defining query type categories
class QueryType(Enum):
    """Type classification for queries."""

    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"


##Class purpose: Enum defining topic categories for topic modeling
class TopicCategory(Enum):
    """Category classification for identified topics."""

    TECHNICAL = "technical"
    PERSONAL = "personal"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


##Class purpose: Store topic modeling results with confidence scores
@dataclass
class TopicResult:
    """Represents an extracted topic with category and confidence."""

    topic: str
    """The topic name or label."""

    category: TopicCategory = TopicCategory.UNKNOWN
    """Topic category classification."""

    confidence: float = 0.5
    """Confidence score (0.0 to 1.0)."""

    keywords: list[str] = field(default_factory=list)
    """Related keywords for this topic."""


##Class purpose: Store entity link information connecting entities to Cortex nodes
@dataclass
class EntityLink:
    """Represents a link between an extracted entity and a graph node."""

    entity: str
    """The extracted entity text."""

    node_id: str | None = None
    """ID of linked graph node (None if unlinked)."""

    node_type: str | None = None
    """Type of the linked node."""

    confidence: float = 0.0
    """Confidence in the link (0.0 to 1.0)."""

    relationship: str = "related_to"
    """Type of relationship to the node."""


##Class purpose: Complete analysis result for a query
@dataclass
class AnalyzedQuery:
    """Complete analysis result for a user query."""

    original_query: str
    """The original query text."""

    intent: QueryIntent = QueryIntent.CONVERSATION
    """Classified query intent."""

    intent_confidence: float = 0.5
    """Confidence in intent classification (0.0 to 1.0)."""

    complexity: QueryComplexity = QueryComplexity.SIMPLE
    """Query complexity level."""

    complexity_confidence: float = 0.5
    """Confidence in complexity assessment (0.0 to 1.0)."""

    query_type: QueryType = QueryType.FACTUAL
    """Query type classification."""

    type_confidence: float = 0.5
    """Confidence in type classification (0.0 to 1.0)."""

    entities: list[str] = field(default_factory=list)
    """Extracted entities (people, places, concepts)."""

    keywords: list[str] = field(default_factory=list)
    """Key terms and phrases."""

    topics: list[TopicResult] = field(default_factory=list)
    """Identified topics with categories."""

    entity_links: list[EntityLink] = field(default_factory=list)
    """Links between entities and graph nodes."""

    reformulations: list[str] = field(default_factory=list)
    """Alternative phrasings of the query."""

    overall_confidence: float = 0.5
    """Overall analysis confidence (0.0 to 1.0)."""

    ##Method purpose: Get summary string for logging
    def summary(self) -> str:
        """Generate human-readable summary of query analysis."""
        topic_str = ", ".join(t.topic for t in self.topics[:3]) if self.topics else "none"
        linked_count = sum(1 for el in self.entity_links if el.node_id)

        return (
            f"Intent: {self.intent.value} ({self.intent_confidence:.0%}) | "
            f"Type: {self.query_type.value} | Complexity: {self.complexity.value} | "
            f"Topics: {topic_str} | "
            f"Entities: {len(self.entities)} ({linked_count} linked) | "
            f"Confidence: {self.overall_confidence:.0%}"
        )


##Class purpose: Configuration for query analyzer
@dataclass
class QueryAnalyzerConfig:
    """Configuration for QueryAnalyzer behavior."""

    enabled: bool = True
    """Whether query analysis is enabled."""

    use_llm: bool = False
    """Whether to use LLM for analysis (more accurate but slower)."""

    topic_modeling_enabled: bool = True
    """Whether to extract topics from queries."""

    entity_linking_enabled: bool = True
    """Whether to link entities to graph nodes."""

    reformulation_enabled: bool = True
    """Whether to generate query reformulations."""

    confidence_scoring_enabled: bool = True
    """Whether to compute confidence scores."""

    min_entity_match_score: float = 0.3
    """Minimum score for entity-node matching."""

    max_entity_links: int = 5
    """Maximum entity links to return per entity."""


##Class purpose: Protocol for LLM interface to avoid circular imports
class LLMProtocol(Protocol):
    """Protocol defining LLM interface for query analysis."""

    ##Method purpose: Generate text from prompt
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        ...


##Class purpose: Analyzes user queries to extract structured information
class QueryAnalyzer:
    """
    Analyzes user queries for intent, entities, complexity, and topics.

    Provides structured analysis to improve context retrieval and
    response generation. Can link entities to graph nodes when
    a CognitiveGraph is provided.

    Supports two modes:
    - Heuristic mode (default): Fast, rule-based analysis
    - LLM mode: More accurate analysis using language model
    """

    ##Method purpose: Initialize query analyzer with configuration
    def __init__(
        self,
        config: QueryAnalyzerConfig | None = None,
        llm: LLMProtocol | None = None,
    ) -> None:
        """
        Initialize query analyzer.

        Args:
            config: Analyzer configuration
            llm: Optional LLM for advanced analysis
        """
        ##Step purpose: Store configuration
        self._config = config or QueryAnalyzerConfig()
        self._llm = llm

        ##Step purpose: Initialize optional graph reference for entity linking
        self._graph: CognitiveGraph | None = None

        ##Step purpose: Initialize username for per-user filtering
        self._username: str | None = None

        ##Update: Add entity match cache for performance
        self._entity_match_cache: TTLCache[str, list[dict[str, str | float]]] = TTLCache(
            max_size=1000,
            default_ttl=600,  # 10 minutes
        )

        logger.debug("query_analyzer_initialized", config=self._config)

    ##Method purpose: Set graph reference for entity linking
    def set_graph(self, graph: CognitiveGraph) -> None:
        """
        Set CognitiveGraph reference for entity linking.

        Args:
            graph: The cognitive graph to use for entity linking
        """
        self._graph = graph
        logger.debug("graph_set_for_entity_linking")

    ##Method purpose: Set username for per-user entity filtering
    def set_username(self, username: str) -> None:
        """
        Set current username for per-user entity filtering.

        Args:
            username: The username to filter entities by
        """
        self._username = username
        logger.debug("username_set", username=username)

    ##Method purpose: Get current username
    def get_username(self) -> str | None:
        """
        Get current username for entity filtering.

        Returns:
            Current username or None if not set
        """
        return self._username

    ##Method purpose: Analyze query to extract structured information
    def analyze(self, query: str) -> AnalyzedQuery:
        """
        Comprehensive query analysis.

        Args:
            query: The user query to analyze

        Returns:
            AnalyzedQuery with all extracted information
        """
        ##Condition purpose: Return default analysis if disabled
        if not self._config.enabled:
            logger.debug("query_analysis_disabled")
            return self._default_analysis(query)

        ##Condition purpose: Use LLM analysis if configured and LLM available
        if self._config.use_llm and self._llm is not None:
            analysis = self._llm_analysis(query)
        else:
            ##Step purpose: Perform heuristic-based analysis
            analysis = self._heuristic_analysis(query)

        ##Condition purpose: Perform entity linking if graph available
        if self._config.entity_linking_enabled and self._graph is not None:
            analysis.entity_links = self._link_entities_to_graph(analysis.entities)

        ##Condition purpose: Generate reformulations if enabled
        if self._config.reformulation_enabled:
            analysis.reformulations = self.generate_reformulations(query, analysis)

        logger.info(
            "query_analyzed",
            intent=analysis.intent.value,
            complexity=analysis.complexity.value,
            entity_count=len(analysis.entities),
            confidence=analysis.overall_confidence,
        )

        return analysis

    ##Method purpose: Perform LLM-based query analysis
    def _llm_analysis(self, query: str) -> AnalyzedQuery:
        """
        Analyze query using LLM for more accurate results.

        Args:
            query: Query text to analyze

        Returns:
            AnalyzedQuery with LLM-based analysis
        """
        ##Condition purpose: Fall back to heuristics if no LLM
        if self._llm is None:
            return self._heuristic_analysis(query)

        ##Step purpose: Build analysis prompt
        intent_options = [e.value for e in QueryIntent]
        complexity_options = [e.value for e in QueryComplexity]
        topic_options = [e.value for e in TopicCategory]

        prompt = f"""Analyze the following user query and extract structured information:

1. Intent: Classify the query intent as one of: {intent_options}
2. Intent Confidence: Rate confidence in intent classification (0.0 to 1.0)
3. Entities: Extract a list of key entities (people, places, concepts, temporal references, objects)
4. Complexity: Assess complexity as one of: {complexity_options}
5. Complexity Confidence: Rate confidence in complexity assessment (0.0 to 1.0)
6. Query type: Classify as one of: factual, procedural, analytical, creative, conversational
7. Type Confidence: Rate confidence in type classification (0.0 to 1.0)
8. Keywords: Extract 3-5 key terms or phrases
9. Topics: Identify main topics discussed (1-3 topics)
10. Topic Categories: Classify each topic as one of: {topic_options}
11. Alternative Phrasings: Suggest 2-3 alternative ways to phrase this query

Query: "{query}"

Respond with a valid JSON object:
{{
    "intent": "<one of {intent_options}>",
    "intent_confidence": 0.0-1.0,
    "entities": ["entity1", "entity2"],
    "complexity": "<one of {complexity_options}>",
    "complexity_confidence": 0.0-1.0,
    "type": "<factual|procedural|analytical|creative|conversational>",
    "type_confidence": 0.0-1.0,
    "keywords": ["keyword1", "keyword2"],
    "topics": [
        {{"topic": "topic_name", "category": "<one of {topic_options}>", "confidence": 0.0-1.0}}
    ],
    "reformulations": ["alternative phrasing 1", "alternative phrasing 2"]
}}"""

        ##Error purpose: Handle LLM errors gracefully
        try:
            response = self._llm.generate(prompt, temperature=0.2)
            return self._parse_llm_response(response, query)
        except Exception as e:
            logger.warning("llm_analysis_failed", error=str(e), query=query[:50])
            return self._heuristic_analysis(query)

    ##Method purpose: Parse LLM JSON response into AnalyzedQuery
    def _parse_llm_response(self, response: str, query: str) -> AnalyzedQuery:
        """
        Parse LLM JSON response into structured analysis.

        Args:
            response: LLM response string (should be JSON)
            query: Original query for fallback

        Returns:
            AnalyzedQuery parsed from response
        """
        import json

        ##Error purpose: Handle JSON parse errors
        try:
            ##Step purpose: Extract JSON from response
            try:
                json_str = extract_json_from_response(response)
            except ValueError:
                return self._heuristic_analysis(query)

            ##Action purpose: Parse with size limits
            data = safe_json_loads(json_str)
        except (json.JSONDecodeError, JSONSizeError):
            return self._heuristic_analysis(query)

        ##Condition purpose: Handle invalid data
        if not isinstance(data, dict):
            return self._heuristic_analysis(query)

        ##Step purpose: Parse and validate intent
        intent_str = data.get("intent", "conversation")
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.CONVERSATION
        intent_conf = self._validate_confidence(data.get("intent_confidence", 0.5))

        ##Step purpose: Parse and validate complexity
        complexity_str = data.get("complexity", "simple")
        try:
            complexity = QueryComplexity(complexity_str)
        except ValueError:
            complexity = QueryComplexity.SIMPLE
        complexity_conf = self._validate_confidence(data.get("complexity_confidence", 0.5))

        ##Step purpose: Parse and validate type
        type_str = data.get("type", "factual")
        try:
            query_type = QueryType(type_str)
        except ValueError:
            query_type = QueryType.FACTUAL
        type_conf = self._validate_confidence(data.get("type_confidence", 0.5))

        ##Step purpose: Parse entities and keywords
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        entities = [e for e in entities if isinstance(e, str)]

        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [k for k in keywords if isinstance(k, str)]

        ##Step purpose: Parse topics
        topics = self._parse_llm_topics(data.get("topics", []))

        ##Step purpose: Parse reformulations
        reformulations = data.get("reformulations", [])
        if not isinstance(reformulations, list):
            reformulations = []
        reformulations = [r for r in reformulations if isinstance(r, str) and r.strip()]

        ##Step purpose: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            intent_conf, complexity_conf, type_conf
        )

        return AnalyzedQuery(
            original_query=query,
            intent=intent,
            intent_confidence=intent_conf,
            complexity=complexity,
            complexity_confidence=complexity_conf,
            query_type=query_type,
            type_confidence=type_conf,
            entities=entities[:10],
            keywords=keywords[:10],
            topics=topics,
            entity_links=[],
            reformulations=reformulations[:3],
            overall_confidence=overall_confidence,
        )

    ##Method purpose: Validate confidence score to 0.0-1.0 range
    def _validate_confidence(self, value: object) -> float:
        """
        Validate and normalize confidence score.

        Args:
            value: Raw confidence value

        Returns:
            Confidence clamped to 0.0-1.0
        """
        try:
            conf = float(value)  # type: ignore[arg-type]
            return max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            return 0.5

    ##Method purpose: Parse topics from LLM response
    def _parse_llm_topics(self, topics_data: object) -> list[TopicResult]:
        """
        Parse topics from LLM response.

        Args:
            topics_data: Raw topics data from LLM

        Returns:
            List of TopicResult objects
        """
        if not isinstance(topics_data, list):
            return []

        topics: list[TopicResult] = []

        ##Loop purpose: Parse each topic
        for item in topics_data:
            if isinstance(item, dict):
                topic_name = item.get("topic", "")
                if not isinstance(topic_name, str) or not topic_name:
                    continue

                category_str = item.get("category", "unknown")
                try:
                    category = TopicCategory(category_str)
                except ValueError:
                    category = TopicCategory.UNKNOWN

                confidence = self._validate_confidence(item.get("confidence", 0.5))

                topics.append(
                    TopicResult(
                        topic=topic_name,
                        category=category,
                        confidence=confidence,
                    )
                )
            elif isinstance(item, str):
                topics.append(
                    TopicResult(
                        topic=item,
                        category=TopicCategory.UNKNOWN,
                        confidence=0.5,
                    )
                )

        return topics

    ##Method purpose: Perform heuristic-based query analysis
    def _heuristic_analysis(self, query: str) -> AnalyzedQuery:
        """
        Analyze query using heuristic rules.

        Args:
            query: Query text to analyze

        Returns:
            AnalyzedQuery with heuristic analysis
        """
        query_lower = query.lower()

        ##Step purpose: Detect intent from query patterns
        intent, intent_confidence = self._classify_intent(query, query_lower)

        ##Step purpose: Estimate complexity from query structure
        complexity, complexity_confidence = self._assess_complexity(query)

        ##Step purpose: Classify query type
        query_type, type_confidence = self._classify_type(query_lower)

        ##Step purpose: Extract entities using patterns
        entities = self._extract_entities(query)

        ##Step purpose: Extract keywords
        keywords = self._extract_keywords(query)

        ##Step purpose: Extract topics
        topics = self._extract_topics(query, query_lower)

        ##Step purpose: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            intent_confidence,
            complexity_confidence,
            type_confidence,
        )

        return AnalyzedQuery(
            original_query=query,
            intent=intent,
            intent_confidence=intent_confidence,
            complexity=complexity,
            complexity_confidence=complexity_confidence,
            query_type=query_type,
            type_confidence=type_confidence,
            entities=entities,
            keywords=keywords,
            topics=topics,
            entity_links=[],
            reformulations=[],
            overall_confidence=overall_confidence,
        )

    ##Method purpose: Classify query intent
    def _classify_intent(
        self,
        query: str,
        query_lower: str,
    ) -> tuple[QueryIntent, float]:
        """
        Classify the intent of a query.

        Args:
            query: Original query text
            query_lower: Lowercase query for matching

        Returns:
            Tuple of (intent, confidence)
        """
        ##Condition purpose: Check for command patterns
        command_patterns = ["do", "make", "create", "show", "list", "run", "execute"]
        if query.startswith("/") or any(query_lower.startswith(cmd) for cmd in command_patterns):
            return QueryIntent.COMMAND, 0.8

        ##Condition purpose: Check for question patterns
        question_words = [
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "which",
            "is",
            "are",
            "can",
            "do",
            "does",
        ]
        if query.endswith("?") or any(query_lower.startswith(w) for w in question_words):
            ##Step purpose: Distinguish between question types
            if any(w in query_lower for w in ["clarify", "mean", "understand"]):
                return QueryIntent.CLARIFICATION, 0.75
            return QueryIntent.QUESTION, 0.85

        ##Condition purpose: Check for information seeking patterns
        info_patterns = ["tell me about", "explain", "describe", "i want to know"]
        if any(p in query_lower for p in info_patterns):
            return QueryIntent.INFORMATION_SEEKING, 0.8

        ##Condition purpose: Check for statement patterns
        statement_patterns = ["i think", "i believe", "in my opinion", "i feel"]
        if any(p in query_lower for p in statement_patterns):
            return QueryIntent.STATEMENT, 0.7

        ##Step purpose: Default to conversation
        return QueryIntent.CONVERSATION, 0.5

    ##Method purpose: Assess query complexity
    def _assess_complexity(self, query: str) -> tuple[QueryComplexity, float]:
        """
        Assess the complexity of a query.

        Args:
            query: Query text to assess

        Returns:
            Tuple of (complexity, confidence)
        """
        word_count = len(query.split())
        sentence_count = len(re.split(r"[.!?]+", query))

        ##Step purpose: Count complexity indicators
        complexity_indicators = [
            "and",
            "but",
            "or",
            "because",
            "however",
            "although",
            "therefore",
            "furthermore",
            "additionally",
            "moreover",
        ]
        indicator_count = sum(1 for ind in complexity_indicators if ind in query.lower())

        ##Condition purpose: Simple queries
        if word_count < 6 and sentence_count <= 1 and indicator_count == 0:
            return QueryComplexity.SIMPLE, 0.85

        ##Condition purpose: Moderate queries
        if word_count < 15 and sentence_count <= 2 and indicator_count <= 1:
            return QueryComplexity.MODERATE, 0.75

        ##Condition purpose: Complex queries
        if word_count < 30 and sentence_count <= 3 and indicator_count <= 3:
            return QueryComplexity.COMPLEX, 0.7

        ##Step purpose: Very complex queries
        return QueryComplexity.VERY_COMPLEX, 0.65

    ##Method purpose: Classify query type
    def _classify_type(self, query_lower: str) -> tuple[QueryType, float]:
        """
        Classify the type of query.

        Args:
            query_lower: Lowercase query text

        Returns:
            Tuple of (query_type, confidence)
        """
        ##Condition purpose: Check for procedural queries
        procedural_patterns = ["how to", "steps to", "procedure", "process", "guide", "tutorial"]
        if any(p in query_lower for p in procedural_patterns):
            return QueryType.PROCEDURAL, 0.8

        ##Condition purpose: Check for analytical queries
        analytical_patterns = [
            "analyze",
            "compare",
            "evaluate",
            "difference",
            "pros and cons",
            "versus",
        ]
        if any(p in query_lower for p in analytical_patterns):
            return QueryType.ANALYTICAL, 0.75

        ##Condition purpose: Check for creative queries
        creative_patterns = ["write", "create", "imagine", "story", "poem", "design", "invent"]
        if any(p in query_lower for p in creative_patterns):
            return QueryType.CREATIVE, 0.75

        ##Condition purpose: Check for conversational queries
        conversational_patterns = ["hello", "hi", "hey", "thanks", "bye", "how are you"]
        if any(p in query_lower for p in conversational_patterns):
            return QueryType.CONVERSATIONAL, 0.85

        ##Step purpose: Default to factual
        return QueryType.FACTUAL, 0.6

    ##Method purpose: Extract entities from query
    def _extract_entities(self, query: str) -> list[str]:
        """
        Extract named entities from query.

        Args:
            query: Query text to extract from

        Returns:
            List of extracted entity strings
        """
        entities: list[str] = []

        ##Sec: Limit query length to prevent ReDoS in nested quantifier patterns (P2-002)
        MAX_QUERY_LENGTH_FOR_REGEX: int = 2000
        safe_query = query[:MAX_QUERY_LENGTH_FOR_REGEX]

        ##Step purpose: Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", safe_query)
        ##Loop purpose: Filter out sentence starters
        for cap in capitalized:
            ##Condition purpose: Skip if at start of sentence
            pattern = rf"^{re.escape(cap)}|[.!?]\s+{re.escape(cap)}"
            if not re.search(pattern, safe_query):
                entities.append(cap)

        ##Step purpose: Extract quoted strings as entities
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        ##Loop purpose: Flatten quoted matches
        for match in quoted:
            entity = match[0] or match[1]
            if entity and entity not in entities:
                entities.append(entity)

        ##Step purpose: Extract technical terms (all caps or CamelCase)
        technical = re.findall(r"\b[A-Z]{2,}\b|\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b", safe_query)
        ##Loop purpose: Add technical terms
        for term in technical:
            if term not in entities:
                entities.append(term)

        return entities[:10]  # Limit to 10 entities

    ##Method purpose: Extract keywords from query
    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract key terms from query.

        Args:
            query: Query text to extract from

        Returns:
            List of keyword strings
        """
        ##Step purpose: Remove common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "and",
            "but",
            "if",
            "or",
            "because",
            "about",
        }

        ##Step purpose: Tokenize and filter
        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        ##Step purpose: Remove duplicates while preserving order
        seen: set[str] = set()
        unique_keywords: list[str] = []
        ##Loop purpose: Deduplicate keywords
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:10]  # Limit to 10 keywords

    ##Method purpose: Extract topics from query
    def _extract_topics(self, query: str, query_lower: str) -> list[TopicResult]:
        """
        Extract and categorize topics from query.

        Args:
            query: Original query text
            query_lower: Lowercase query for matching

        Returns:
            List of TopicResult objects
        """
        topics: list[TopicResult] = []

        ##Step purpose: Technical topic detection
        tech_keywords = [
            "code",
            "programming",
            "software",
            "algorithm",
            "api",
            "database",
            "system",
            "python",
            "javascript",
        ]
        if any(kw in query_lower for kw in tech_keywords):
            topics.append(
                TopicResult(
                    topic="technology",
                    category=TopicCategory.TECHNICAL,
                    confidence=0.7,
                    keywords=[kw for kw in tech_keywords if kw in query_lower],
                )
            )

        ##Step purpose: Personal topic detection
        personal_keywords = ["i", "my", "me", "feel", "think", "believe", "want", "need"]
        if any(kw in query_lower.split() for kw in personal_keywords):
            topics.append(
                TopicResult(
                    topic="personal",
                    category=TopicCategory.PERSONAL,
                    confidence=0.6,
                    keywords=[kw for kw in personal_keywords if kw in query_lower.split()],
                )
            )

        ##Step purpose: Procedural topic detection
        procedural_keywords = ["how to", "steps", "guide", "tutorial", "process", "method"]
        if any(kw in query_lower for kw in procedural_keywords):
            topics.append(
                TopicResult(
                    topic="procedures",
                    category=TopicCategory.PROCEDURAL,
                    confidence=0.75,
                    keywords=[kw for kw in procedural_keywords if kw in query_lower],
                )
            )

        ##Step purpose: Temporal topic detection
        temporal_keywords = [
            "when",
            "today",
            "yesterday",
            "tomorrow",
            "last",
            "next",
            "time",
            "schedule",
        ]
        if any(kw in query_lower for kw in temporal_keywords):
            topics.append(
                TopicResult(
                    topic="time-related",
                    category=TopicCategory.TEMPORAL,
                    confidence=0.65,
                    keywords=[kw for kw in temporal_keywords if kw in query_lower],
                )
            )

        ##Step purpose: Creative topic detection
        creative_keywords = ["write", "create", "imagine", "story", "poem", "design", "art"]
        if any(kw in query_lower for kw in creative_keywords):
            topics.append(
                TopicResult(
                    topic="creative",
                    category=TopicCategory.CREATIVE,
                    confidence=0.7,
                    keywords=[kw for kw in creative_keywords if kw in query_lower],
                )
            )

        ##Condition purpose: Add general topic if no specific topics found
        if not topics:
            topics.append(
                TopicResult(
                    topic="general",
                    category=TopicCategory.UNKNOWN,
                    confidence=0.4,
                )
            )

        return topics

    ##Method purpose: Calculate overall confidence from component scores
    def _calculate_overall_confidence(
        self,
        intent_conf: float,
        complexity_conf: float,
        type_conf: float,
    ) -> float:
        """
        Calculate weighted overall confidence score.

        Args:
            intent_conf: Intent classification confidence
            complexity_conf: Complexity assessment confidence
            type_conf: Type classification confidence

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        ##Step purpose: Use weighted average with intent having highest weight
        weights = {"intent": 0.4, "complexity": 0.3, "type": 0.3}
        overall = (
            intent_conf * weights["intent"]
            + complexity_conf * weights["complexity"]
            + type_conf * weights["type"]
        )
        return round(overall, 3)

    ##Method purpose: Link entities to graph nodes
    def _link_entities_to_graph(self, entities: list[str]) -> list[EntityLink]:
        """
        Search graph for nodes related to extracted entities.

        Args:
            entities: List of entity strings to link

        Returns:
            List of EntityLink objects
        """
        ##Condition purpose: Return empty if no graph
        if not self._graph or not entities:
            return []

        entity_links: list[EntityLink] = []

        ##Loop purpose: Link each entity
        for entity in entities:
            if not entity.strip():
                continue

            ##Error purpose: Handle graph search errors gracefully
            try:
                matching_nodes = self._search_graph_for_entity(entity)

                ##Condition purpose: Create link for best match
                if matching_nodes:
                    best_match = matching_nodes[0]
                    entity_links.append(
                        EntityLink(
                            entity=entity,
                            node_id=best_match["id"],
                            node_type=best_match.get("type", "unknown"),
                            confidence=best_match.get("score", 0.0),
                            relationship="related_to",
                        )
                    )
                else:
                    ##Step purpose: Include unlinked entity
                    entity_links.append(
                        EntityLink(
                            entity=entity,
                            node_id=None,
                            node_type=None,
                            confidence=0.0,
                            relationship="unlinked",
                        )
                    )
            except Exception as e:
                logger.warning("entity_link_failed", entity=entity, error=str(e))
                entity_links.append(
                    EntityLink(
                        entity=entity,
                        node_id=None,
                        node_type=None,
                        confidence=0.0,
                        relationship="error",
                    )
                )

        return entity_links

    ##Method purpose: Search graph for entity matches
    def _search_graph_for_entity(self, entity: str) -> list[dict[str, str | float]]:
        """
        Search graph for nodes matching an entity.

        Args:
            entity: Entity string to search for

        Returns:
            List of matching node dicts with scores
        """
        ##Condition purpose: Return empty if no graph
        if not self._graph:
            return []

        ##Update: Check cache before computing entity matches
        entity_lower = entity.lower()
        cache_key = f"entity_match:{entity_lower}"

        cached_matches = self._entity_match_cache.get(cache_key)
        if cached_matches is not None:
            logger.debug("entity_match_cache_hit", entity=entity)
            return cached_matches[: self._config.max_entity_links]

        matching_nodes: list[dict[str, str | float]] = []

        ##Error purpose: Handle graph access errors
        try:
            ##Loop purpose: Score each node
            for node in self._graph.all_nodes():
                score = self._calculate_entity_match_score(entity_lower, node)

                ##Condition purpose: Include if above threshold
                if score >= self._config.min_entity_match_score:
                    matching_nodes.append(
                        {
                            "id": node.id,
                            "type": node.node_type,
                            "label": node.label,
                            "score": score,
                        }
                    )

            ##Fix: Guard non-numeric score in sort key (e.g. cache or legacy data)
            def _score_key(x: dict[str, str | float]) -> float:
                try:
                    return float(x.get("score", 0))
                except (TypeError, ValueError):
                    return 0.0

            ##Step purpose: Sort by score descending
            matching_nodes.sort(key=_score_key, reverse=True)

            result = matching_nodes[: self._config.max_entity_links]

            ##Update: Cache the result
            self._entity_match_cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.warning("graph_search_failed", entity=entity, error=str(e))
            return []

    ##Method purpose: Calculate match score between entity and node
    def _calculate_entity_match_score(
        self,
        entity: str,
        node: object,
    ) -> float:
        """
        Calculate relevance score between entity and graph node.

        Checks content, label, title, and keywords fields for matches.

        Args:
            entity: Lowercase entity string
            node: Graph node object

        Returns:
            Match score (0.0 to 1.0)
        """
        score = 0.0

        ##Fix: Use 'or' for None coalescing - getattr returns None if attr exists but is None
        content = (getattr(node, "content", "") or "").lower()
        label = (getattr(node, "label", "") or "").lower()
        title = (getattr(node, "title", "") or "").lower() if hasattr(node, "title") else ""
        keywords = getattr(node, "keywords", None) or [] if hasattr(node, "keywords") else []

        ##Condition purpose: Check content match
        if entity in content:
            score += 0.4
            ##Condition purpose: Boost for exact word match
            if re.search(rf"\b{re.escape(entity)}\b", content):
                score += 0.2

        ##Condition purpose: Check label match (higher weight)
        if entity in label:
            score += 0.5
            ##Condition purpose: Boost for exact label match
            if entity == label:
                score += 0.3

        ##Condition purpose: Check title match
        if title and entity in title:
            score += 0.4

        ##Condition purpose: Check keywords
        if isinstance(keywords, list):
            for kw in keywords:
                if isinstance(kw, str) and entity in kw.lower():
                    score += 0.2
                    break

        return min(1.0, score)

    ##Method purpose: Generate query reformulations (public method)
    def generate_reformulations(
        self,
        query: str,
        analysis: AnalyzedQuery,
    ) -> list[str]:
        """
        Generate alternative phrasings of the query.

        This is a public method that can be called to generate reformulations
        for a given query and its analysis. Used by analyze() internally
        when reformulation is enabled.

        Args:
            query: Original query text
            analysis: Current analysis results (from analyze())

        Returns:
            List of reformulated query strings (max 3)
        """
        ##Condition purpose: Return existing reformulations if present
        if analysis.reformulations:
            return analysis.reformulations

        ##Condition purpose: Skip if reformulation disabled
        if not self._config.reformulation_enabled:
            return []

        reformulations: list[str] = []

        ##Condition purpose: Add question form if not already a question
        if not query.endswith("?") and analysis.intent == QueryIntent.QUESTION:
            reformulations.append(query.rstrip(".!") + "?")

        ##Condition purpose: Create keyword-focused reformulation
        if analysis.keywords:
            keyword_str = ", ".join(analysis.keywords[:3])
            reformulations.append(f"Tell me about {keyword_str}")

        ##Condition purpose: Create entity-focused reformulation
        if analysis.entities:
            entity_str = " and ".join(analysis.entities[:2])
            if "how" in query.lower():
                reformulations.append(f"Explain how {entity_str} works")
            elif "what" in query.lower():
                reformulations.append(f"What is {entity_str}")
            else:
                reformulations.append(f"Information about {entity_str}")

        ##Condition purpose: Add simplification for complex queries
        if (
            analysis.complexity in (QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX)
            and analysis.keywords
        ):
            reformulations.append(f"Explain {analysis.keywords[0]} simply")

        return reformulations[:3]  # Return max 3 reformulations

    ##Method purpose: Get human-readable summary of analysis (standalone method)
    def get_analysis_summary(self, analysis: AnalyzedQuery) -> str:
        """
        Generate human-readable summary of query analysis.

        This is a standalone method for compatibility with legacy code.
        Same as calling analysis.summary().

        Args:
            analysis: The analyzed query to summarize

        Returns:
            Human-readable summary string
        """
        return analysis.summary()

    ##Method purpose: Generate default analysis when disabled or on error
    def _default_analysis(self, query: str) -> AnalyzedQuery:
        """
        Generate minimal default analysis.

        Args:
            query: Query text

        Returns:
            AnalyzedQuery with basic defaults
        """
        return AnalyzedQuery(
            original_query=query,
            intent=QueryIntent.CONVERSATION,
            intent_confidence=0.5,
            complexity=QueryComplexity.SIMPLE,
            complexity_confidence=0.5,
            query_type=QueryType.FACTUAL,
            type_confidence=0.5,
            entities=[],
            keywords=query.split()[:5],
            topics=[TopicResult(topic="general", category=TopicCategory.UNKNOWN, confidence=0.4)],
            entity_links=[],
            reformulations=[],
            overall_confidence=0.5,
        )
