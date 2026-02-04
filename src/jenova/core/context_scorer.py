##Script function and purpose: Context relevance scoring with multi-factor analysis
"""
Context Scorer

Scores context items based on multiple relevance factors for prioritization.
Enables intelligent ranking of retrieved context for LLM prompt construction.
"""

from __future__ import annotations

import hashlib
import heapq
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import structlog

from jenova.core.query_analyzer import AnalyzedQuery, QueryType
from jenova.utils.cache import TTLCache

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Store individual scoring factor weights
@dataclass
class ScoringWeights:
    """Weights for multi-factor context scoring."""

    semantic_similarity: float = 0.4
    """Weight for embedding-based semantic similarity."""

    entity_overlap: float = 0.3
    """Weight for entity overlap between query and context."""

    keyword_match: float = 0.2
    """Weight for keyword matching score."""

    query_type_match: float = 0.1
    """Weight for query type alignment."""

    ##Method purpose: Validate weights sum to 1.0
    def normalize(self) -> ScoringWeights:
        """Normalize weights to sum to 1.0."""
        total = (
            self.semantic_similarity
            + self.entity_overlap
            + self.keyword_match
            + self.query_type_match
        )
        ##Condition purpose: Avoid division by zero
        if total <= 0:
            return ScoringWeights()

        return ScoringWeights(
            semantic_similarity=self.semantic_similarity / total,
            entity_overlap=self.entity_overlap / total,
            keyword_match=self.keyword_match / total,
            query_type_match=self.query_type_match / total,
        )


##Class purpose: Store detailed scoring breakdown for a context item
@dataclass
class ScoringBreakdown:
    """Detailed scoring breakdown for a context item."""

    content: str
    """The context content that was scored."""

    total_score: float
    """Combined weighted score (0.0 to 1.0)."""

    semantic_score: float = 0.0
    """Semantic similarity score."""

    entity_score: float = 0.0
    """Entity overlap score."""

    keyword_score: float = 0.0
    """Keyword match score."""

    type_score: float = 0.0
    """Query type match score."""

    ##Fix: heapq compares breakdowns when scores tie; use total_score then content for ordering
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ScoringBreakdown):
            return NotImplemented
        if self.total_score != other.total_score:
            return self.total_score < other.total_score
        return self.content < other.content


##Class purpose: Result of context scoring operation
@dataclass
class ScoredContext:
    """Result of context scoring with rankings."""

    items: list[ScoringBreakdown]
    """Scored items sorted by total_score descending."""

    query: str
    """Original query used for scoring."""

    ##Method purpose: Get top N items
    def top(self, n: int) -> list[ScoringBreakdown]:
        """Get top N scored items."""
        ##Fix: Clamp negative n to avoid surprising slice semantics (BUG-003)
        n = max(0, n)
        return self.items[:n]

    ##Method purpose: Get items above threshold
    def above_threshold(self, threshold: float) -> list[ScoringBreakdown]:
        """Get items with score above threshold."""
        return [item for item in self.items if item.total_score >= threshold]

    ##Method purpose: Get content strings only
    def as_strings(self, n: int | None = None) -> list[str]:
        """Get content strings, optionally limited to top N."""
        ##Fix: Clamp negative n to avoid surprising slice semantics (BUG-003)
        items = self.items[: max(0, n)] if n is not None else self.items
        return [item.content for item in items]


##Class purpose: Configuration for context scorer
@dataclass
class ContextScorerConfig:
    """Configuration for ContextScorer behavior."""

    enabled: bool = True
    """Whether context scoring is enabled."""

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    """Scoring factor weights."""

    normalize_weights: bool = True
    """Whether to normalize weights to sum to 1.0."""


##Class purpose: Protocol for embedding model interface
@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Protocol for embedding model used in semantic similarity."""

    ##Method purpose: Encode texts to embeddings
    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
    ) -> list[list[float]]:
        """Encode texts to embedding vectors."""
        ...


##Class purpose: Scores context items for relevance and prioritization
class ContextScorer:
    """
    Scores context items based on multiple relevance factors.

    Uses semantic similarity, entity overlap, keyword matching, and
    query type alignment to compute relevance scores for context items.
    """

    ##Method purpose: Initialize scorer with configuration
    def __init__(
        self,
        config: ContextScorerConfig | None = None,
        embedding_model: EmbeddingModelProtocol | None = None,
    ) -> None:
        """
        Initialize context scorer.

        Args:
            config: Scorer configuration
            embedding_model: Optional embedding model for semantic similarity
        """
        ##Step purpose: Store configuration
        self._config = config or ContextScorerConfig()
        self._embedding_model = embedding_model

        ##Step purpose: Normalize weights if configured
        if self._config.normalize_weights:
            self._weights = self._config.weights.normalize()
        else:
            self._weights = self._config.weights

        ##Update: Add embedding cache for performance (P1-004)
        self._embedding_cache: TTLCache[str, list[float]] = TTLCache(
            max_size=1000,
            default_ttl=600,  # 10 minutes
        )

        logger.debug(
            "context_scorer_initialized",
            enabled=self._config.enabled,
            has_embedding_model=embedding_model is not None,
        )

    ##Method purpose: Score context items based on query and analysis
    def score(
        self,
        context_items: list[str],
        query: str,
        analysis: AnalyzedQuery,
    ) -> ScoredContext:
        """
        Score and rank context items by relevance.

        Args:
            context_items: List of context strings to score
            query: The user query
            analysis: Analyzed query with extracted information

        Returns:
            ScoredContext with ranked items
        """
        ##Condition purpose: Return unscored items if disabled
        if not self._config.enabled:
            logger.debug("context_scoring_disabled")
            items = [ScoringBreakdown(content=item, total_score=1.0) for item in context_items]
            return ScoredContext(items=items, query=query)

        ##Update: Optimize scoring with batching, caching, and early termination (P1-004)
        ##Step purpose: Use heap for top-k selection (more efficient than sorting all)
        top_k_heap: list[tuple[float, ScoringBreakdown]] = []
        k = min(len(context_items), 50)  # Track top 50 for early termination

        ##Update: Batch embedding operations (P1-004)
        item_embeddings: dict[str, list[float]] = {}
        query_embedding: list[float] | None = None

        if self._embedding_model is not None:
            ##Step purpose: Get or compute query embedding
            query_cache_key = hashlib.sha256(query.encode()).hexdigest()
            query_embedding = self._embedding_cache.get(query_cache_key)
            if query_embedding is None:
                try:
                    query_emb_raw = self._embedding_model.encode([query], convert_to_numpy=True)
                    query_embedding = (
                        query_emb_raw[0].tolist()
                        if hasattr(query_emb_raw[0], "tolist")
                        else list(query_emb_raw[0])
                    )
                    self._embedding_cache.set(query_cache_key, query_embedding)
                except Exception as e:
                    logger.warning("query_embedding_failed", error=str(e))
                    query_embedding = None

            ##Step purpose: Batch compute item embeddings
            items_to_embed: list[str] = []
            item_indices: list[int] = []
            for i, item in enumerate(context_items):
                item_cache_key = hashlib.sha256(item.encode()).hexdigest()
                cached_emb = self._embedding_cache.get(item_cache_key)
                if cached_emb is not None:
                    item_embeddings[item] = cached_emb
                else:
                    items_to_embed.append(item)
                    item_indices.append(i)

            ##Action purpose: Batch encode items that aren't cached
            if items_to_embed:
                try:
                    batch_embeddings = self._embedding_model.encode(
                        items_to_embed, convert_to_numpy=True
                    )
                    for item, emb in zip(items_to_embed, batch_embeddings, strict=False):
                        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                        item_cache_key = hashlib.sha256(item.encode()).hexdigest()
                        item_embeddings[item] = emb_list
                        self._embedding_cache.set(item_cache_key, emb_list)
                except Exception as e:
                    logger.warning("batch_embedding_failed", error=str(e))

        ##Step purpose: Score items with early termination
        early_termination_threshold = 0.8  # Stop if top-k scores are above this

        ##Loop purpose: Calculate scores for each item
        for item in context_items:
            breakdown = self._score_item_optimized(
                item, query, analysis, item_embeddings.get(item), query_embedding
            )

            ##Update: Use heap for top-k (P1-004)
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (breakdown.total_score, breakdown))
            elif breakdown.total_score > top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (breakdown.total_score, breakdown))

            ##Update: Early termination if top scores are high (P1-004)
            if len(top_k_heap) >= k and top_k_heap[0][0] >= early_termination_threshold:
                logger.debug("early_termination_triggered", threshold=early_termination_threshold)
                break

        ##Step purpose: Extract and sort top-k results
        scored_items = [breakdown for _, breakdown in top_k_heap]
        scored_items.sort(key=lambda x: x.total_score, reverse=True)

        logger.info(
            "context_scored",
            item_count=len(scored_items),
            total_items=len(context_items),
            top_score=scored_items[0].total_score if scored_items else 0.0,
        )

        return ScoredContext(items=scored_items, query=query)

    ##Method purpose: Score a single context item
    def _score_item(
        self,
        item: str,
        query: str,
        analysis: AnalyzedQuery,
    ) -> ScoringBreakdown:
        """
        Calculate multi-factor score for a context item.

        Args:
            item: Context content to score
            query: User query
            analysis: Analyzed query

        Returns:
            ScoringBreakdown with detailed scores
        """
        ##Step purpose: Calculate individual factor scores
        semantic_score = self._semantic_similarity(item, query)
        entity_score = self._entity_overlap(item, analysis.entities)
        keyword_score = self._keyword_match(item, analysis.keywords)
        type_score = self._query_type_match(item, analysis.query_type)

        ##Step purpose: Calculate weighted total
        total_score = (
            semantic_score * self._weights.semantic_similarity
            + entity_score * self._weights.entity_overlap
            + keyword_score * self._weights.keyword_match
            + type_score * self._weights.query_type_match
        )

        ##Step purpose: Clamp to valid range
        total_score = max(0.0, min(1.0, total_score))

        return ScoringBreakdown(
            content=item,
            total_score=total_score,
            semantic_score=semantic_score,
            entity_score=entity_score,
            keyword_score=keyword_score,
            type_score=type_score,
        )

    ##Update: Optimized scoring method using pre-computed embeddings (P1-004)
    ##Method purpose: Score item with pre-computed embeddings
    def _score_item_optimized(
        self,
        item: str,
        query: str,
        analysis: AnalyzedQuery,
        item_embedding: list[float] | None,
        query_embedding: list[float] | None,
    ) -> ScoringBreakdown:
        """
        Calculate multi-factor score using pre-computed embeddings.

        Args:
            item: Context content to score
            query: User query
            analysis: Analyzed query
            item_embedding: Pre-computed item embedding (None if unavailable)
            query_embedding: Pre-computed query embedding (None if unavailable)

        Returns:
            ScoringBreakdown with detailed scores
        """
        ##Step purpose: Calculate semantic score from pre-computed embeddings
        if item_embedding is not None and query_embedding is not None:
            ##Step purpose: Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(item_embedding, query_embedding, strict=False))
            norm_item = sum(a * a for a in item_embedding) ** 0.5
            norm_query = sum(b * b for b in query_embedding) ** 0.5

            ##Condition purpose: Avoid division by zero
            if norm_item > 0 and norm_query > 0:
                similarity = dot_product / (norm_item * norm_query)
                semantic_score = (similarity + 1) / 2  # Normalize to [0, 1]
            else:
                semantic_score = 0.5
        else:
            ##Step purpose: Fallback to word overlap if embeddings unavailable
            semantic_score = self._word_overlap_similarity(item, query)

        ##Step purpose: Calculate other factor scores
        entity_score = self._entity_overlap(item, analysis.entities)
        keyword_score = self._keyword_match(item, analysis.keywords)
        type_score = self._query_type_match(item, analysis.query_type)

        ##Step purpose: Calculate weighted total
        total_score = (
            semantic_score * self._weights.semantic_similarity
            + entity_score * self._weights.entity_overlap
            + keyword_score * self._weights.keyword_match
            + type_score * self._weights.query_type_match
        )

        ##Step purpose: Clamp to valid range
        total_score = max(0.0, min(1.0, total_score))

        return ScoringBreakdown(
            content=item,
            total_score=total_score,
            semantic_score=semantic_score,
            entity_score=entity_score,
            keyword_score=keyword_score,
            type_score=type_score,
        )

    ##Method purpose: Calculate semantic similarity score
    def _semantic_similarity(self, item: str, query: str) -> float:
        """
        Calculate semantic similarity using embeddings or word overlap.

        Args:
            item: Context content
            query: User query

        Returns:
            Similarity score (0.0 to 1.0)
        """
        ##Condition purpose: Use embedding model if available
        if self._embedding_model is not None:
            ##Error purpose: Handle embedding errors gracefully
            try:
                return self._embedding_similarity(item, query)
            except Exception as e:
                logger.warning("embedding_similarity_failed", error=str(e))

        ##Step purpose: Fallback to word overlap
        return self._word_overlap_similarity(item, query)

    ##Method purpose: Calculate embedding-based similarity
    def _embedding_similarity(self, item: str, query: str) -> float:
        """
        Calculate cosine similarity using embeddings.

        Args:
            item: Context content
            query: User query

        Returns:
            Cosine similarity normalized to (0.0 to 1.0)
        """
        ##Condition purpose: Check for embedding model
        if self._embedding_model is None:
            return 0.5

        ##Step purpose: Get embeddings
        embeddings = self._embedding_model.encode([item, query], convert_to_numpy=True)
        item_embedding = embeddings[0]
        query_embedding = embeddings[1]

        ##Step purpose: Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(item_embedding, query_embedding, strict=False))
        norm_item = sum(a * a for a in item_embedding) ** 0.5
        norm_query = sum(b * b for b in query_embedding) ** 0.5

        ##Condition purpose: Avoid division by zero
        if norm_item <= 0 or norm_query <= 0:
            return 0.5

        similarity = dot_product / (norm_item * norm_query)

        ##Step purpose: Normalize from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    ##Method purpose: Calculate word overlap similarity
    def _word_overlap_similarity(self, item: str, query: str) -> float:
        """
        Calculate Jaccard similarity based on word overlap.

        Args:
            item: Context content
            query: User query

        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        ##Step purpose: Tokenize and lowercase
        item_words = set(re.findall(r"\b\w+\b", item.lower()))
        query_words = set(re.findall(r"\b\w+\b", query.lower()))

        ##Condition purpose: Handle empty query
        if not query_words:
            return 0.0

        ##Step purpose: Calculate Jaccard similarity
        intersection = len(item_words & query_words)
        union = len(item_words | query_words)

        ##Condition purpose: Avoid division by zero
        if union == 0:
            return 0.0

        return intersection / union

    ##Method purpose: Calculate entity overlap score
    def _entity_overlap(self, item: str, entities: list[str]) -> float:
        """
        Calculate overlap between context and query entities.

        Args:
            item: Context content
            entities: Extracted entities from query

        Returns:
            Overlap score (0.0 to 1.0)
        """
        ##Condition purpose: Handle empty entities
        if not entities:
            return 0.0

        item_lower = item.lower()
        matches = 0

        ##Loop purpose: Count entity matches
        for entity in entities:
            entity_lower = entity.lower()
            ##Condition purpose: Check for whole word match
            if re.search(rf"\b{re.escape(entity_lower)}\b", item_lower):
                matches += 1

        return matches / len(entities)

    ##Method purpose: Calculate keyword match score
    def _keyword_match(self, item: str, keywords: list[str]) -> float:
        """
        Calculate keyword match score.

        Args:
            item: Context content
            keywords: Extracted keywords from query

        Returns:
            Match score (0.0 to 1.0)
        """
        ##Condition purpose: Handle empty keywords
        if not keywords:
            return 0.0

        item_lower = item.lower()
        matches = 0

        ##Loop purpose: Count keyword matches
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in item_lower:
                matches += 1

        return matches / len(keywords)

    ##Method purpose: Calculate query type match score
    def _query_type_match(self, item: str, query_type: QueryType) -> float:
        """
        Calculate how well context matches query type.

        Args:
            item: Context content
            query_type: Classified query type

        Returns:
            Type alignment score (0.0 to 1.0)
        """
        item_lower = item.lower()

        ##Condition purpose: Score based on query type
        if query_type == QueryType.FACTUAL:
            ##Step purpose: Factual queries benefit from declarative content
            factual_indicators = [
                "is",
                "was",
                "are",
                "were",
                "fact",
                "information",
                "data",
                "known",
            ]
            if any(ind in item_lower for ind in factual_indicators):
                return 0.8
            return 0.5

        elif query_type == QueryType.PROCEDURAL:
            ##Step purpose: Procedural queries benefit from step-by-step content
            procedural_indicators = [
                "step",
                "procedure",
                "process",
                "how to",
                "method",
                "way",
                "first",
                "then",
                "finally",
            ]
            if any(ind in item_lower for ind in procedural_indicators):
                return 0.9
            return 0.4

        elif query_type == QueryType.ANALYTICAL:
            ##Step purpose: Analytical queries benefit from comparative content
            analytical_indicators = [
                "compare",
                "analyze",
                "evaluate",
                "difference",
                "similarity",
                "why",
                "because",
                "reason",
            ]
            if any(ind in item_lower for ind in analytical_indicators):
                return 0.85
            return 0.45

        elif query_type == QueryType.CREATIVE:
            ##Step purpose: Creative queries benefit from diverse content
            creative_indicators = [
                "imagine",
                "create",
                "story",
                "idea",
                "design",
                "unique",
                "original",
            ]
            if any(ind in item_lower for ind in creative_indicators):
                return 0.8
            return 0.6  # Creative benefits from variety

        elif query_type == QueryType.CONVERSATIONAL:
            ##Step purpose: Conversational benefits from personal content
            conversational_indicators = ["feel", "think", "believe", "personal", "experience"]
            if any(ind in item_lower for ind in conversational_indicators):
                return 0.75
            return 0.5

        ##Step purpose: Default neutral score
        return 0.5

    ##Method purpose: Score simple list of strings without full analysis
    def score_simple(
        self,
        context_items: list[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """
        Simple scoring without AnalyzedQuery (for backward compatibility).

        Args:
            context_items: List of context strings
            query: User query

        Returns:
            List of (content, score) tuples sorted by score
        """
        ##Condition purpose: Return unscored if disabled
        if not self._config.enabled:
            return [(item, 1.0) for item in context_items]

        scored: list[tuple[str, float]] = []

        ##Loop purpose: Score each item
        for item in context_items:
            ##Step purpose: Calculate simple semantic score
            score = self._semantic_similarity(item, query)
            scored.append((item, score))

        ##Step purpose: Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored
