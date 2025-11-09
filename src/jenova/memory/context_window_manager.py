# The JENOVA Cognitive Architecture - Adaptive Context Window Manager
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 24: Adaptive Context Window Management.

Intelligent context management for improved response quality with:
- Dynamic relevance scoring of memories
- Automatic context compression for low-relevance items
- Priority queuing for high-value context
- Graceful degradation when context exceeds limits
"""

import heapq
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass(order=True)
class ContextItem:
    """
    Represents a single context item with priority.

    Attributes:
        priority: Relevance score (0.0-1.0), higher is better
        content: Text content
        context_type: Type of context (episodic, semantic, procedural, insight)
        metadata: Additional metadata (timestamp, entities, etc.)
        token_count: Number of tokens in content
        access_count: Number of times accessed
        last_access_time: Timestamp of last access
    """

    priority: float = field(compare=True)
    content: str = field(compare=False)
    context_type: str = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    token_count: int = field(default=0, compare=False)
    access_count: int = field(default=0, compare=False)
    last_access_time: float = field(default_factory=time.time, compare=False)

    def __post_init__(self):
        """Negate priority for max-heap behavior using heapq (min-heap)."""
        self.priority = -self.priority  # Negate for max-heap


class ContextWindowManager:
    """
    Manages context window with intelligent prioritization.

    Features:
        - Relevance scoring based on recency, access frequency, semantic similarity
        - Dynamic priority queue for context items
        - Automatic eviction of low-priority items
        - Token counting with configurable limits

    Example:
        >>> manager = ContextWindowManager(max_tokens=4096)
        >>> manager.add_context("User prefers Python", "semantic", {...})
        >>> context = manager.get_optimal_context("How do I write Python?")
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        compression_threshold: float = 0.8,
        min_priority_score: float = 0.3,
        relevance_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize context window manager.

        Args:
            max_tokens: Maximum tokens allowed in context window
            compression_threshold: Start compression at this % full (0.0-1.0)
            min_priority_score: Drop items below this score (0.0-1.0)
            relevance_weights: Weights for relevance calculation
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.min_priority_score = min_priority_score

        # Default relevance weights
        self.relevance_weights = relevance_weights or {
            "semantic_similarity": 0.4,
            "recency": 0.3,
            "frequency": 0.2,
            "user_priority": 0.1,
        }

        # Priority queue (max-heap via negated priorities)
        self.priority_queue: List[ContextItem] = []
        self.current_token_count = 0

        # Access frequency tracking
        self.access_frequency: Dict[str, int] = defaultdict(int)

        # Content hash to item mapping for deduplication
        self.content_hashes: Dict[int, ContextItem] = {}

    def add_context(
        self,
        content: str,
        context_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
    ) -> None:
        """
        Add context item with automatic prioritization.

        Args:
            content: Text content to add
            context_type: Type (episodic, semantic, procedural, insight)
            metadata: Optional metadata (timestamp, entities, confidence, etc.)
            token_count: Pre-calculated token count (auto-calculated if None)

        Example:
            >>> manager.add_context(
            ...     "User loves functional programming",
            ...     "semantic",
            ...     {"source": "conversation", "confidence": 0.9}
            ... )
        """
        if not content or not content.strip():
            return

        # Check for duplicates
        content_hash = hash(content)
        if content_hash in self.content_hashes:
            # Update existing item's access stats
            existing = self.content_hashes[content_hash]
            existing.access_count += 1
            existing.last_access_time = time.time()
            return

        # Count tokens if not provided
        if token_count is None:
            token_count = self._count_tokens(content)

        # Calculate initial priority (will be refined during get_optimal_context)
        metadata = metadata or {}
        priority = self._calculate_base_priority(context_type, metadata)

        # Create context item
        item = ContextItem(
            priority=priority,
            content=content,
            context_type=context_type,
            metadata=metadata,
            token_count=token_count,
            access_count=1,
            last_access_time=time.time(),
        )

        # Add to heap and tracking
        heapq.heappush(self.priority_queue, item)
        self.content_hashes[content_hash] = item
        self.current_token_count += token_count

        # Evict low-priority items if over limit
        self._maybe_evict()

    def get_optimal_context(
        self, query: str, max_tokens: Optional[int] = None
    ) -> str:
        """
        Get optimally prioritized context for query.

        Args:
            query: User query for relevance calculation
            max_tokens: Override max tokens for this query

        Returns:
            Assembled context string optimized for query

        Example:
            >>> context = manager.get_optimal_context("Explain decorators")
            >>> # Returns highest-relevance context about decorators
        """
        if not self.priority_queue:
            return ""

        effective_max_tokens = max_tokens or self.max_tokens

        # Recalculate priorities based on query
        self._recalculate_priorities(query)

        # Sort by priority (highest first due to negation)
        sorted_items = sorted(self.priority_queue)

        # Assemble context staying within token limit
        selected_items: List[ContextItem] = []
        total_tokens = 0

        for item in sorted_items:
            if total_tokens + item.token_count <= effective_max_tokens:
                selected_items.append(item)
                total_tokens += item.token_count
                item.access_count += 1
                item.last_access_time = time.time()

                # Track access frequency
                self.access_frequency[item.content] += 1
            else:
                # Check if compression threshold reached
                if total_tokens / effective_max_tokens >= self.compression_threshold:
                    break

        # Assemble context string
        # Group by type for better organization
        context_by_type: Dict[str, List[str]] = defaultdict(list)
        for item in selected_items:
            context_by_type[item.context_type].append(item.content)

        # Build final context
        context_parts = []
        type_order = ["semantic", "procedural", "episodic", "insight"]

        for ctx_type in type_order:
            if ctx_type in context_by_type:
                context_parts.append(f"[{ctx_type.upper()}]")
                context_parts.extend(context_by_type[ctx_type])

        return "\n\n".join(context_parts)

    def calculate_relevance(
        self, content: str, query: str, metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance score (0.0-1.0) for content given query.

        Factors:
            - Semantic similarity to query (40%)
            - Recency (30%)
            - Access frequency (20%)
            - User-specified priority (10%)

        Args:
            content: Content to score
            query: User query
            metadata: Item metadata

        Returns:
            Relevance score between 0.0 and 1.0

        Example:
            >>> score = manager.calculate_relevance(
            ...     "Python is great for ML",
            ...     "machine learning tools",
            ...     {"timestamp": time.time()}
            ... )
        """
        weights = self.relevance_weights

        # 1. Semantic similarity (simple keyword overlap for now)
        #    In production, this would use embedding similarity
        semantic_score = self._calculate_semantic_similarity(content, query)

        # 2. Recency score
        recency_score = self._calculate_recency_score(metadata.get("timestamp"))

        # 3. Frequency score
        frequency_score = self._calculate_frequency_score(content)

        # 4. User priority
        user_priority = metadata.get("priority", 0.5)

        # Weighted combination
        relevance = (
            semantic_score * weights["semantic_similarity"]
            + recency_score * weights["recency"]
            + frequency_score * weights["frequency"]
            + user_priority * weights["user_priority"]
        )

        return max(0.0, min(1.0, relevance))

    def clear_context(self) -> None:
        """Clear all context items."""
        self.priority_queue.clear()
        self.content_hashes.clear()
        self.access_frequency.clear()
        self.current_token_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get context window statistics.

        Returns:
            Dictionary with stats (item_count, token_count, utilization, etc.)
        """
        return {
            "item_count": len(self.priority_queue),
            "total_tokens": self.current_token_count,
            "max_tokens": self.max_tokens,
            "utilization": self.current_token_count / self.max_tokens
            if self.max_tokens > 0
            else 0.0,
            "unique_items": len(self.content_hashes),
        }

    # Private methods

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text (simple approximation).

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token
        # In production, use tiktoken or model-specific tokenizer
        return len(text) // 4

    def _calculate_base_priority(
        self, context_type: str, metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate base priority before query-specific relevance.

        Args:
            context_type: Type of context
            metadata: Item metadata

        Returns:
            Base priority score
        """
        # Type-based base priorities
        type_priorities = {
            "semantic": 0.7,  # Facts generally important
            "procedural": 0.6,  # How-to knowledge
            "episodic": 0.5,  # Past conversations
            "insight": 0.8,  # Learned insights have high value
        }

        base = type_priorities.get(context_type, 0.5)

        # Boost by confidence if available
        confidence = metadata.get("confidence", 1.0)
        return base * confidence

    def _calculate_semantic_similarity(self, content: str, query: str) -> float:
        """
        Calculate semantic similarity between content and query.

        Args:
            content: Content text
            query: Query text

        Returns:
            Similarity score 0.0-1.0
        """
        # Simple keyword overlap (would use embeddings in production)
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.5  # Neutral if no query

        overlap = len(content_words & query_words)
        return min(1.0, overlap / len(query_words))

    def _calculate_recency_score(self, timestamp: Optional[float]) -> float:
        """
        Calculate recency score (newer = higher score).

        Args:
            timestamp: Unix timestamp or None

        Returns:
            Recency score 0.0-1.0
        """
        if timestamp is None:
            return 0.5  # Neutral if no timestamp

        # Exponential decay: half-life of 7 days
        now = time.time()
        age_seconds = now - timestamp
        age_days = age_seconds / 86400  # Seconds per day

        # Decay function: 0.5^(age_days / 7)
        return 0.5 ** (age_days / 7.0)

    def _calculate_frequency_score(self, content: str) -> float:
        """
        Calculate access frequency score.

        Args:
            content: Content text

        Returns:
            Frequency score 0.0-1.0
        """
        access_count = self.access_frequency.get(content, 0)

        # Logarithmic scaling to prevent dominance
        if access_count == 0:
            return 0.0

        # log10(count + 1) / log10(101) gives 0-1 range for counts 0-100
        return min(1.0, (1.0 + access_count) ** 0.5 / 10.0)

    def _recalculate_priorities(self, query: str) -> None:
        """
        Recalculate all priorities based on query.

        Args:
            query: User query
        """
        for item in self.priority_queue:
            # Calculate query-specific relevance
            relevance = self.calculate_relevance(item.content, query, item.metadata)

            # Update priority (remember to negate for max-heap)
            item.priority = -relevance

        # Rebuild heap with new priorities
        heapq.heapify(self.priority_queue)

    def _maybe_evict(self) -> None:
        """Evict low-priority items if over token limit."""
        while self.current_token_count > self.max_tokens and self.priority_queue:
            # Remove lowest priority item (highest value due to negation)
            item = heapq.heappop(self.priority_queue)

            # Update tracking
            self.current_token_count -= item.token_count
            content_hash = hash(item.content)
            if content_hash in self.content_hashes:
                del self.content_hashes[content_hash]

        # Also remove items below minimum priority threshold
        self.priority_queue = [
            item
            for item in self.priority_queue
            if -item.priority >= self.min_priority_score
        ]
        heapq.heapify(self.priority_queue)
