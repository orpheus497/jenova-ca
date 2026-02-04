##Script function and purpose: Context organization into hierarchical tiers for
##                          LLM prompt construction
"""
Context Organizer

Organizes context into categories and hierarchical tiers for improved
prompt construction. Prioritizes context based on relevance scoring
and manages token budgets for LLM prompts.

Supports two categorization modes:
- Heuristic mode (default): Fast, keyword-based categorization
- LLM mode: More accurate categorization using language model (for sets > 3 items)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import structlog

from jenova.core.context_scorer import ScoredContext, ScoringBreakdown
from jenova.utils.json_safe import JSONSizeError, extract_json_from_response, safe_json_loads

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Protocol for LLM interface
class LLMProtocol(Protocol):
    """Protocol defining LLM interface for categorization."""

    ##Method purpose: Generate text from prompt
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        ...


##Class purpose: Enum defining context priority tiers
class ContextTier(Enum):
    """Priority tier for context items."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


##Class purpose: Store organized context with categories and tiers
@dataclass
class OrganizedContext:
    """Context organized into categories and priority tiers."""

    categorized: dict[str, list[str]]
    """Items grouped by topic category."""

    tiers: dict[ContextTier, list[str]]
    """Items grouped by priority tier."""

    query: str
    """Original query used for organization."""

    ##Method purpose: Get all items in a specific tier
    def get_tier(self, tier: ContextTier) -> list[str]:
        """Get all items in a priority tier."""
        return self.tiers.get(tier, [])

    ##Method purpose: Get high priority items only
    @property
    def high_priority(self) -> list[str]:
        """Get high priority items."""
        return self.tiers.get(ContextTier.HIGH, [])

    ##Method purpose: Get medium priority items only
    @property
    def medium_priority(self) -> list[str]:
        """Get medium priority items."""
        return self.tiers.get(ContextTier.MEDIUM, [])

    ##Method purpose: Get low priority items only
    @property
    def low_priority(self) -> list[str]:
        """Get low priority items."""
        return self.tiers.get(ContextTier.LOW, [])

    ##Method purpose: Get total item count
    @property
    def total_items(self) -> int:
        """Get total number of items across all tiers."""
        return sum(len(items) for items in self.tiers.values())

    ##Method purpose: Check if context is empty
    def is_empty(self) -> bool:
        """Check if no context items exist."""
        return self.total_items == 0

    ##Method purpose: Format for LLM prompt
    def format_for_prompt(
        self,
        max_tokens: int = 2000,
        chars_per_token: int = 4,
    ) -> str:
        """
        Format organized context for LLM prompt.

        Args:
            max_tokens: Maximum tokens for context
            chars_per_token: Estimated characters per token

        Returns:
            Formatted context string
        """
        max_chars = max_tokens * chars_per_token
        parts: list[str] = []
        current_chars = 0

        ##Step purpose: Add high priority first
        if self.high_priority:
            header = "### Highly Relevant Context:\n"
            current_chars += len(header)
            items_text = self._format_items(
                self.high_priority,
                max_chars - current_chars,
            )
            if items_text:
                parts.append(header + items_text)
                current_chars += len(items_text)

        ##Step purpose: Add medium priority if space allows
        if self.medium_priority and current_chars < max_chars * 0.8:
            header = "\n### Related Context:\n"
            current_chars += len(header)
            items_text = self._format_items(
                self.medium_priority,
                max_chars - current_chars,
            )
            if items_text:
                parts.append(header + items_text)
                current_chars += len(items_text)

        ##Step purpose: Add low priority if space allows
        if self.low_priority and current_chars < max_chars * 0.6:
            header = "\n### Additional Context:\n"
            current_chars += len(header)
            items_text = self._format_items(
                self.low_priority,
                max_chars - current_chars,
            )
            if items_text:
                parts.append(header + items_text)

        return "".join(parts)

    ##Method purpose: Format list of items with character limit
    def _format_items(self, items: list[str], max_chars: int) -> str:
        """Format items as bullet list within character limit."""
        formatted: list[str] = []
        chars_used = 0

        ##Loop purpose: Add items until limit reached
        for item in items:
            line = f"- {item}\n"
            if chars_used + len(line) > max_chars:
                ##Step purpose: Truncate if needed
                remaining = max_chars - chars_used - 10
                if remaining > 50:
                    formatted.append(f"- {item[:remaining]}...\n")
                break
            formatted.append(line)
            chars_used += len(line)

        return "".join(formatted)


##Class purpose: Configuration for context organizer
@dataclass
class ContextOrganizerConfig:
    """Configuration for ContextOrganizer behavior."""

    enabled: bool = True
    """Whether organization is enabled."""

    categorization_enabled: bool = True
    """Whether to categorize by topic."""

    use_llm_categorization: bool = False
    """Whether to use LLM for categorization (for sets > 3 items)."""

    llm_categorization_threshold: int = 3
    """Minimum items to trigger LLM categorization."""

    tier_classification_enabled: bool = True
    """Whether to classify by priority tier."""

    high_tier_threshold: float = 0.7
    """Score threshold for high priority."""

    medium_tier_threshold: float = 0.4
    """Score threshold for medium priority."""

    max_items_per_tier: int = 10
    """Maximum items per priority tier."""

    max_categories: int = 5
    """Maximum number of categories."""


##Class purpose: Organizes context into structured categories and tiers
class ContextOrganizer:
    """
    Organizes context into categories and priority tiers.

    Provides hierarchical organization of context items for optimal
    prompt construction and token budget management.

    Supports two categorization modes:
    - Heuristic mode (default): Fast, keyword-based categorization
    - LLM mode: More accurate categorization for larger item sets
    """

    ##Method purpose: Initialize organizer with configuration
    def __init__(
        self,
        config: ContextOrganizerConfig | None = None,
        llm: LLMProtocol | None = None,
    ) -> None:
        """
        Initialize context organizer.

        Args:
            config: Organizer configuration
            llm: Optional LLM for advanced categorization
        """
        self._config = config or ContextOrganizerConfig()
        self._llm = llm

        logger.debug(
            "context_organizer_initialized",
            enabled=self._config.enabled,
            categorization=self._config.categorization_enabled,
            tier_classification=self._config.tier_classification_enabled,
            use_llm=self._config.use_llm_categorization and llm is not None,
        )

    ##Method purpose: Organize scored context into categories and tiers
    def organize(
        self,
        scored_context: ScoredContext,
    ) -> OrganizedContext:
        """
        Organize scored context into categories and tiers.

        Args:
            scored_context: Context with relevance scores

        Returns:
            OrganizedContext with categorization and tiering
        """
        ##Condition purpose: Return flat organization if disabled
        if not self._config.enabled:
            return self._flat_organization(scored_context)

        ##Step purpose: Categorize items by topic
        if self._config.categorization_enabled:
            categorized = self._categorize_by_topic(
                scored_context.items,
                scored_context.query,
            )
        else:
            categorized = {"general": [item.content for item in scored_context.items]}

        ##Step purpose: Classify by relevance tier
        if self._config.tier_classification_enabled:
            tiers = self._classify_by_tier(scored_context.items)
        else:
            tiers = {
                ContextTier.MEDIUM: [item.content for item in scored_context.items],
                ContextTier.HIGH: [],
                ContextTier.LOW: [],
            }

        logger.info(
            "context_organized",
            categories=len(categorized),
            high_count=len(tiers.get(ContextTier.HIGH, [])),
            medium_count=len(tiers.get(ContextTier.MEDIUM, [])),
            low_count=len(tiers.get(ContextTier.LOW, [])),
        )

        return OrganizedContext(
            categorized=categorized,
            tiers=tiers,
            query=scored_context.query,
        )

    ##Method purpose: Organize raw context items without scores
    def organize_raw(
        self,
        context_items: list[str],
        query: str,
    ) -> OrganizedContext:
        """
        Organize raw context items without pre-scoring.

        Args:
            context_items: List of context strings
            query: User query for categorization

        Returns:
            OrganizedContext with categorization
        """
        ##Condition purpose: Return flat if disabled
        if not self._config.enabled:
            return OrganizedContext(
                categorized={},
                tiers={ContextTier.MEDIUM: context_items},
                query=query,
            )

        ##Step purpose: Categorize items
        if self._config.categorization_enabled:
            categorized = self._categorize_by_keywords(context_items, query)
        else:
            categorized = {"general": context_items}

        ##Step purpose: Simple tier classification by category size
        tiers = self._classify_by_category_relevance(categorized, query)

        return OrganizedContext(
            categorized=categorized,
            tiers=tiers,
            query=query,
        )

    ##Method purpose: Categorize items by detected topic
    def _categorize_by_topic(
        self,
        items: list[ScoringBreakdown],
        query: str,
    ) -> dict[str, list[str]]:
        """
        Group items by detected topic.

        Uses LLM categorization for larger item sets if configured,
        otherwise falls back to keyword-based categorization.

        Args:
            items: Scored context items
            query: User query for context

        Returns:
            Dict mapping category names to item lists
        """
        ##Condition purpose: Use LLM for larger sets if configured
        if (
            self._config.use_llm_categorization
            and self._llm is not None
            and len(items) > self._config.llm_categorization_threshold
        ):
            return self._llm_categorize(items, query)

        return self._heuristic_categorize(items, query)

    ##Method purpose: LLM-based categorization for more accurate grouping
    def _llm_categorize(
        self,
        items: list[ScoringBreakdown],
        query: str,
    ) -> dict[str, list[str]]:
        """
        Categorize items using LLM for better accuracy.

        Args:
            items: Scored context items
            query: User query for context

        Returns:
            Dict mapping category names to item lists
        """
        ##Condition purpose: Fallback if no LLM
        if self._llm is None:
            return self._heuristic_categorize(items, query)

        ##Step purpose: Build categorization prompt
        items_str = "\n".join(f"{i + 1}. {item.content[:200]}" for i, item in enumerate(items))

        prompt = f"""Categorize the following context items by topic. Group related items together.
Each item should be assigned to exactly one category. Use concise category names (1-3 words).

Query: "{query}"

Context Items:
{items_str}

Respond with a valid JSON object where keys are category names and values are
arrays of item numbers (1-indexed).
Example: {{"Python Programming": [1, 3], "Data Structures": [2, 4]}}

JSON Response:"""

        ##Error purpose: Handle LLM errors gracefully
        try:
            response = self._llm.generate(prompt, temperature=0.3)
            return self._parse_llm_categorization(response, items)
        except Exception as e:
            logger.warning("llm_categorization_failed", error=str(e))
            return self._heuristic_categorize(items, query)

    ##Method purpose: Parse LLM categorization response
    def _parse_llm_categorization(
        self,
        response: str,
        items: list[ScoringBreakdown],
    ) -> dict[str, list[str]]:
        """
        Parse LLM JSON response for categorization.

        Args:
            response: LLM response string
            items: Original items for index lookup

        Returns:
            Dict mapping category names to item content lists
        """
        ##Error purpose: Handle JSON parse errors
        try:
            ##Step purpose: Extract JSON from response
            try:
                json_str = extract_json_from_response(response)
            except ValueError:
                logger.warning("no_json_in_llm_response")
                return self._heuristic_categorize(items, "")

            ##Action purpose: Parse with size limits
            data = safe_json_loads(json_str)
        except (json.JSONDecodeError, JSONSizeError) as e:
            logger.warning("json_parse_failed", error=str(e))
            return self._heuristic_categorize(items, "")

        ##Condition purpose: Validate data type
        if not isinstance(data, dict):
            return self._heuristic_categorize(items, "")

        ##Step purpose: Convert indices to content
        categorized: dict[str, list[str]] = {}
        categorized_indices: set[int] = set()

        ##Loop purpose: Process each category
        for category, indices in data.items():
            if not isinstance(category, str) or not isinstance(indices, list):
                continue

            categorized[category] = []
            for idx in indices:
                if isinstance(idx, int) and 1 <= idx <= len(items):
                    categorized[category].append(items[idx - 1].content)
                    categorized_indices.add(idx - 1)

        ##Step purpose: Add uncategorized items
        uncategorized = [
            items[i].content for i in range(len(items)) if i not in categorized_indices
        ]
        if uncategorized:
            categorized["Other"] = uncategorized

        logger.info(
            "llm_categorization_complete",
            categories=len(categorized),
            items=len(items),
        )

        return categorized

    ##Method purpose: Heuristic keyword-based categorization
    def _heuristic_categorize(
        self,
        items: list[ScoringBreakdown],
        query: str,
    ) -> dict[str, list[str]]:
        """
        Categorize items using keyword matching (fallback method).

        Args:
            items: Scored context items
            query: User query for context

        Returns:
            Dict mapping category names to item lists
        """
        ##Step purpose: Define topic keyword mappings
        topic_keywords: dict[str, list[str]] = {
            "Technical": [
                "code",
                "programming",
                "software",
                "api",
                "function",
                "class",
                "method",
                "bug",
                "error",
            ],
            "Procedural": ["step", "how to", "guide", "process", "procedure", "method", "way"],
            "Conceptual": ["concept", "idea", "theory", "principle", "definition", "meaning"],
            "Personal": ["i", "my", "me", "feel", "think", "believe", "experience"],
            "Temporal": ["time", "when", "date", "schedule", "yesterday", "today", "tomorrow"],
        }

        categorized: dict[str, list[str]] = defaultdict(list)
        uncategorized: list[str] = []

        ##Loop purpose: Categorize each item
        for item in items:
            content_lower = item.content.lower()
            matched_category: str | None = None
            best_match_count = 0

            ##Loop purpose: Find best matching category
            for category, keywords in topic_keywords.items():
                match_count = sum(1 for kw in keywords if kw in content_lower)
                if match_count > best_match_count:
                    best_match_count = match_count
                    matched_category = category

            ##Condition purpose: Add to category or uncategorized
            if matched_category and best_match_count > 0:
                categorized[matched_category].append(item.content)
            else:
                uncategorized.append(item.content)

        ##Step purpose: Add uncategorized items
        if uncategorized:
            categorized["Other"] = uncategorized

        ##Step purpose: Limit number of categories
        if len(categorized) > self._config.max_categories:
            ##Step purpose: Keep largest categories
            sorted_categories = sorted(
                categorized.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
            categorized = dict(sorted_categories[: self._config.max_categories])

        return dict(categorized)

    ##Method purpose: Classify items by priority tier based on scores
    def _classify_by_tier(
        self,
        items: list[ScoringBreakdown],
    ) -> dict[ContextTier, list[str]]:
        """
        Classify items into priority tiers based on scores.

        Args:
            items: Scored context items

        Returns:
            Dict mapping tiers to item lists
        """
        tiers: dict[ContextTier, list[str]] = {
            ContextTier.HIGH: [],
            ContextTier.MEDIUM: [],
            ContextTier.LOW: [],
        }

        ##Loop purpose: Assign each item to a tier
        for item in items:
            ##Condition purpose: Classify by score thresholds
            if item.total_score >= self._config.high_tier_threshold:
                tier = ContextTier.HIGH
            elif item.total_score >= self._config.medium_tier_threshold:
                tier = ContextTier.MEDIUM
            else:
                tier = ContextTier.LOW

            ##Condition purpose: Respect max items per tier
            if len(tiers[tier]) < self._config.max_items_per_tier:
                tiers[tier].append(item.content)

        return tiers

    ##Method purpose: Categorize by keyword matching
    def _categorize_by_keywords(
        self,
        items: list[str],
        query: str,
    ) -> dict[str, list[str]]:
        """
        Simple keyword-based categorization.

        Args:
            items: Context strings
            query: User query

        Returns:
            Dict mapping categories to items
        """
        ##Step purpose: Extract query words for categorization
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        categorized: dict[str, list[str]] = defaultdict(list)

        ##Loop purpose: Categorize each item
        for item in items:
            item_words = set(re.findall(r"\b\w{3,}\b", item.lower()))
            overlap = query_words & item_words

            ##Condition purpose: Use overlap word as category
            if overlap:
                ##Step purpose: Pick most common overlapping word
                category = max(overlap, key=lambda w: query.lower().count(w))
                categorized[category.title()].append(item)
            else:
                categorized["Other"].append(item)

        return dict(categorized)

    ##Method purpose: Classify by category relevance to query
    def _classify_by_category_relevance(
        self,
        categorized: Mapping[str, list[str]],
        query: str,
    ) -> dict[ContextTier, list[str]]:
        """
        Classify items by category relevance to query.

        Args:
            categorized: Items grouped by category
            query: User query

        Returns:
            Dict mapping tiers to items
        """
        query_words = set(query.lower().split())
        tiers: dict[ContextTier, list[str]] = {
            ContextTier.HIGH: [],
            ContextTier.MEDIUM: [],
            ContextTier.LOW: [],
        }

        ##Loop purpose: Classify each category
        for category, items in categorized.items():
            category_words = set(category.lower().split())
            word_overlap = len(query_words & category_words)
            category_size = len(items)

            ##Condition purpose: High tier - direct matches, focused categories
            if word_overlap > 0 and category_size <= 3:
                tiers[ContextTier.HIGH].extend(items[: self._config.max_items_per_tier])
            ##Condition purpose: Medium tier - related categories
            elif word_overlap > 0 or category_size <= 5:
                tiers[ContextTier.MEDIUM].extend(items[: self._config.max_items_per_tier])
            ##Condition purpose: Low tier - distant categories
            else:
                tiers[ContextTier.LOW].extend(items[: self._config.max_items_per_tier])

        return tiers

    ##Method purpose: Create flat organization when disabled
    def _flat_organization(
        self,
        scored_context: ScoredContext,
    ) -> OrganizedContext:
        """
        Create flat organization without categorization or tiering.

        Args:
            scored_context: Scored context items

        Returns:
            OrganizedContext with all items in medium tier
        """
        items = [item.content for item in scored_context.items]

        return OrganizedContext(
            categorized={},
            tiers={
                ContextTier.HIGH: [],
                ContextTier.MEDIUM: items,
                ContextTier.LOW: [],
            },
            query=scored_context.query,
        )

    ##Method purpose: Summarize a tier for compact representation
    def summarize_tier(
        self,
        organized: OrganizedContext,
        tier: ContextTier,
        max_length: int = 500,
    ) -> str:
        """
        Create a summary of items in a tier.

        Args:
            organized: Organized context
            tier: Tier to summarize
            max_length: Maximum summary length

        Returns:
            Summary string
        """
        items = organized.get_tier(tier)

        ##Condition purpose: Handle empty tier
        if not items:
            return f"No {tier.value} priority items."

        ##Step purpose: Combine items with truncation
        combined = " | ".join(items)

        ##Condition purpose: Truncate if too long
        if len(combined) > max_length:
            return combined[: max_length - 3] + "..."

        return combined
