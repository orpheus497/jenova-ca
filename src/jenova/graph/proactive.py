##Script function and purpose: Proactive Engine - Autonomous suggestion system for cognitive engagement
##Dependency purpose: Analyzes cognitive graph state and conversation history to generate context-aware proactive suggestions
"""Proactive Engine for JENOVA.

This module generates proactive suggestions by analyzing the cognitive graph
state and conversation history. Provides context-aware suggestions to encourage
user engagement and cognitive development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from jenova.config.models import JenovaConfig

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Enum defining suggestion categories for variety and targeted engagement
class SuggestionCategory(Enum):
    """Suggestion category types."""
    
    EXPLORE = "explore"
    """Encourage exploration of new topics."""
    
    VERIFY = "verify"
    """Verify unconfirmed assumptions."""
    
    DEVELOP = "develop"
    """Develop underdeveloped insights."""
    
    CONNECT = "connect"
    """Connect related but unlinked concepts."""
    
    REFLECT = "reflect"
    """Encourage reflection on patterns."""


##Class purpose: Protocol for graph operations needed by proactive engine
class GraphProtocol(Protocol):
    """Protocol for graph operations."""
    
    ##Method purpose: Get nodes by type and username
    def get_nodes_by_user(self, username: str) -> list[object]:
        """Get all nodes for a user."""
        ...
    
    ##Method purpose: Add node to graph
    def add_node(self, node: object) -> None:
        """Add node to graph."""
        ...


##Class purpose: Protocol for LLM operations
class LLMProtocol(Protocol):
    """Protocol for LLM operations."""
    
    ##Method purpose: Generate text from prompt
    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        ...


##Class purpose: Configuration for proactive engine
@dataclass
class ProactiveConfig:
    """Configuration for proactive engine.
    
    Attributes:
        min_history_length: Minimum conversation history length (default: 3).
        suggestion_cooldown_hours: Hours between suggestions (default: 1.0).
        category_rotation: Whether to rotate categories (default: True).
        enabled: Whether proactive engine is enabled (default: True).
    """
    
    min_history_length: int = 3
    suggestion_cooldown_hours: float = 1.0
    category_rotation: bool = True
    enabled: bool = True


##Class purpose: Suggestion result with metadata
@dataclass
class SuggestionResult:
    """Result from proactive suggestion generation.
    
    Attributes:
        suggestion: The suggestion text.
        category: Suggestion category.
        timestamp: When suggestion was generated.
    """
    
    suggestion: str
    category: SuggestionCategory
    timestamp: datetime = field(default_factory=datetime.now)


##Class purpose: Analyzes cognitive graph and conversation history to generate context-aware proactive suggestions
class ProactiveEngine:
    """Analyzes cognitive graph to generate proactive suggestions.
    
    The Proactive Engine analyzes the cognitive graph to identify areas of
    interest and generates proactive suggestions or questions for the user.
    Enhanced with conversation history analysis, configurable triggers, and
    suggestion categorization.
    
    Attributes:
        config: Proactive engine configuration.
        graph: Graph protocol for state queries.
        llm: LLM protocol for suggestion generation.
        last_category: Last used suggestion category for rotation.
        suggestion_history: History of generated suggestions.
    """
    
    ##Method purpose: Initialize proactive engine with graph, LLM, and configuration
    def __init__(
        self,
        config: ProactiveConfig,
        graph: GraphProtocol,
        llm: LLMProtocol,
    ) -> None:
        """Initialize the proactive engine.
        
        Args:
            config: Proactive engine configuration.
            graph: Graph for state queries.
            llm: LLM for suggestion generation.
        """
        ##Step purpose: Store configuration and dependencies
        self.config = config
        self.graph = graph
        self.llm = llm
        
        ##Step purpose: Initialize state
        self.last_category: SuggestionCategory | None = None
        self.suggestion_history: list[dict[str, object]] = []
        
        ##Action purpose: Log initialization
        logger.info(
            "proactive_engine_initialized",
            enabled=self.config.enabled,
            min_history_length=self.config.min_history_length,
            cooldown_hours=self.config.suggestion_cooldown_hours,
        )
    
    ##Method purpose: Analyze conversation history to extract patterns and topics
    def _analyze_conversation_patterns(self, history: list[str]) -> dict[str, object]:
        """Analyze recent conversation for recurring topics and patterns.
        
        Args:
            history: List of conversation messages.
            
        Returns:
            Dictionary with topics, question_count, sentiment_trend, recurring_themes.
        """
        ##Condition purpose: Return empty analysis if history too short
        if len(history) < self.config.min_history_length:
            return {
                "topics": [],
                "question_count": 0,
                "sentiment_trend": "neutral",
                "recurring_themes": [],
                "has_sufficient_context": False,
            }
        
        ##Step purpose: Count questions in history
        question_count = sum(1 for msg in history if "?" in msg)
        
        ##Step purpose: Extract topics using keyword analysis
        all_words = " ".join(history).lower().split()
        word_freq: dict[str, int] = {}
        
        ##Step purpose: Define stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
            "this", "that", "these", "those", "am", "and", "or", "but", "if",
            "then", "so", "than", "too", "very", "just", "only", "own", "same",
            "all", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        }
        
        ##Loop purpose: Count word frequencies
        for word in all_words:
            ##Condition purpose: Filter short words and stop words
            if len(word) > 3 and word not in stop_words and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        ##Step purpose: Identify recurring themes (words appearing 2+ times)
        recurring_themes = [
            word for word, count in word_freq.items() if count >= 2
        ]
        recurring_themes.sort(key=lambda w: word_freq[w], reverse=True)
        
        ##Step purpose: Extract top topics
        topics = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:5]
        
        return {
            "topics": topics,
            "question_count": question_count,
            "sentiment_trend": "neutral",  # Could be enhanced with sentiment analysis
            "recurring_themes": recurring_themes[:3],
            "has_sufficient_context": True,
        }
    
    ##Method purpose: Determine which suggestion category to use based on context and rotation
    def _select_suggestion_category(
        self,
        unverified_count: int,
        underdeveloped_count: int,
        high_potential_count: int,
        conversation_patterns: dict[str, object],
    ) -> SuggestionCategory:
        """Select the most appropriate suggestion category based on current state.
        
        Args:
            unverified_count: Number of unverified assumptions.
            underdeveloped_count: Number of underdeveloped insights.
            high_potential_count: Number of high-potential nodes.
            conversation_patterns: Conversation analysis results.
            
        Returns:
            Selected suggestion category.
        """
        ##Step purpose: Build priority list based on current state
        priorities: list[tuple[SuggestionCategory, float]] = []
        
        ##Condition purpose: High priority for unverified assumptions
        if unverified_count > 0:
            priorities.append((SuggestionCategory.VERIFY, 0.8 + (unverified_count * 0.1)))
        
        ##Condition purpose: Medium-high priority for underdeveloped insights
        if underdeveloped_count > 0:
            priorities.append((SuggestionCategory.DEVELOP, 0.7 + (underdeveloped_count * 0.05)))
        
        ##Condition purpose: Medium priority for high-potential nodes
        if high_potential_count > 2:
            priorities.append((SuggestionCategory.CONNECT, 0.6 + (high_potential_count * 0.05)))
        
        ##Condition purpose: Add exploration if conversation has clear topics
        if (
            conversation_patterns.get("has_sufficient_context")
            and conversation_patterns.get("topics")
        ):
            priorities.append((SuggestionCategory.EXPLORE, 0.5))
        
        ##Step purpose: Always include reflection as fallback
        priorities.append((SuggestionCategory.REFLECT, 0.3))
        
        ##Step purpose: Sort by priority score
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        ##Condition purpose: Apply category rotation if enabled
        if self.config.category_rotation and self.last_category:
            ##Loop purpose: Prefer different category than last time
            for category, _score in priorities:
                if category != self.last_category:
                    return category
        
        ##Step purpose: Return highest priority category
        return priorities[0][0] if priorities else SuggestionCategory.REFLECT
    
    ##Method purpose: Check if trigger conditions are met for generating a suggestion
    def should_suggest(self, username: str, turn_count: int) -> bool:
        """Determine if conditions are right for generating a proactive suggestion.
        
        Args:
            username: Username to check for.
            turn_count: Current turn count.
            
        Returns:
            True if suggestion should be generated, False otherwise.
        """
        ##Condition purpose: Return False if disabled
        if not self.config.enabled:
            return False
        
        ##Error purpose: Handle errors gracefully
        try:
            ##Step purpose: Check cooldown period
            user_nodes = self.graph.get_nodes_by_user(username)
            recent_suggestions = [
                node for node in user_nodes
                if getattr(node, "node_type", "") == "proactive_suggestion"
            ]
            
            if recent_suggestions:
                ##Step purpose: Find latest suggestion timestamp
                latest_timestamp: datetime | None = None
                for suggestion in recent_suggestions:
                    metadata = getattr(suggestion, "metadata", {})
                    timestamp_str = metadata.get("timestamp", "")
                    if timestamp_str:
                        try:
                            ts = datetime.fromisoformat(timestamp_str)
                            if latest_timestamp is None or ts > latest_timestamp:
                                latest_timestamp = ts
                        except (ValueError, TypeError):
                            continue
                
                ##Condition purpose: Check cooldown
                if latest_timestamp:
                    hours_since = (datetime.now() - latest_timestamp).total_seconds() / 3600
                    if hours_since < self.config.suggestion_cooldown_hours:
                        return False
            
            ##Step purpose: Check if there's content to suggest about
            unverified = [
                node for node in user_nodes
                if getattr(node, "node_type", "") == "assumption"
                and getattr(node, "metadata", {}).get("status") == "unverified"
            ]
            
            insights = [
                node for node in user_nodes
                if getattr(node, "node_type", "") == "insight"
            ]
            
            ##Condition purpose: Need at least some cognitive content
            if len(unverified) == 0 and len(insights) < 3:
                return False
            
            return True
            
        except Exception as e:
            logger.warning("proactive_should_suggest_failed", error=str(e), username=username)
            return False
    
    ##Method purpose: Generate proactive suggestion based on cognitive graph state and conversation history
    def get_suggestion(self, username: str, history: list[str]) -> SuggestionResult | None:
        """Generate a context-aware proactive suggestion for the user.
        
        Args:
            username: Username to generate suggestion for.
            history: Conversation history.
            
        Returns:
            SuggestionResult if successful, None otherwise.
        """
        ##Condition purpose: Return None if disabled
        if not self.config.enabled:
            return None
        
        ##Error purpose: Handle errors gracefully
        try:
            ##Step purpose: Analyze conversation patterns
            conversation_patterns = self._analyze_conversation_patterns(history)
            
            ##Step purpose: Get user nodes
            user_nodes = self.graph.get_nodes_by_user(username)
            
            ##Step purpose: Get unverified assumptions
            unverified_assumptions = [
                getattr(node, "content", "")
                for node in user_nodes
                if getattr(node, "node_type", "") == "assumption"
                and getattr(node, "metadata", {}).get("status") == "unverified"
            ]
            
            ##Step purpose: Get insights
            all_insights = [
                node for node in user_nodes
                if getattr(node, "node_type", "") == "insight"
            ]
            
            ##Step purpose: Categorize insights by centrality (proxy)
            underdeveloped_nodes = [
                node for node in all_insights
                if getattr(node, "metadata", {}).get("centrality", 0) < 0.5
            ]
            underdeveloped_content = [
                getattr(node, "content", "") for node in underdeveloped_nodes[:3]
            ]
            
            high_potential_nodes = [
                node for node in all_insights
                if getattr(node, "metadata", {}).get("centrality", 0) > 1.5
            ]
            high_potential_content = [
                getattr(node, "content", "") for node in high_potential_nodes[:3]
            ]
            
            ##Step purpose: Get recent suggestions to avoid repetition
            recent_suggestions = [
                getattr(node, "content", "")
                for node in user_nodes
                if getattr(node, "node_type", "") == "proactive_suggestion"
            ]
            
            ##Step purpose: Select suggestion category
            category = self._select_suggestion_category(
                len(unverified_assumptions),
                len(underdeveloped_nodes),
                len(high_potential_nodes),
                conversation_patterns,
            )
            
            ##Step purpose: Get category guidance
            category_guidance = self._get_category_guidance(category)
            
            ##Step purpose: Build conversation context
            conversation_context = ""
            if conversation_patterns.get("has_sufficient_context"):
                topics = conversation_patterns.get("topics", [])
                themes = conversation_patterns.get("recurring_themes", [])
                if topics or themes:
                    conversation_context = f"""
- Recent Conversation Topics: {topics[:3] if topics else "None"}
- Recurring Themes: {themes if themes else "None"}
- Questions Asked: {conversation_patterns.get("question_count", 0)}"""
            
            ##Step purpose: Build prompt
            prompt = f"""Based on the following information about the user's cognitive state and recent conversation, generate a single, concise, and engaging proactive suggestion or question.

== Suggestion Category: {category.value.upper()} ==
{category_guidance}

== Cognitive Graph Context ==
- Unverified Assumptions: {unverified_assumptions if unverified_assumptions else "None"}
- Underdeveloped Insights (low connections): {underdeveloped_content if underdeveloped_content else "None"}
- High-Potential Insights (well-connected): {high_potential_content if high_potential_content else "None"}{conversation_context}

== Constraints ==
- Recent Suggestions (avoid repeating): {recent_suggestions if recent_suggestions else "None"}
- Keep the suggestion focused and actionable
- Be creative and thought-provoking

Generate a {category.value} suggestion:"""
            
            ##Action purpose: Generate suggestion via LLM
            suggestion_text = self.llm.generate_text(prompt, temperature=0.7)
            
            ##Condition purpose: Validate and return suggestion
            if suggestion_text and suggestion_text.strip() and suggestion_text not in recent_suggestions:
                ##Step purpose: Create suggestion result
                result = SuggestionResult(
                    suggestion=suggestion_text.strip(),
                    category=category,
                )
                
                ##Step purpose: Update category rotation tracking
                self.last_category = category
                
                ##Step purpose: Track in history
                self.suggestion_history.append({
                    "suggestion": suggestion_text.strip(),
                    "category": category.value,
                    "timestamp": datetime.now().isoformat(),
                    "acted_upon": False,
                })
                
                logger.debug(
                    "proactive_suggestion_generated",
                    category=category.value,
                    username=username,
                )
                
                return result
            
            return None
            
        except Exception as e:
            logger.error("proactive_suggestion_failed", error=str(e), username=username)
            return None
    
    ##Method purpose: Get category-specific guidance for prompt construction
    def _get_category_guidance(self, category: SuggestionCategory) -> str:
        """Return guidance text for each suggestion category.
        
        Args:
            category: Suggestion category.
            
        Returns:
            Guidance text for the category.
        """
        guidance_map = {
            SuggestionCategory.EXPLORE: "Encourage the user to explore a new topic or dive deeper into something they've mentioned. Spark curiosity about unexplored areas.",
            SuggestionCategory.VERIFY: "Ask the user to confirm or clarify an assumption that hasn't been verified yet. Frame it as a question seeking clarity.",
            SuggestionCategory.DEVELOP: "Suggest that the user elaborate on an underdeveloped insight. Help them build connections to other ideas.",
            SuggestionCategory.CONNECT: "Point out potential connections between well-developed insights. Encourage synthesis of related concepts.",
            SuggestionCategory.REFLECT: "Invite the user to reflect on patterns or themes in their recent conversations. Encourage meta-level thinking.",
        }
        return guidance_map.get(category, "Generate a helpful suggestion for the user.")
    
    ##Method purpose: Mark a suggestion as acted upon for engagement tracking
    def mark_suggestion_engaged(self, suggestion_text: str) -> None:
        """Mark a suggestion as having been acted upon by the user.
        
        Args:
            suggestion_text: Text of the suggestion.
        """
        ##Loop purpose: Find and update suggestion in history
        for item in self.suggestion_history:
            if item.get("suggestion") == suggestion_text:
                item["acted_upon"] = True
                break
    
    ##Method purpose: Get engagement statistics for suggestion categories
    def get_engagement_stats(self) -> dict[str, dict[str, int]]:
        """Return engagement statistics by suggestion category.
        
        Returns:
            Dictionary mapping category to total and engaged counts.
        """
        stats: dict[str, dict[str, int]] = {}
        
        ##Loop purpose: Aggregate statistics
        for item in self.suggestion_history:
            category = str(item.get("category", "unknown"))
            if category not in stats:
                stats[category] = {"total": 0, "engaged": 0}
            stats[category]["total"] += 1
            if item.get("acted_upon"):
                stats[category]["engaged"] += 1
        
        return stats
