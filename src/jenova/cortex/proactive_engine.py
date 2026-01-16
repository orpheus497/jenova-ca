##Script function and purpose: Proactive Engine for The JENOVA Cognitive Architecture
##This module generates proactive suggestions by analyzing the cognitive graph state and conversation history

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

##Enum purpose: Define suggestion categories for variety and targeted engagement
class SuggestionCategory(Enum):
    EXPLORE = "explore"      # Encourage exploration of new topics
    VERIFY = "verify"        # Verify unconfirmed assumptions
    DEVELOP = "develop"      # Develop underdeveloped insights
    CONNECT = "connect"      # Connect related but unlinked concepts
    REFLECT = "reflect"      # Encourage reflection on patterns

##Class purpose: Analyzes cognitive graph and conversation history to generate context-aware proactive suggestions
class ProactiveEngine:
    """
    The Proactive Engine analyzes the cognitive graph to identify areas of interest
    and generates proactive suggestions or questions for the user. Enhanced with
    conversation history analysis, configurable triggers, and suggestion categorization.
    """
    ##Function purpose: Initialize proactive engine with cortex, LLM, logger, and configuration
    def __init__(
        self, 
        cortex: Any, 
        llm: Any, 
        ui_logger: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.cortex = cortex
        self.llm = llm
        self.ui_logger = ui_logger
        self.config = config or {}
        self.recent_suggestions: List[str] = []
        self.suggestion_history: List[Dict[str, Any]] = []
        
        ##Block purpose: Load proactive engine configuration with defaults
        proactive_config = self.config.get('proactive_engine', {})
        self.min_history_length = proactive_config.get('min_history_length', 3)
        self.suggestion_cooldown_hours = proactive_config.get('suggestion_cooldown_hours', 1)
        self.category_rotation = proactive_config.get('category_rotation', True)
        self.last_category: Optional[SuggestionCategory] = None

    ##Function purpose: Analyze conversation history to extract patterns and topics
    def _analyze_conversation_patterns(self, history: List[str]) -> Dict[str, Any]:
        """Analyze recent conversation for recurring topics, questions, and user interests.
        
        Returns a dictionary with:
        - topics: List of identified topics from conversation
        - question_count: Number of questions asked by user
        - sentiment_trend: Overall sentiment direction (positive/neutral/negative)
        - recurring_themes: Topics mentioned multiple times
        """
        ##Block purpose: Return empty analysis if history is too short
        if len(history) < self.min_history_length:
            return {
                'topics': [],
                'question_count': 0,
                'sentiment_trend': 'neutral',
                'recurring_themes': [],
                'has_sufficient_context': False
            }
        
        ##Block purpose: Count questions in history
        question_count = sum(1 for msg in history if '?' in msg)
        
        ##Block purpose: Extract topics using simple keyword analysis (fast, no LLM call)
        all_words = ' '.join(history).lower().split()
        word_freq: Dict[str, int] = {}
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                      'this', 'that', 'these', 'those', 'am', 'and', 'or', 'but', 'if',
                      'then', 'so', 'than', 'too', 'very', 'just', 'only', 'own', 'same',
                      'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too'}
        
        for word in all_words:
            ##Block purpose: Filter short words and stop words
            if len(word) > 3 and word not in stop_words and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        ##Block purpose: Identify recurring themes (words appearing 2+ times)
        recurring_themes = [word for word, count in word_freq.items() if count >= 2]
        recurring_themes.sort(key=lambda w: word_freq[w], reverse=True)
        
        ##Block purpose: Extract top topics (most frequent meaningful words)
        topics = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:5]
        
        return {
            'topics': topics,
            'question_count': question_count,
            'sentiment_trend': 'neutral',  # Could be enhanced with sentiment analysis
            'recurring_themes': recurring_themes[:3],
            'has_sufficient_context': True
        }

    ##Function purpose: Determine which suggestion category to use based on context and rotation
    def _select_suggestion_category(
        self, 
        unverified_count: int,
        underdeveloped_count: int,
        high_potential_count: int,
        conversation_patterns: Dict[str, Any]
    ) -> SuggestionCategory:
        """Select the most appropriate suggestion category based on current state.
        
        Uses a priority system with category rotation to ensure variety.
        """
        ##Block purpose: Build priority list based on current state
        priorities: List[Tuple[SuggestionCategory, float]] = []
        
        ##Block purpose: High priority for unverified assumptions
        if unverified_count > 0:
            priorities.append((SuggestionCategory.VERIFY, 0.8 + (unverified_count * 0.1)))
        
        ##Block purpose: Medium-high priority for underdeveloped insights
        if underdeveloped_count > 0:
            priorities.append((SuggestionCategory.DEVELOP, 0.7 + (underdeveloped_count * 0.05)))
        
        ##Block purpose: Medium priority for high-potential nodes (meta-insight opportunity)
        if high_potential_count > 2:
            priorities.append((SuggestionCategory.CONNECT, 0.6 + (high_potential_count * 0.05)))
        
        ##Block purpose: Add exploration if conversation has clear topics
        if conversation_patterns.get('has_sufficient_context') and conversation_patterns.get('topics'):
            priorities.append((SuggestionCategory.EXPLORE, 0.5))
        
        ##Block purpose: Always include reflection as fallback
        priorities.append((SuggestionCategory.REFLECT, 0.3))
        
        ##Block purpose: Sort by priority score
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        ##Block purpose: Apply category rotation if enabled to ensure variety
        if self.category_rotation and self.last_category:
            ##Block purpose: Prefer different category than last time if close in priority
            for category, score in priorities:
                if category != self.last_category:
                    return category
        
        ##Block purpose: Return highest priority category
        return priorities[0][0]

    ##Function purpose: Check if trigger conditions are met for generating a suggestion
    def should_suggest(self, username: str, turn_count: int) -> bool:
        """Determine if conditions are right for generating a proactive suggestion.
        
        Considers:
        - Minimum turn count since last suggestion
        - Time since last suggestion (cooldown)
        - Availability of content to suggest about
        """
        ##Block purpose: Check cooldown period
        recent_suggestions = self.cortex.get_all_nodes_by_type('proactive_suggestion', username)
        if recent_suggestions:
            latest = max(recent_suggestions, key=lambda s: s['timestamp'])
            hours_since = (datetime.now() - datetime.fromisoformat(latest['timestamp'])).total_seconds() / 3600
            if hours_since < self.suggestion_cooldown_hours:
                return False
        
        ##Block purpose: Check if there's content to suggest about
        unverified = self.cortex.get_all_nodes_by_type('assumption', username)
        unverified = [a for a in unverified if a.get('metadata', {}).get('status') == 'unverified']
        
        insights = self.cortex.get_all_nodes_by_type('insight', username)
        
        ##Block purpose: Need at least some cognitive content to work with
        if len(unverified) == 0 and len(insights) < 3:
            return False
        
        return True

    ##Function purpose: Generate proactive suggestion based on cognitive graph state and conversation history
    def get_suggestion(self, username: str, history: List[str]) -> Optional[str]:
        """Generates a context-aware proactive suggestion for the user.
        
        Analyzes conversation history, cognitive graph state, and uses category
        rotation to provide varied, relevant suggestions.
        """
        ##Block purpose: Analyze conversation patterns for context
        conversation_patterns = self._analyze_conversation_patterns(history)
        
        ##Block purpose: Get unverified assumptions
        unverified_assumptions = self.cortex.get_all_nodes_by_type('assumption', username)
        unverified_assumptions = [a['content'] for a in unverified_assumptions 
                                   if a.get('metadata', {}).get('status') == 'unverified']

        ##Block purpose: Ensure centrality is up-to-date and get node categories
        self.cortex.calculate_centrality()
        
        all_insights = self.cortex.get_all_nodes_by_type('insight', username)
        underdeveloped_nodes = [node for node in all_insights 
                                if node['metadata'].get('centrality', 0) < 0.5]
        underdeveloped_content = [node['content'] for node in underdeveloped_nodes[:3]]

        high_potential_nodes = [node for node in all_insights 
                                if node['metadata'].get('centrality', 0) > 1.5]
        high_potential_content = [node['content'] for node in high_potential_nodes[:3]]

        ##Block purpose: Get recent suggestions to avoid repetition
        recent_suggestions = self.cortex.get_all_nodes_by_type('proactive_suggestion', username)
        recent_suggestions_content = [s['content'] for s in recent_suggestions 
                                       if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days < 1]

        ##Block purpose: Select suggestion category based on current state
        category = self._select_suggestion_category(
            len(unverified_assumptions),
            len(underdeveloped_nodes),
            len(high_potential_nodes),
            conversation_patterns
        )
        
        ##Block purpose: Build category-specific prompt guidance
        category_guidance = self._get_category_guidance(category)

        ##Block purpose: Build context-aware prompt with conversation patterns
        conversation_context = ""
        if conversation_patterns.get('has_sufficient_context'):
            topics = conversation_patterns.get('topics', [])
            themes = conversation_patterns.get('recurring_themes', [])
            if topics or themes:
                conversation_context = f"""
- Recent Conversation Topics: {topics[:3] if topics else "None"}
- Recurring Themes: {themes if themes else "None"}
- Questions Asked: {conversation_patterns.get('question_count', 0)}"""

        prompt = f"""Based on the following information about the user's cognitive state and recent conversation, generate a single, concise, and engaging proactive suggestion or question.

== Suggestion Category: {category.value.upper()} ==
{category_guidance}

== Cognitive Graph Context ==
- Unverified Assumptions: {unverified_assumptions if unverified_assumptions else "None"}
- Underdeveloped Insights (low connections): {underdeveloped_content if underdeveloped_content else "None"}
- High-Potential Insights (well-connected): {high_potential_content if high_potential_content else "None"}{conversation_context}

== Constraints ==
- Recent Suggestions (avoid repeating): {recent_suggestions_content if recent_suggestions_content else "None"}
- Keep the suggestion focused and actionable
- Be creative and thought-provoking

Generate a {category.value} suggestion:"""

        with self.ui_logger.thinking_process("Generating proactive suggestion..."):
            suggestion = self.llm.generate(prompt, temperature=0.7, stop=["\n\n"])
        
        ##Block purpose: Validate and store suggestion if unique
        if suggestion and suggestion.strip() and suggestion not in recent_suggestions_content:
            ##Block purpose: Store suggestion with category metadata
            metadata = {
                'category': category.value,
                'conversation_topics': conversation_patterns.get('topics', [])[:3]
            }
            self.cortex.add_node('proactive_suggestion', suggestion.strip(), username, metadata=metadata)
            
            ##Block purpose: Update category rotation tracking
            self.last_category = category
            
            ##Block purpose: Track suggestion in local history for engagement analysis
            self.suggestion_history.append({
                'suggestion': suggestion.strip(),
                'category': category.value,
                'timestamp': datetime.now().isoformat(),
                'acted_upon': False  # Will be updated if user engages
            })
            
            return suggestion.strip()
        return None

    ##Function purpose: Get category-specific guidance for prompt construction
    def _get_category_guidance(self, category: SuggestionCategory) -> str:
        """Return guidance text for each suggestion category."""
        guidance = {
            SuggestionCategory.EXPLORE: "Encourage the user to explore a new topic or dive deeper into something they've mentioned. Spark curiosity about unexplored areas.",
            SuggestionCategory.VERIFY: "Ask the user to confirm or clarify an assumption that hasn't been verified yet. Frame it as a question seeking clarity.",
            SuggestionCategory.DEVELOP: "Suggest that the user elaborate on an underdeveloped insight. Help them build connections to other ideas.",
            SuggestionCategory.CONNECT: "Point out potential connections between well-developed insights. Encourage synthesis of related concepts.",
            SuggestionCategory.REFLECT: "Invite the user to reflect on patterns or themes in their recent conversations. Encourage meta-level thinking."
        }
        return guidance.get(category, "Generate a helpful suggestion for the user.")

    ##Function purpose: Mark a suggestion as acted upon for engagement tracking
    def mark_suggestion_engaged(self, suggestion_text: str) -> None:
        """Mark a suggestion as having been acted upon by the user.
        
        This helps track which types of suggestions are most effective.
        """
        for item in self.suggestion_history:
            if item['suggestion'] == suggestion_text:
                item['acted_upon'] = True
                break

    ##Function purpose: Get engagement statistics for suggestion categories
    def get_engagement_stats(self) -> Dict[str, Dict[str, int]]:
        """Return engagement statistics by suggestion category.
        
        Returns counts of total and engaged suggestions per category.
        """
        stats: Dict[str, Dict[str, int]] = {}
        for item in self.suggestion_history:
            category = item['category']
            if category not in stats:
                stats[category] = {'total': 0, 'engaged': 0}
            stats[category]['total'] += 1
            if item['acted_upon']:
                stats[category]['engaged'] += 1
        return stats
