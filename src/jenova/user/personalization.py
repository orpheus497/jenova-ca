# The JENOVA Cognitive Architecture - Personalization Engine
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 10: Personalization engine for adaptive responses.

Provides:
- Adaptive response styling based on user preferences
- Proactive suggestions based on context and patterns
- Custom shortcuts and workflows
- Context-aware recommendations
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Suggestion:
    """A proactive suggestion for the user."""

    content: str
    reason: str
    confidence: float
    category: str  # tip, command, topic, resource


class PersonalizationEngine:
    """
    Personalizes interactions based on user profile and patterns.

    Features:
    - Adaptive response style (concise, balanced, detailed)
    - Expertise-appropriate explanations
    - Proactive suggestions
    - Context-aware recommendations
    """

    def __init__(self, user_profile, file_logger):
        self.user_profile = user_profile
        self.file_logger = file_logger

    def adapt_response(self, response: str, context: Dict) -> str:
        """
        Adapt a response based on user preferences and context.

        Args:
            response: The original response
            context: Context information (semantic analysis, etc.)

        Returns:
            Adapted response
        """
        if not self.user_profile:
            return response

        # Get user preferences
        prefs = self.user_profile.preferences

        # Apply response style
        adapted = self._apply_response_style(response, prefs.response_style)

        # Apply communication style
        adapted = self._apply_communication_style(adapted, prefs.communication_style)

        # Add proactive suggestions if enabled
        if prefs.proactive_suggestions:
            suggestions = self.generate_suggestions(context)
            if suggestions:
                suggestion_text = self._format_suggestions(suggestions[:2])  # Top 2
                adapted += f"\n\n{suggestion_text}"

        return adapted

    def _apply_response_style(self, response: str, style: str) -> str:
        """
        Adapt response verbosity based on preferred style.

        Args:
            response: Original response
            style: One of "concise", "balanced", "detailed"

        Returns:
            Adapted response
        """
        if style == "concise":
            # For concise style, try to shorten without losing key information
            # In a full implementation, would use summarization
            lines = response.split("\n")
            # Keep only essential lines (not empty, not just transitional)
            essential = [
                line
                for line in lines
                if line.strip()
                and not line.strip().lower().startswith(("so", "therefore", "thus"))
            ]
            return "\n".join(essential[:5])  # Limit to 5 key lines

        elif style == "detailed":
            # For detailed style, response is already good
            # Could add examples, references, etc.
            return response

        # Balanced is the default
        return response

    def _apply_communication_style(self, response: str, style: str) -> str:
        """
        Adapt communication tone based on preferred style.

        Args:
            response: Original response
            style: One of "formal", "friendly", "casual", "technical"

        Returns:
            Adapted response
        """
        # This is a simplified implementation
        # In production, would use more sophisticated NLG techniques

        if style == "technical":
            # Add technical precision indicators
            response = response.replace("about", "approximately")
            response = response.replace("kind of", "similar to")

        elif style == "casual":
            # Make it more conversational
            if not response.startswith(("Hey", "So", "Well")):
                response = f"So, {response[0].lower()}{response[1:]}"

        elif style == "formal":
            # Make it more formal
            response = response.replace("don't", "do not")
            response = response.replace("can't", "cannot")
            response = response.replace("won't", "will not")

        # Friendly is the default
        return response

    def generate_suggestions(self, context: Dict) -> List[Suggestion]:
        """
        Generate proactive suggestions based on context and user patterns.

        Args:
            context: Context dictionary with semantic_analysis, etc.

        Returns:
            List of suggestions ordered by relevance
        """
        suggestions = []

        if not self.user_profile:
            return suggestions

        # Get semantic analysis from context
        semantic = context.get("semantic_analysis")
        if not semantic:
            return suggestions

        # Suggest commands based on intent
        if semantic.intent.value == "question":
            # If user asks questions frequently, suggest memory commands
            if self.user_profile.stats.questions_asked > 10:
                if not self._command_used_recently("/memory-insight"):
                    suggestions.append(
                        Suggestion(
                            content="Try /memory-insight to discover patterns in your past questions",
                            reason="You've asked many questions - insights might help",
                            confidence=0.8,
                            category="command",
                        )
                    )

        # Suggest topics based on discussion patterns
        top_topics = self.user_profile.get_top_topics(limit=3)
        if top_topics and semantic.topics:
            current_topic = semantic.topics[0] if semantic.topics else None
            if current_topic and current_topic not in [t for t, _ in top_topics]:
                # New topic - suggest related topics from history
                suggestions.append(
                    Suggestion(
                        content=f"This relates to {top_topics[0][0]} which you've discussed {top_topics[0][1]} times",
                        reason="Connecting to your past interests",
                        confidence=0.7,
                        category="topic",
                    )
                )

        # Suggest expertise adjustment
        if self._should_adjust_expertise(semantic):
            current_level = self.user_profile.preferences.expertise_level
            if current_level == "intermediate":
                suggestions.append(
                    Suggestion(
                        content="You seem comfortable with advanced concepts - consider updating your expertise level to 'advanced' in /profile",
                        reason="Your vocabulary and questions indicate higher expertise",
                        confidence=0.6,
                        category="tip",
                    )
                )

        # Suggest learning mode if disabled but user shows learning behavior
        if not self.user_profile.preferences.learning_mode:
            if len(self.user_profile.corrections) > 5:
                suggestions.append(
                    Suggestion(
                        content="Enable learning mode to help JENOVA adapt better to your preferences",
                        reason="You've provided corrections that could improve the system",
                        confidence=0.75,
                        category="tip",
                    )
                )

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions

    def _command_used_recently(self, command: str) -> bool:
        """Check if a command was used in recent interactions."""
        if not self.user_profile:
            return False

        # Check last 20 interactions (simplified - would check actual history)
        return self.user_profile.preferred_commands.get(command, 0) > 0

    def _should_adjust_expertise(self, semantic_analysis) -> bool:
        """
        Determine if user expertise level should be adjusted.

        Args:
            semantic_analysis: Semantic analysis of current query

        Returns:
            True if expertise should be adjusted upward
        """
        if not self.user_profile:
            return False

        # Check vocabulary sophistication
        vocab_size = len(self.user_profile.vocabulary)
        technical_terms = sum(
            1
            for kw in semantic_analysis.keywords
            if kw
            in {
                "algorithm",
                "optimization",
                "architecture",
                "implementation",
                "refactor",
                "paradigm",
                "abstraction",
                "encapsulation",
            }
        )

        current_level = self.user_profile.preferences.expertise_level

        # Suggest upgrade if:
        # - Intermediate user with large vocabulary and technical terms
        if (
            current_level == "intermediate"
            and vocab_size > 500
            and technical_terms >= 2
        ):
            return True

        # - Beginner user with moderate vocabulary
        if current_level == "beginner" and vocab_size > 200:
            return True

        return False

    def _format_suggestions(self, suggestions: List[Suggestion]) -> str:
        """Format suggestions for display."""
        if not suggestions:
            return ""

        lines = ["💡 Suggestions:"]
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"  {i}. {suggestion.content}")

        return "\n".join(lines)

    def get_custom_shortcuts(self) -> Dict[str, str]:
        """
        Get custom shortcuts based on user patterns.

        Returns:
            Dictionary mapping shortcut to expansion
        """
        shortcuts = {}

        if not self.user_profile:
            return shortcuts

        # Create shortcuts for frequently discussed topics
        top_topics = self.user_profile.get_top_topics(limit=5)
        for topic, count in top_topics:
            if count > 10:
                shortcut_key = topic[:3].lower()
                shortcuts[shortcut_key] = f"Tell me more about {topic}"

        # Create shortcuts for frequently used commands
        for cmd, count in self.user_profile.preferred_commands.most_common(5):
            if count > 5:
                # Already a slash command, no need for shortcut
                pass

        return shortcuts

    def recommend_next_action(self, context: Dict) -> Optional[str]:
        """
        Recommend the next logical action based on context and patterns.

        Args:
            context: Current context

        Returns:
            Recommendation string or None
        """
        if not self.user_profile:
            return None

        semantic = context.get("semantic_analysis")
        if not semantic:
            return None

        # Pattern: After many questions, suggest insight development
        if self.user_profile.stats.questions_asked % 10 == 0:
            return "You've asked 10+ questions. Consider using /insight to develop insights from this conversation."

        # Pattern: New topic - suggest related memory search
        if semantic.topics and semantic.topics[0] not in [
            t for t, _ in self.user_profile.get_top_topics()
        ]:
            return f"New topic detected: {semantic.topics[0]}. Use memory search to find related past discussions."

        # Pattern: Complex query - suggest breaking it down
        if semantic.rhetorical_elements.get("complexity") == "complex":
            return "Complex query detected. Consider breaking it into smaller questions for better results."

        return None

    def adapt_search_parameters(self, base_params: Dict) -> Dict:
        """
        Adapt search parameters based on user expertise and preferences.

        Args:
            base_params: Base search parameters

        Returns:
            Adapted parameters
        """
        if not self.user_profile:
            return base_params

        adapted = base_params.copy()

        # Adjust based on expertise level
        if self.user_profile.preferences.expertise_level == "expert":
            # Experts want more results, higher precision
            adapted["max_results"] = adapted.get("max_results", 5) + 3
            adapted["similarity_threshold"] = max(
                adapted.get("similarity_threshold", 0.7), 0.8
            )

        elif self.user_profile.preferences.expertise_level == "beginner":
            # Beginners want fewer, more relevant results
            adapted["max_results"] = min(adapted.get("max_results", 5), 3)
            adapted["similarity_threshold"] = (
                adapted.get("similarity_threshold", 0.7) - 0.1
            )

        return adapted

    def track_suggestion_feedback(self, suggestion: Suggestion, accepted: bool):
        """
        Track whether a suggestion was accepted.

        Args:
            suggestion: The suggestion that was presented
            accepted: Whether the user accepted it
        """
        if self.user_profile:
            self.user_profile.record_suggestion_feedback(accepted)
            self.file_logger.log_info(
                f"Suggestion feedback: {suggestion.category} - "
                f"{'accepted' if accepted else 'declined'} "
                f"(success rate: {self.user_profile.get_suggestion_success_rate():.2%})"
            )
