"""
JENOVA Cognitive Architecture - Command Disambiguator Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides command disambiguation using fuzzy matching and scoring
algorithms to help users select the correct command when ambiguous input is given.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from difflib import SequenceMatcher
import re


logger = logging.getLogger(__name__)


@dataclass
class CommandCandidate:
    """Represents a candidate command with relevance score."""

    command: str
    score: float
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare candidates by score (descending)."""
        return self.score > other.score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "command": self.command,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class CommandDisambiguator:
    """
    Command disambiguation system using fuzzy matching and heuristic scoring.

    This disambiguator helps resolve ambiguous commands by:
    - Fuzzy string matching
    - Substring matching
    - Prefix/suffix matching
    - Acronym matching
    - Phonetic similarity
    - Context-based scoring
    """

    def __init__(self, threshold: float = 0.3, max_suggestions: int = 5):
        """
        Initialize the command disambiguator.

        Args:
            threshold: Minimum similarity score to consider a match (0.0-1.0)
            max_suggestions: Maximum number of suggestions to return
        """
        self.threshold = threshold
        self.max_suggestions = max_suggestions
        self.command_history: List[str] = []
        self.command_frequency: Dict[str, int] = {}

    def disambiguate(
        self, command: str, options: List[str], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Disambiguate an ambiguous command by finding the best match.

        Args:
            command: The ambiguous command input
            options: List of possible command options
            context: Optional context information for scoring

        Returns:
            The best matching command, or the original command if no good match
        """
        if not options:
            return command

        if not command or not command.strip():
            return options[0] if options else command

        # Get all candidates with scores
        candidates = self.get_candidates(command, options, context)

        if not candidates:
            logger.warning(f"No candidates found for command: {command}")
            return command

        # Return best candidate
        best = candidates[0]
        logger.info(
            f"Disambiguated '{command}' -> '{best.command}' (score: {best.score:.2f})"
        )
        return best.command

    def get_candidates(
        self, command: str, options: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[CommandCandidate]:
        """
        Get all candidate matches sorted by relevance score.

        Args:
            command: The command to disambiguate
            options: List of possible options
            context: Optional context information

        Returns:
            List of CommandCandidate objects sorted by score (highest first)
        """
        context = context or {}
        candidates = []

        for option in options:
            score, reason = self._calculate_similarity(command, option, context)

            if score >= self.threshold:
                candidates.append(
                    CommandCandidate(
                        command=option,
                        score=score,
                        reason=reason,
                        metadata={"original_score": score},
                    )
                )

        # Sort by score (descending)
        candidates.sort()

        # Limit to max suggestions
        return candidates[: self.max_suggestions]

    def _calculate_similarity(
        self, command: str, option: str, context: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Calculate similarity score between command and option.

        Args:
            command: The input command
            option: The option to compare against
            context: Context information

        Returns:
            Tuple of (score, reason) where score is 0.0-1.0
        """
        command_lower = command.lower().strip()
        option_lower = option.lower().strip()

        # Exact match
        if command_lower == option_lower:
            return (1.0, "exact_match")

        scores = []
        reasons = []

        # 1. Sequence matching (difflib)
        seq_score = SequenceMatcher(None, command_lower, option_lower).ratio()
        scores.append(seq_score * 0.3)
        if seq_score > 0.8:
            reasons.append("high_sequence_similarity")

        # 2. Prefix matching
        if option_lower.startswith(command_lower):
            prefix_score = len(command_lower) / len(option_lower)
            scores.append(prefix_score * 0.3)
            reasons.append("prefix_match")
        elif command_lower.startswith(option_lower):
            prefix_score = len(option_lower) / len(command_lower)
            scores.append(prefix_score * 0.2)
            reasons.append("reverse_prefix_match")

        # 3. Substring matching
        if command_lower in option_lower:
            substr_score = len(command_lower) / len(option_lower)
            scores.append(substr_score * 0.25)
            reasons.append("substring_match")
        elif option_lower in command_lower:
            substr_score = len(option_lower) / len(command_lower)
            scores.append(substr_score * 0.15)
            reasons.append("reverse_substring_match")

        # 4. Word-level matching
        command_words = set(command_lower.split())
        option_words = set(option_lower.split())
        if command_words and option_words:
            word_overlap = len(command_words & option_words)
            word_score = word_overlap / max(len(command_words), len(option_words))
            scores.append(word_score * 0.25)
            if word_score > 0.5:
                reasons.append("word_overlap")

        # 5. Acronym matching
        acronym_score = self._acronym_similarity(command_lower, option_lower)
        if acronym_score > 0:
            scores.append(acronym_score * 0.2)
            reasons.append("acronym_match")

        # 6. Edit distance (Levenshtein-like)
        edit_score = self._edit_distance_similarity(command_lower, option_lower)
        scores.append(edit_score * 0.2)

        # 7. Character overlap
        char_score = self._character_overlap(command_lower, option_lower)
        scores.append(char_score * 0.15)

        # 8. Frequency/history boost
        if option in self.command_frequency:
            freq = self.command_frequency[option]
            freq_score = min(1.0, freq / 100.0)  # Cap at 100 uses
            scores.append(freq_score * 0.1)
            if freq > 5:
                reasons.append("frequently_used")

        # 9. Context-based scoring
        if context:
            context_score = self._context_similarity(command, option, context)
            if context_score > 0:
                scores.append(context_score * 0.15)
                reasons.append("context_match")

        # Calculate final score
        final_score = sum(scores)
        final_score = min(1.0, final_score)  # Cap at 1.0

        reason = ", ".join(reasons) if reasons else "low_similarity"
        return (final_score, reason)

    def _acronym_similarity(self, command: str, option: str) -> float:
        """
        Calculate similarity based on acronym matching.

        For example: 'gc' matches 'git commit'
        """
        # Get first letters of words in option
        option_words = option.split()
        if len(option_words) < 2:
            return 0.0

        acronym = "".join(word[0] for word in option_words if word)

        # Check if command matches the acronym
        if command == acronym:
            return 1.0
        elif command in acronym:
            return len(command) / len(acronym)
        elif acronym.startswith(command):
            return len(command) / len(acronym) * 0.8

        return 0.0

    def _edit_distance_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity based on edit distance (Levenshtein).

        Returns a score between 0 and 1, where 1 is identical.
        """
        if not s1 or not s2:
            return 0.0

        # Simple edit distance implementation
        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        current_row = range(len1 + 1)
        for i in range(1, len2 + 1):
            previous_row = current_row
            current_row = [i] + [0] * len1
            for j in range(1, len1 + 1):
                add = previous_row[j] + 1
                delete = current_row[j - 1] + 1
                change = previous_row[j - 1]
                if s1[j - 1] != s2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        distance = current_row[len1]
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def _character_overlap(self, s1: str, s2: str) -> float:
        """Calculate character overlap between two strings."""
        if not s1 or not s2:
            return 0.0

        chars1 = set(s1)
        chars2 = set(s2)
        overlap = len(chars1 & chars2)
        total = len(chars1 | chars2)

        return overlap / total if total > 0 else 0.0

    def _context_similarity(
        self, command: str, option: str, context: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity based on context information.

        Context can include:
        - 'category': Expected command category
        - 'recent_commands': Recently used commands
        - 'environment': Current environment/mode
        """
        score = 0.0

        # Category matching
        if "category" in context and "category" in context.get(
            "option_metadata", {}
        ).get(option, {}):
            if context["category"] == context["option_metadata"][option]["category"]:
                score += 0.3

        # Recent commands boost
        if "recent_commands" in context and option in context["recent_commands"]:
            # More recent = higher score
            recency_index = context["recent_commands"].index(option)
            recency_score = 1.0 - (recency_index / len(context["recent_commands"]))
            score += recency_score * 0.2

        # Environment matching
        if "environment" in context and "environments" in context.get(
            "option_metadata", {}
        ).get(option, {}):
            if (
                context["environment"]
                in context["option_metadata"][option]["environments"]
            ):
                score += 0.2

        return min(1.0, score)

    def record_usage(self, command: str) -> None:
        """
        Record command usage for frequency-based scoring.

        Args:
            command: The command that was used
        """
        self.command_history.append(command)
        self.command_frequency[command] = self.command_frequency.get(command, 0) + 1

        # Keep history limited to last 1000 commands
        if len(self.command_history) > 1000:
            # Remove oldest commands
            removed = self.command_history[:100]
            self.command_history = self.command_history[100:]

            # Decrease frequency for removed commands
            for cmd in removed:
                if cmd in self.command_frequency:
                    self.command_frequency[cmd] = max(
                        0, self.command_frequency[cmd] - 1
                    )

    def get_suggestions_with_scores(
        self, command: str, options: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[CommandCandidate]:
        """
        Get command suggestions with detailed scoring information.

        This is useful for displaying to users why certain suggestions were made.

        Args:
            command: The ambiguous command
            options: List of possible commands
            context: Optional context information

        Returns:
            List of CommandCandidate objects with scores and reasons
        """
        return self.get_candidates(command, options, context)

    def explain_match(
        self, command: str, option: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Explain why a particular option matches a command.

        Args:
            command: The input command
            option: The option to explain
            context: Optional context information

        Returns:
            Human-readable explanation of the match
        """
        context = context or {}
        score, reason = self._calculate_similarity(command, option, context)

        explanation_parts = [f"Match score: {score:.2%}", f"Reasons: {reason}"]

        # Add specific details
        command_lower = command.lower()
        option_lower = option.lower()

        if command_lower == option_lower:
            explanation_parts.append("Exact match")
        elif option_lower.startswith(command_lower):
            explanation_parts.append(f"'{option}' starts with '{command}'")
        elif command_lower in option_lower:
            explanation_parts.append(f"'{command}' is contained in '{option}'")

        # Check acronym
        option_words = option_lower.split()
        if len(option_words) > 1:
            acronym = "".join(word[0] for word in option_words if word)
            if command_lower == acronym:
                explanation_parts.append(f"'{command}' matches acronym of '{option}'")

        return ". ".join(explanation_parts)

    def set_threshold(self, threshold: float) -> None:
        """
        Set the minimum similarity threshold.

        Args:
            threshold: Minimum score (0.0-1.0) to consider a match
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        logger.info(f"Similarity threshold set to {threshold}")

    def set_max_suggestions(self, max_suggestions: int) -> None:
        """
        Set the maximum number of suggestions to return.

        Args:
            max_suggestions: Maximum number of suggestions
        """
        if max_suggestions < 1:
            raise ValueError("Max suggestions must be at least 1")
        self.max_suggestions = max_suggestions
        logger.info(f"Max suggestions set to {max_suggestions}")

    def clear_history(self) -> None:
        """Clear command history and frequency data."""
        self.command_history.clear()
        self.command_frequency.clear()
        logger.info("Command history and frequency data cleared")

    def get_most_used_commands(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most frequently used commands.

        Args:
            n: Number of top commands to return

        Returns:
            List of (command, frequency) tuples sorted by frequency
        """
        sorted_commands = sorted(
            self.command_frequency.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_commands[:n]

    def fuzzy_search(
        self,
        query: str,
        candidates: List[str],
        key_func: Optional[Callable[[str], str]] = None,
    ) -> List[str]:
        """
        Perform fuzzy search on a list of candidates.

        Args:
            query: Search query
            candidates: List of candidates to search
            key_func: Optional function to extract searchable text from candidate

        Returns:
            List of matching candidates sorted by relevance
        """
        if not query or not candidates:
            return candidates

        key_func = key_func or (lambda x: x)

        scored_candidates = []
        for candidate in candidates:
            searchable_text = key_func(candidate)
            score, _ = self._calculate_similarity(query, searchable_text, {})

            if score >= self.threshold:
                scored_candidates.append((candidate, score))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return [candidate for candidate, score in scored_candidates]

    def interactive_disambiguate(
        self,
        command: str,
        options: List[str],
        context: Optional[Dict[str, Any]] = None,
        auto_select_threshold: float = 0.9,
    ) -> Optional[str]:
        """
        Interactively disambiguate a command.

        If the best match has a score above auto_select_threshold, it's automatically selected.
        Otherwise, returns None to indicate user input is needed.

        Args:
            command: The ambiguous command
            options: List of possible options
            context: Optional context information
            auto_select_threshold: Score threshold for automatic selection

        Returns:
            The selected command if auto-selected, None otherwise
        """
        candidates = self.get_candidates(command, options, context)

        if not candidates:
            return None

        best_candidate = candidates[0]

        if best_candidate.score >= auto_select_threshold:
            logger.info(
                f"Auto-selected '{best_candidate.command}' with score {best_candidate.score:.2f}"
            )
            return best_candidate.command

        logger.info(
            f"Best candidate '{best_candidate.command}' has score {best_candidate.score:.2f}, "
            f"below auto-select threshold {auto_select_threshold}"
        )
        return None
