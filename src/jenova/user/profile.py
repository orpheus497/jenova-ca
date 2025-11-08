# The JENOVA Cognitive Architecture - User Profiling System
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 10: User recognition and personalization system.

Tracks user preferences, interaction patterns, and adapts behavior accordingly.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import Counter, defaultdict


@dataclass
class UserPreferences:
    """User preferences and settings."""

    response_style: str = "balanced"  # concise, balanced, detailed
    expertise_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    preferred_topics: List[str] = None
    communication_style: str = "friendly"  # formal, friendly, casual, technical
    learning_mode: bool = True
    proactive_suggestions: bool = True

    def __post_init__(self):
        if self.preferred_topics is None:
            self.preferred_topics = []


@dataclass
class InteractionStats:
    """Statistics about user interactions."""

    total_interactions: int = 0
    questions_asked: int = 0
    commands_used: int = 0
    topics_discussed: Dict[str, int] = None
    common_patterns: List[str] = None
    average_session_length: float = 0.0
    last_interaction: Optional[str] = None

    def __post_init__(self):
        if self.topics_discussed is None:
            self.topics_discussed = {}
        if self.common_patterns is None:
            self.common_patterns = []


class UserProfile:
    """
    User profile with learning and adaptation capabilities.

    Tracks:
    - Interaction history
    - Preferences and patterns
    - Topic interests
    - Expertise indicators
    - Communication style
    """

    def __init__(self, username: str, profile_dir: str):
        self.username = username
        self.profile_dir = profile_dir
        self.profile_file = os.path.join(profile_dir, "profile.json")

        # Core profile data
        self.preferences = UserPreferences()
        self.stats = InteractionStats()

        # Learning data
        self.vocabulary: Set[str] = set()  # User's vocabulary
        self.topic_history: List[str] = []  # Recent topics
        self.interaction_times: List[float] = []  # Session durations
        self.corrections: List[Dict] = []  # User corrections to learn from

        # Adaptation data
        self.successful_suggestions: int = 0
        self.total_suggestions: int = 0
        self.preferred_commands: Counter = Counter()

        # Load existing profile if available
        self.load()

    def load(self):
        """Load profile from disk."""
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, "r") as f:
                    data = json.load(f)

                # Load preferences
                if "preferences" in data:
                    self.preferences = UserPreferences(**data["preferences"])

                # Load stats
                if "stats" in data:
                    self.stats = InteractionStats(**data["stats"])

                # Load learning data
                self.vocabulary = set(data.get("vocabulary", []))
                self.topic_history = data.get("topic_history", [])
                self.interaction_times = data.get("interaction_times", [])
                self.corrections = data.get("corrections", [])

                # Load adaptation data
                self.successful_suggestions = data.get("successful_suggestions", 0)
                self.total_suggestions = data.get("total_suggestions", 0)
                self.preferred_commands = Counter(data.get("preferred_commands", {}))

            except Exception as e:
                print(f"Error loading profile: {e}")

    def save(self):
        """Save profile to disk."""
        os.makedirs(self.profile_dir, exist_ok=True)

        data = {
            "username": self.username,
            "preferences": asdict(self.preferences),
            "stats": asdict(self.stats),
            "vocabulary": list(self.vocabulary),
            "topic_history": self.topic_history[-100:],  # Keep last 100
            "interaction_times": self.interaction_times[-100:],
            "corrections": self.corrections[-50:],  # Keep last 50
            "successful_suggestions": self.successful_suggestions,
            "total_suggestions": self.total_suggestions,
            "preferred_commands": dict(self.preferred_commands),
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.profile_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_interaction(self, user_input: str, interaction_type: str = "query"):
        """Record a user interaction."""
        self.stats.total_interactions += 1
        self.stats.last_interaction = datetime.now().isoformat()

        if interaction_type == "question":
            self.stats.questions_asked += 1
        elif interaction_type == "command":
            self.stats.commands_used += 1

        # Extract vocabulary
        words = set(user_input.lower().split())
        self.vocabulary.update(words)

        self.save()

    def record_topic(self, topic: str):
        """Record discussion of a topic."""
        self.topic_history.append(topic)

        if topic not in self.stats.topics_discussed:
            self.stats.topics_discussed[topic] = 0
        self.stats.topics_discussed[topic] += 1

        # Update preferred topics
        if topic not in self.preferences.preferred_topics:
            # Add if discussed more than 3 times
            if self.stats.topics_discussed[topic] > 3:
                self.preferences.preferred_topics.append(topic)

        self.save()

    def record_command_use(self, command: str):
        """Record use of a command."""
        self.preferred_commands[command] += 1
        self.save()

    def record_correction(self, original: str, corrected: str, context: str):
        """Record a user correction to learn from."""
        self.corrections.append(
            {
                "original": original,
                "corrected": corrected,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.save()

    def record_suggestion_feedback(self, accepted: bool):
        """Record feedback on a suggestion."""
        self.total_suggestions += 1
        if accepted:
            self.successful_suggestions += 1
        self.save()

    def get_suggestion_success_rate(self) -> float:
        """Get the success rate of suggestions."""
        if self.total_suggestions == 0:
            return 0.0
        return self.successful_suggestions / self.total_suggestions

    def get_top_topics(self, limit: int = 5) -> List[tuple]:
        """Get most discussed topics."""
        return Counter(self.stats.topics_discussed).most_common(limit)

    def get_expertise_indicators(self) -> Dict[str, Any]:
        """Get indicators of user expertise."""
        return {
            "vocabulary_size": len(self.vocabulary),
            "total_interactions": self.stats.total_interactions,
            "topics_mastered": len(
                [t for t, c in self.stats.topics_discussed.items() if c > 10]
            ),
            "command_proficiency": len(self.preferred_commands),
            "learning_progress": self.get_suggestion_success_rate(),
        }

    def adapt_response_style(self):
        """Adapt response style based on user patterns."""
        # If user uses many technical terms, increase expertise level
        technical_words = {
            "algorithm",
            "optimization",
            "architecture",
            "implementation",
        }
        tech_usage = sum(1 for word in technical_words if word in self.vocabulary)

        if tech_usage > 3 and self.preferences.expertise_level == "intermediate":
            self.preferences.expertise_level = "advanced"
            self.save()

        # If user prefers short responses (avg input < 20 words), use concise style
        # This would be calculated from actual interaction data


class UserProfileManager:
    """Manages user profiles across the system."""

    def __init__(self, config: dict, file_logger):
        self.config = config
        self.file_logger = file_logger
        self.profiles: Dict[str, UserProfile] = {}

        user_data_root = config.get("user_data_root", "~/.jenova-ai")
        self.profiles_dir = os.path.join(user_data_root, "profiles")
        os.makedirs(self.profiles_dir, exist_ok=True)

    def get_profile(self, username: str) -> UserProfile:
        """Get or create a user profile."""
        if username not in self.profiles:
            profile_dir = os.path.join(self.profiles_dir, username)
            self.profiles[username] = UserProfile(username, profile_dir)
            self.file_logger.log_info(f"Loaded profile for user: {username}")

        return self.profiles[username]

    def save_all(self):
        """Save all loaded profiles."""
        for profile in self.profiles.values():
            profile.save()
