# The JENOVA Cognitive Architecture - Emotional State Manager
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 27: Emotional State Manager - Tracks emotional state over conversation.

Maintains history of detected emotions, tracks emotional trends, and provides
context for empathetic response generation.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from jenova.emotional.emotion_detector import EmotionType, EmotionDetection


@dataclass
class EmotionalMoment:
    """
    Snapshot of emotional state at a point in time.

    Attributes:
        timestamp: Unix timestamp
        detection: Emotion detection result
        message: User message that triggered detection
        turn_number: Conversation turn number
    """

    timestamp: float
    detection: EmotionDetection
    message: str
    turn_number: int


@dataclass
class EmotionalTrend:
    """
    Detected emotional trend over time.

    Attributes:
        emotion: Emotion type trending
        direction: 'increasing', 'decreasing', or 'stable'
        strength: Trend strength (0.0-1.0)
        duration_turns: Number of turns trend spans
    """

    emotion: EmotionType
    direction: str  # increasing, decreasing, stable
    strength: float
    duration_turns: int


class EmotionalStateManager:
    """
    Manage emotional state tracking across conversation.

    Tracks emotion history, detects trends, and provides emotional context
    for response generation.

    Example:
        >>> manager = EmotionalStateManager(history_length=10)
        >>> manager.update(detection, message="I'm really happy!", turn=1)
        >>> state = manager.get_current_state()
        >>> print(state["primary_emotion"])  # EmotionType.JOY
    """

    def __init__(self, history_length: int = 20):
        """
        Initialize emotional state manager.

        Args:
            history_length: Maximum emotion history to maintain
        """
        self.history_length = history_length
        self.history: deque[EmotionalMoment] = deque(maxlen=history_length)

        # Current emotional state
        self.current_emotion: EmotionType = EmotionType.NEUTRAL
        self.current_intensity: float = 0.0

        # Emotional baseline (average over conversation)
        self.baseline_emotions: Dict[EmotionType, float] = {}

        # Turn counter
        self.turn_counter = 0

    def update(
        self, detection: EmotionDetection, message: str, turn: Optional[int] = None
    ) -> None:
        """
        Update emotional state with detection.

        Args:
            detection: Emotion detection result
            message: User message
            turn: Turn number (auto-incremented if None)

        Example:
            >>> manager.update(detection, "I'm excited!", turn=5)
        """
        if turn is None:
            self.turn_counter += 1
            turn = self.turn_counter
        else:
            self.turn_counter = turn

        # Create moment
        moment = EmotionalMoment(
            timestamp=time.time(),
            detection=detection,
            message=message,
            turn_number=turn,
        )

        # Add to history
        self.history.append(moment)

        # Update current state
        self.current_emotion = detection.primary_emotion
        self.current_intensity = detection.intensity

        # Update baseline
        self._update_baseline()

    def get_current_state(self) -> Dict[str, any]:
        """
        Get current emotional state.

        Returns:
            Dict with current emotion, intensity, baseline, trends

        Example:
            >>> state = manager.get_current_state()
            >>> print(state["primary_emotion"], state["intensity"])
        """
        return {
            "primary_emotion": self.current_emotion,
            "intensity": self.current_intensity,
            "baseline": self.baseline_emotions.copy(),
            "history_length": len(self.history),
            "recent_trend": self.detect_recent_trend(),
        }

    def detect_recent_trend(self, window_turns: int = 5) -> Optional[EmotionalTrend]:
        """
        Detect emotional trend in recent history.

        Args:
            window_turns: Number of recent turns to analyze

        Returns:
            EmotionalTrend or None if no clear trend

        Example:
            >>> trend = manager.detect_recent_trend(window=5)
            >>> if trend:
            ...     print(trend.emotion, trend.direction)
        """
        if len(self.history) < 2:
            return None

        # Get recent moments
        recent = list(self.history)[-window_turns:]
        if len(recent) < 2:
            return None

        # Track emotion intensities over time
        emotion_series: Dict[EmotionType, List[float]] = {}

        for moment in recent:
            for emotion, intensity in moment.detection.all_emotions.items():
                if emotion not in emotion_series:
                    emotion_series[emotion] = []
                emotion_series[emotion].append(intensity)

        # Find strongest trend
        strongest_trend: Optional[EmotionalTrend] = None
        max_strength = 0.0

        for emotion, intensities in emotion_series.items():
            if len(intensities) < 2:
                continue

            # Calculate linear trend
            trend_direction, trend_strength = self._calculate_linear_trend(intensities)

            if trend_strength > max_strength:
                max_strength = trend_strength
                strongest_trend = EmotionalTrend(
                    emotion=emotion,
                    direction=trend_direction,
                    strength=trend_strength,
                    duration_turns=len(intensities),
                )

        return strongest_trend

    def _calculate_linear_trend(
        self, values: List[float]
    ) -> Tuple[str, float]:
        """
        Calculate linear trend for value series.

        Args:
            values: Time series of values

        Returns:
            (direction, strength) tuple
        """
        if len(values) < 2:
            return "stable", 0.0

        # Simple linear regression slope
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x2_sum = sum(i * i for i in range(n))

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)

        # Determine direction and strength
        if abs(slope) < 0.05:
            return "stable", abs(slope)
        elif slope > 0:
            return "increasing", min(1.0, abs(slope) * 2)
        else:
            return "decreasing", min(1.0, abs(slope) * 2)

    def _update_baseline(self) -> None:
        """Update emotional baseline from history."""
        if not self.history:
            return

        # Aggregate all emotion scores
        emotion_totals: Dict[EmotionType, float] = {}
        emotion_counts: Dict[EmotionType, int] = {}

        for moment in self.history:
            for emotion, score in moment.detection.all_emotions.items():
                if emotion not in emotion_totals:
                    emotion_totals[emotion] = 0.0
                    emotion_counts[emotion] = 0

                emotion_totals[emotion] += score
                emotion_counts[emotion] += 1

        # Calculate averages
        self.baseline_emotions = {
            emotion: total / emotion_counts[emotion]
            for emotion, total in emotion_totals.items()
        }

    def get_emotional_context(self, recent_turns: int = 3) -> str:
        """
        Get textual summary of emotional context.

        Args:
            recent_turns: Number of recent turns to summarize

        Returns:
            Emotional context summary

        Example:
            >>> context = manager.get_emotional_context(recent_turns=3)
            >>> print(context)
            "User is currently joyful (high intensity). Recent trend: increasing happiness."
        """
        if not self.history:
            return "No emotional context available."

        # Current state
        intensity_desc = self._intensity_description(self.current_intensity)
        context_parts = [
            f"User is currently {self.current_emotion.value} ({intensity_desc} intensity)."
        ]

        # Recent trend
        trend = self.detect_recent_trend(window_turns=recent_turns)
        if trend:
            trend_desc = f"{trend.direction} {trend.emotion.value}"
            context_parts.append(f"Recent trend: {trend_desc}.")

        # Dominant baseline emotion
        if self.baseline_emotions:
            dominant = max(self.baseline_emotions.items(), key=lambda x: x[1])
            context_parts.append(
                f"Overall conversation tone: {dominant[0].value}."
            )

        return " ".join(context_parts)

    def _intensity_description(self, intensity: float) -> str:
        """
        Convert intensity to descriptive term.

        Args:
            intensity: Intensity value 0.0-1.0

        Returns:
            Description (low, medium, high, very high)
        """
        if intensity < 0.3:
            return "low"
        elif intensity < 0.6:
            return "medium"
        elif intensity < 0.85:
            return "high"
        else:
            return "very high"

    def get_history_summary(self, max_turns: int = 5) -> List[Dict[str, any]]:
        """
        Get summary of recent emotional history.

        Args:
            max_turns: Maximum turns to include

        Returns:
            List of emotion summaries

        Example:
            >>> history = manager.get_history_summary(max_turns=3)
            >>> for entry in history:
            ...     print(f"Turn {entry['turn']}: {entry['emotion']}")
        """
        recent = list(self.history)[-max_turns:]

        return [
            {
                "turn": moment.turn_number,
                "emotion": moment.detection.primary_emotion.value,
                "intensity": moment.detection.intensity,
                "message": moment.message[:50] + "..." if len(moment.message) > 50 else moment.message,
                "timestamp": moment.timestamp,
            }
            for moment in recent
        ]

    def reset(self) -> None:
        """
        Reset emotional state.

        Example:
            >>> manager.reset()
        """
        self.history.clear()
        self.current_emotion = EmotionType.NEUTRAL
        self.current_intensity = 0.0
        self.baseline_emotions.clear()
        self.turn_counter = 0

    def get_stats(self) -> Dict[str, any]:
        """
        Get emotional state statistics.

        Returns:
            Statistics dictionary

        Example:
            >>> stats = manager.get_stats()
            >>> print(stats["total_turns"])
        """
        emotion_distribution = {}
        for moment in self.history:
            emotion = moment.detection.primary_emotion.value
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1

        return {
            "total_turns": self.turn_counter,
            "history_size": len(self.history),
            "current_emotion": self.current_emotion.value,
            "current_intensity": self.current_intensity,
            "emotion_distribution": emotion_distribution,
            "baseline_emotions": {
                e.value: score for e, score in self.baseline_emotions.items()
            },
        }
