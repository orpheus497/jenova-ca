# The JENOVA Cognitive Architecture - Emotional Intelligence Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 27: Emotional Intelligence Layer - Emotion detection and empathetic responses.

Provides emotion detection, emotional state tracking, and empathetic
response generation for emotionally-aware interactions.

Components:
    - EmotionDetector: Keyword-based emotion detection with intensity scoring
    - EmotionalStateManager: Tracks emotional state over conversation
    - EmpatheticResponseGenerator: Generates emotionally-aware responses

Example:
    >>> from jenova.emotional import (
    ...     EmotionDetector,
    ...     EmotionalStateManager,
    ...     EmpatheticResponseGenerator
    ... )
    >>>
    >>> # Detect emotion
    >>> detector = EmotionDetector()
    >>> detection = detector.detect("I'm really happy about this!")
    >>> print(detection.primary_emotion, detection.intensity)
    >>>
    >>> # Track emotional state
    >>> manager = EmotionalStateManager()
    >>> manager.update(detection, message="I'm really happy!", turn=1)
    >>> state = manager.get_current_state()
    >>>
    >>> # Generate empathetic response
    >>> generator = EmpatheticResponseGenerator()
    >>> tone = generator.select_tone(detection.primary_emotion, detection.intensity)
    >>> prefix = generator.get_empathetic_prefix(detection.primary_emotion)
"""

from jenova.emotional.emotion_detector import (
    EmotionType,
    EmotionDetection,
    EmotionDetector,
)
from jenova.emotional.state_manager import (
    EmotionalMoment,
    EmotionalTrend,
    EmotionalStateManager,
)
from jenova.emotional.response_generator import (
    ResponseTone,
    EmpatheticResponseGenerator,
)

__all__ = [
    # Emotion Detection
    "EmotionType",
    "EmotionDetection",
    "EmotionDetector",
    # State Management
    "EmotionalMoment",
    "EmotionalTrend",
    "EmotionalStateManager",
    # Response Generation
    "ResponseTone",
    "EmpatheticResponseGenerator",
]
