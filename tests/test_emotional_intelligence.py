# The JENOVA Cognitive Architecture - Emotional Intelligence Tests
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 27: Tests for Emotional Intelligence Layer.

Tests emotion detection, state management, and empathetic response generation
with comprehensive coverage.
"""

import pytest
import time

from jenova.emotional import (
    EmotionType,
    EmotionDetection,
    EmotionDetector,
    EmotionalMoment,
    EmotionalTrend,
    EmotionalStateManager,
    ResponseTone,
    EmpatheticResponseGenerator,
)


class TestEmotionDetector:
    """Test suite for EmotionDetector."""

    @pytest.fixture
    def detector(self):
        """Fixture providing EmotionDetector instance."""
        return EmotionDetector()

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert len(detector.emotion_keywords) > 0
        assert EmotionType.JOY in detector.emotion_keywords
        assert len(detector.intensifiers) > 0
        assert len(detector.negation_words) > 0

    def test_joy_detection(self, detector):
        """Test joy emotion detection."""
        texts = [
            "I'm so happy and excited!",
            "This is wonderful news!",
            "I love this, it's amazing!",
        ]

        for text in texts:
            result = detector.detect(text)
            assert result.primary_emotion == EmotionType.JOY
            assert result.intensity > 0.5

    def test_sadness_detection(self, detector):
        """Test sadness emotion detection."""
        texts = [
            "I'm feeling really sad today",
            "This is heartbreaking and depressing",
            "I'm so disappointed and hurt",
        ]

        for text in texts:
            result = detector.detect(text)
            assert result.primary_emotion == EmotionType.SADNESS
            assert result.intensity > 0.5

    def test_anger_detection(self, detector):
        """Test anger emotion detection."""
        texts = [
            "I'm so angry and frustrated",
            "This makes me furious and mad",
            "I hate this situation",
        ]

        for text in texts:
            result = detector.detect(text)
            assert result.primary_emotion == EmotionType.ANGER
            assert result.intensity > 0.5

    def test_fear_detection(self, detector):
        """Test fear emotion detection."""
        texts = [
            "I'm terrified and scared",
            "This is making me very anxious",
            "I'm worried and nervous about this",
        ]

        for text in texts:
            result = detector.detect(text)
            assert result.primary_emotion == EmotionType.FEAR
            assert result.intensity > 0.5

    def test_intensifier_boost(self, detector):
        """Test intensifiers boost emotion scores."""
        weak = detector.detect("I'm happy")
        strong = detector.detect("I'm extremely happy")

        assert strong.intensity > weak.intensity

    def test_diminisher_reduction(self, detector):
        """Test diminishers reduce emotion scores."""
        strong = detector.detect("I'm sad")
        weak = detector.detect("I'm slightly sad")

        assert weak.intensity < strong.intensity

    def test_negation_handling(self, detector):
        """Test negation word handling."""
        result = detector.detect("I'm not happy at all")
        # Negation should reduce joy or introduce sadness
        assert result.primary_emotion != EmotionType.JOY or result.intensity < 0.3

    def test_empty_text(self, detector):
        """Test empty text handling."""
        result = detector.detect("")
        assert result.primary_emotion == EmotionType.NEUTRAL
        assert result.intensity == 0.0

    def test_neutral_text(self, detector):
        """Test neutral text detection."""
        result = detector.detect("The weather is 72 degrees today")
        assert result.primary_emotion == EmotionType.NEUTRAL
        assert result.intensity == 0.0

    def test_multiple_emotions(self, detector):
        """Test detection of multiple emotions."""
        result = detector.detect("I'm happy but also a bit worried")
        assert len(result.all_emotions) >= 2
        assert EmotionType.JOY in result.all_emotions
        assert EmotionType.FEAR in result.all_emotions or EmotionType.ANTICIPATION in result.all_emotions

    def test_keywords_matched(self, detector):
        """Test keyword matching tracking."""
        result = detector.detect("I'm very happy and excited")
        assert len(result.keywords_matched) > 0
        assert "happy" in result.keywords_matched or "excited" in result.keywords_matched

    def test_confidence_scoring(self, detector):
        """Test confidence score calculation."""
        # Clear single emotion should have high confidence
        clear = detector.detect("I'm extremely happy and joyful!")
        assert clear.confidence > 0.7

        # Mixed emotions should have lower confidence
        mixed = detector.detect("I'm happy but also sad and confused")
        assert mixed.confidence < clear.confidence

    def test_get_emotion_summary(self, detector):
        """Test emotion summary generation."""
        result = detector.detect("I'm really excited!")
        summary = detector.get_emotion_summary(result)
        assert "joy" in summary.lower() or "anticipation" in summary.lower()
        assert "%" in summary


class TestEmotionalStateManager:
    """Test suite for EmotionalStateManager."""

    @pytest.fixture
    def manager(self):
        """Fixture providing EmotionalStateManager instance."""
        return EmotionalStateManager(history_length=10)

    @pytest.fixture
    def sample_detection(self):
        """Fixture providing sample emotion detection."""
        return EmotionDetection(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            all_emotions={EmotionType.JOY: 0.8},
            confidence=0.9,
            keywords_matched=["happy"],
        )

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.current_emotion == EmotionType.NEUTRAL
        assert manager.current_intensity == 0.0
        assert len(manager.history) == 0

    def test_update_state(self, manager, sample_detection):
        """Test state update."""
        manager.update(sample_detection, "I'm happy!", turn=1)

        assert manager.current_emotion == EmotionType.JOY
        assert manager.current_intensity == 0.8
        assert len(manager.history) == 1

    def test_history_tracking(self, manager, sample_detection):
        """Test emotion history tracking."""
        for i in range(5):
            manager.update(sample_detection, f"Message {i}", turn=i + 1)

        assert len(manager.history) == 5
        assert manager.turn_counter == 5

    def test_history_limit(self, manager, sample_detection):
        """Test history length limit."""
        # Manager has max 10 items
        for i in range(15):
            manager.update(sample_detection, f"Message {i}", turn=i + 1)

        assert len(manager.history) == 10  # Should be capped

    def test_get_current_state(self, manager, sample_detection):
        """Test getting current state."""
        manager.update(sample_detection, "Happy message", turn=1)

        state = manager.get_current_state()
        assert state["primary_emotion"] == EmotionType.JOY
        assert state["intensity"] == 0.8
        assert state["history_length"] == 1

    def test_baseline_calculation(self, manager):
        """Test emotional baseline calculation."""
        # Add multiple emotions
        joy_detection = EmotionDetection(
            EmotionType.JOY, 0.8, {EmotionType.JOY: 0.8}, 0.9, []
        )
        sad_detection = EmotionDetection(
            EmotionType.SADNESS, 0.6, {EmotionType.SADNESS: 0.6}, 0.9, []
        )

        manager.update(joy_detection, "Happy", turn=1)
        manager.update(joy_detection, "Still happy", turn=2)
        manager.update(sad_detection, "Now sad", turn=3)

        assert EmotionType.JOY in manager.baseline_emotions
        assert EmotionType.SADNESS in manager.baseline_emotions

    def test_detect_recent_trend(self, manager):
        """Test recent trend detection."""
        # Create increasing joy trend
        for i in range(5):
            intensity = 0.4 + (i * 0.1)
            detection = EmotionDetection(
                EmotionType.JOY,
                intensity,
                {EmotionType.JOY: intensity},
                0.9,
                [],
            )
            manager.update(detection, f"Message {i}", turn=i + 1)

        trend = manager.detect_recent_trend(window_turns=5)
        assert trend is not None
        assert trend.emotion == EmotionType.JOY
        assert trend.direction == "increasing"

    def test_get_emotional_context(self, manager, sample_detection):
        """Test emotional context generation."""
        manager.update(sample_detection, "Happy message", turn=1)

        context = manager.get_emotional_context()
        assert "joy" in context.lower()
        assert "intensity" in context.lower()

    def test_get_history_summary(self, manager, sample_detection):
        """Test history summary."""
        manager.update(sample_detection, "Message 1", turn=1)
        manager.update(sample_detection, "Message 2", turn=2)

        summary = manager.get_history_summary(max_turns=5)
        assert len(summary) == 2
        assert summary[0]["turn"] == 1
        assert "emotion" in summary[0]

    def test_reset(self, manager, sample_detection):
        """Test state reset."""
        manager.update(sample_detection, "Message", turn=1)
        manager.reset()

        assert len(manager.history) == 0
        assert manager.current_emotion == EmotionType.NEUTRAL
        assert manager.turn_counter == 0

    def test_get_stats(self, manager, sample_detection):
        """Test statistics retrieval."""
        manager.update(sample_detection, "Message", turn=1)

        stats = manager.get_stats()
        assert stats["total_turns"] == 1
        assert stats["history_size"] == 1
        assert stats["current_emotion"] == "joy"


class TestEmpatheticResponseGenerator:
    """Test suite for EmpatheticResponseGenerator."""

    @pytest.fixture
    def generator(self):
        """Fixture providing EmpatheticResponseGenerator instance."""
        return EmpatheticResponseGenerator()

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert len(generator.emotion_to_tone) > 0
        assert len(generator.empathetic_prefixes) > 0
        assert len(generator.response_strategies) > 0

    def test_select_tone_joy(self, generator):
        """Test tone selection for joy."""
        tone = generator.select_tone(EmotionType.JOY, intensity=0.8)
        assert tone == ResponseTone.CELEBRATORY

    def test_select_tone_sadness(self, generator):
        """Test tone selection for sadness."""
        tone = generator.select_tone(EmotionType.SADNESS, intensity=0.8)
        assert tone == ResponseTone.SUPPORTIVE

    def test_select_tone_anger(self, generator):
        """Test tone selection for anger."""
        tone = generator.select_tone(EmotionType.ANGER, intensity=0.8)
        assert tone == ResponseTone.CALMING

    def test_select_tone_fear(self, generator):
        """Test tone selection for fear."""
        tone = generator.select_tone(EmotionType.FEAR, intensity=0.8)
        assert tone == ResponseTone.CALMING

    def test_select_tone_low_intensity(self, generator):
        """Test tone selection for low intensity."""
        tone = generator.select_tone(EmotionType.JOY, intensity=0.2)
        assert tone == ResponseTone.NEUTRAL

    def test_get_empathetic_prefix(self, generator):
        """Test empathetic prefix generation."""
        prefix = generator.get_empathetic_prefix(EmotionType.JOY, intensity=0.8)
        assert len(prefix) > 0
        assert isinstance(prefix, str)

    def test_get_empathetic_prefix_low_intensity(self, generator):
        """Test empathetic prefix with low intensity."""
        prefix = generator.get_empathetic_prefix(EmotionType.JOY, intensity=0.2)
        assert prefix == ""  # Should skip for low intensity

    def test_adjust_response_parameters_joy(self, generator):
        """Test parameter adjustment for joy."""
        params = generator.adjust_response_parameters(EmotionType.JOY, 0.8)
        assert params["temperature"] > 0.7  # More creative
        assert "enthusiastic" in params["style_hints"]

    def test_adjust_response_parameters_anger(self, generator):
        """Test parameter adjustment for anger."""
        params = generator.adjust_response_parameters(EmotionType.ANGER, 0.8)
        assert params["temperature"] < 0.7  # More controlled
        assert "calm" in params["style_hints"]

    def test_adjust_response_parameters_high_intensity(self, generator):
        """Test parameter adjustment for high intensity."""
        low_intensity = generator.adjust_response_parameters(EmotionType.JOY, 0.5)
        high_intensity = generator.adjust_response_parameters(EmotionType.JOY, 0.9)

        # High intensity should be more controlled
        assert high_intensity["temperature"] <= low_intensity["temperature"]

    def test_get_response_strategy(self, generator):
        """Test response strategy retrieval."""
        strategies = generator.get_response_strategy(EmotionType.SADNESS)
        assert len(strategies) > 0
        assert isinstance(strategies, list)

    def test_build_system_prompt_addition(self, generator):
        """Test system prompt addition building."""
        addition = generator.build_system_prompt_addition(
            EmotionType.SADNESS,
            0.8,
            "User is feeling down"
        )
        assert "EMOTIONAL CONTEXT" in addition
        assert "sadness" in addition.lower()
        assert "RESPONSE GUIDELINES" in addition

    def test_generate_empathetic_wrapper(self, generator):
        """Test empathetic response wrapping."""
        base = "Here is the information."
        wrapped = generator.generate_empathetic_wrapper(
            base,
            EmotionType.SADNESS,
            0.8,
            use_prefix=True
        )
        assert len(wrapped) > len(base)
        assert base in wrapped

    def test_generate_empathetic_wrapper_no_prefix(self, generator):
        """Test empathetic wrapping without prefix."""
        base = "Here is the information."
        wrapped = generator.generate_empathetic_wrapper(
            base,
            EmotionType.SADNESS,
            0.8,
            use_prefix=False
        )
        assert wrapped == base


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_emotional_flow(self):
        """Test complete emotion detection to response workflow."""
        # Detect emotion
        detector = EmotionDetector()
        detection = detector.detect("I'm really sad and upset today")

        assert detection.primary_emotion == EmotionType.SADNESS
        assert detection.intensity > 0.5

        # Track state
        manager = EmotionalStateManager()
        manager.update(detection, "I'm really sad and upset today", turn=1)

        state = manager.get_current_state()
        assert state["primary_emotion"] == EmotionType.SADNESS

        # Generate response
        generator = EmpatheticResponseGenerator()
        tone = generator.select_tone(detection.primary_emotion, detection.intensity)
        assert tone == ResponseTone.SUPPORTIVE

        prefix = generator.get_empathetic_prefix(
            detection.primary_emotion, detection.intensity
        )
        assert len(prefix) > 0

    def test_emotional_trend_detection(self):
        """Test detecting emotional trends over conversation."""
        detector = EmotionDetector()
        manager = EmotionalStateManager()

        # Simulate increasing sadness
        messages = [
            ("I'm feeling a bit down", 0.4),
            ("Things are getting worse", 0.6),
            ("I'm really sad now", 0.8),
        ]

        for i, (msg, expected_intensity) in enumerate(messages):
            detection = detector.detect(msg)
            manager.update(detection, msg, turn=i + 1)

        # Check for increasing sadness trend
        trend = manager.detect_recent_trend(window_turns=3)
        assert trend is not None
        # Should detect sadness or related emotion
        assert trend.emotion in [EmotionType.SADNESS, EmotionType.FEAR]

    def test_multi_emotion_conversation(self):
        """Test handling conversation with multiple emotions."""
        detector = EmotionDetector()
        manager = EmotionalStateManager()
        generator = EmpatheticResponseGenerator()

        messages = [
            "I'm excited about the new project!",
            "But I'm also worried it might be too difficult",
            "Still, I'm optimistic and ready to try",
        ]

        for i, msg in enumerate(messages):
            detection = detector.detect(msg)
            manager.update(detection, msg, turn=i + 1)

            tone = generator.select_tone(
                detection.primary_emotion, detection.intensity
            )

            # Each message should get appropriate tone
            assert tone in list(ResponseTone)

        # Check history
        assert len(manager.history) == 3

        # Check that different emotions were detected
        emotions_detected = set(
            moment.detection.primary_emotion for moment in manager.history
        )
        assert len(emotions_detected) >= 2  # At least 2 different emotions
