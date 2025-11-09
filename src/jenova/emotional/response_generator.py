# The JENOVA Cognitive Architecture - Empathetic Response Generator
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 27: Empathetic Response Generator - Generates emotionally-aware responses.

Adjusts response tone, selects appropriate language, and provides empathetic
communication based on detected emotional state.
"""

from typing import Dict, List, Optional, Any
from enum import Enum

from jenova.emotional.emotion_detector import EmotionType
from jenova.emotional.state_manager import EmotionalStateManager


class ResponseTone(str, Enum):
    """Response tone options."""

    SUPPORTIVE = "supportive"
    ENCOURAGING = "encouraging"
    CALMING = "calming"
    CELEBRATORY = "celebratory"
    VALIDATING = "validating"
    NEUTRAL = "neutral"
    GENTLE = "gentle"


class EmpatheticResponseGenerator:
    """
    Generate emotionally-aware responses.

    Provides tone selection, empathetic language, and response strategies
    based on user's emotional state.

    Example:
        >>> generator = EmpatheticResponseGenerator()
        >>> tone = generator.select_tone(EmotionType.SADNESS, intensity=0.8)
        >>> print(tone)  # ResponseTone.SUPPORTIVE
        >>> prefix = generator.get_empathetic_prefix(EmotionType.SADNESS)
        >>> print(prefix)  # "I understand you're feeling down."
    """

    def __init__(self):
        """Initialize empathetic response generator."""
        # Map emotions to appropriate tones
        self.emotion_to_tone = {
            EmotionType.JOY: ResponseTone.CELEBRATORY,
            EmotionType.SADNESS: ResponseTone.SUPPORTIVE,
            EmotionType.ANGER: ResponseTone.CALMING,
            EmotionType.FEAR: ResponseTone.CALMING,
            EmotionType.SURPRISE: ResponseTone.ENCOURAGING,
            EmotionType.DISGUST: ResponseTone.VALIDATING,
            EmotionType.TRUST: ResponseTone.ENCOURAGING,
            EmotionType.ANTICIPATION: ResponseTone.ENCOURAGING,
            EmotionType.NEUTRAL: ResponseTone.NEUTRAL,
        }

        # Empathetic prefixes for each emotion
        self.empathetic_prefixes = {
            EmotionType.JOY: [
                "I'm glad you're feeling positive!",
                "It's wonderful to hear that!",
                "That's great news!",
                "I can sense your enthusiasm!",
            ],
            EmotionType.SADNESS: [
                "I understand you're feeling down.",
                "I hear that you're going through a difficult time.",
                "It sounds like you're struggling right now.",
                "I'm here to support you through this.",
            ],
            EmotionType.ANGER: [
                "I understand your frustration.",
                "It sounds like this situation is really bothering you.",
                "I can see why you'd feel that way.",
                "Your feelings are valid.",
            ],
            EmotionType.FEAR: [
                "I understand this feels overwhelming.",
                "It's okay to feel anxious about this.",
                "Your concerns are valid.",
                "Let's work through this together.",
            ],
            EmotionType.SURPRISE: [
                "That is surprising!",
                "I can see that caught you off guard.",
                "That's quite unexpected!",
            ],
            EmotionType.DISGUST: [
                "I understand your discomfort.",
                "That's a valid reaction.",
                "I can see why that bothers you.",
            ],
            EmotionType.TRUST: [
                "I appreciate your confidence.",
                "I'm glad we can work together on this.",
            ],
            EmotionType.ANTICIPATION: [
                "I can sense your excitement about this!",
                "It sounds like you're looking forward to this.",
            ],
            EmotionType.NEUTRAL: [],
        }

        # Response strategies for different emotions
        self.response_strategies = {
            EmotionType.JOY: [
                "Amplify positive emotions",
                "Share in the excitement",
                "Encourage continuation of positive state",
            ],
            EmotionType.SADNESS: [
                "Validate feelings",
                "Offer support and understanding",
                "Provide gentle encouragement",
                "Avoid toxic positivity",
            ],
            EmotionType.ANGER: [
                "Acknowledge feelings without judgment",
                "Use calm, measured language",
                "Offer constructive perspective",
                "Avoid dismissive language",
            ],
            EmotionType.FEAR: [
                "Provide reassurance",
                "Break down concerns into manageable parts",
                "Offer concrete steps forward",
                "Maintain calm, confident tone",
            ],
            EmotionType.SURPRISE: [
                "Acknowledge the unexpected",
                "Help process new information",
                "Provide context if helpful",
            ],
            EmotionType.DISGUST: [
                "Validate reaction",
                "Maintain professional distance",
                "Focus on solutions",
            ],
            EmotionType.TRUST: [
                "Honor confidence",
                "Maintain reliability",
                "Provide accurate information",
            ],
            EmotionType.ANTICIPATION: [
                "Match enthusiasm appropriately",
                "Help with planning or preparation",
                "Manage expectations realistically",
            ],
            EmotionType.NEUTRAL: [
                "Maintain informative, helpful tone",
                "Focus on task at hand",
            ],
        }

    def select_tone(
        self, emotion: EmotionType, intensity: float
    ) -> ResponseTone:
        """
        Select appropriate response tone.

        Args:
            emotion: Detected emotion
            intensity: Emotion intensity (0.0-1.0)

        Returns:
            Appropriate response tone

        Example:
            >>> tone = generator.select_tone(EmotionType.ANGER, 0.9)
            >>> print(tone)  # ResponseTone.CALMING
        """
        base_tone = self.emotion_to_tone.get(emotion, ResponseTone.NEUTRAL)

        # Adjust for intensity
        if intensity < 0.3:
            # Low intensity - can use neutral tone
            return ResponseTone.NEUTRAL

        return base_tone

    def get_empathetic_prefix(
        self,
        emotion: EmotionType,
        intensity: float = 0.5,
        index: int = 0
    ) -> str:
        """
        Get empathetic prefix for response.

        Args:
            emotion: Detected emotion
            intensity: Emotion intensity
            index: Which prefix variant to use

        Returns:
            Empathetic prefix text

        Example:
            >>> prefix = generator.get_empathetic_prefix(EmotionType.FEAR)
            >>> print(prefix)  # "I understand this feels overwhelming."
        """
        prefixes = self.empathetic_prefixes.get(emotion, [])

        if not prefixes:
            return ""

        # Select prefix based on index
        selected = prefixes[index % len(prefixes)]

        # Skip prefix for low intensity
        if intensity < 0.4:
            return ""

        return selected

    def adjust_response_parameters(
        self, emotion: EmotionType, intensity: float
    ) -> Dict[str, Any]:
        """
        Get LLM parameter adjustments for emotion.

        Args:
            emotion: Detected emotion
            intensity: Emotion intensity

        Returns:
            Dict with suggested parameter adjustments

        Example:
            >>> params = generator.adjust_response_parameters(
            ...     EmotionType.ANGER, 0.8
            ... )
            >>> print(params["temperature"])  # 0.4 (lower for calming)
        """
        # Base parameters
        params = {
            "temperature": 0.7,
            "max_tokens": 512,
            "style_hints": [],
        }

        # Emotion-specific adjustments
        if emotion == EmotionType.JOY:
            params["temperature"] = 0.8  # More creative for celebration
            params["style_hints"] = ["enthusiastic", "positive", "encouraging"]

        elif emotion == EmotionType.SADNESS:
            params["temperature"] = 0.5  # More controlled for support
            params["style_hints"] = ["gentle", "supportive", "understanding"]

        elif emotion == EmotionType.ANGER:
            params["temperature"] = 0.4  # Very controlled for calming
            params["style_hints"] = ["calm", "measured", "respectful"]

        elif emotion == EmotionType.FEAR:
            params["temperature"] = 0.5  # Controlled for reassurance
            params["style_hints"] = ["reassuring", "calm", "confident"]

        elif emotion == EmotionType.SURPRISE:
            params["temperature"] = 0.7  # Balanced
            params["style_hints"] = ["explanatory", "contextual"]

        elif emotion == EmotionType.DISGUST:
            params["temperature"] = 0.6  # Slightly controlled
            params["style_hints"] = ["professional", "solution-focused"]

        elif emotion == EmotionType.TRUST:
            params["temperature"] = 0.6  # Reliable consistency
            params["style_hints"] = ["reliable", "accurate", "thorough"]

        elif emotion == EmotionType.ANTICIPATION:
            params["temperature"] = 0.8  # Creative for excitement
            params["style_hints"] = ["encouraging", "forward-looking"]

        # Intensity adjustments
        if intensity > 0.8:
            # High intensity - be more careful
            params["temperature"] *= 0.9
            params["max_tokens"] = min(params["max_tokens"], 400)

        return params

    def get_response_strategy(self, emotion: EmotionType) -> List[str]:
        """
        Get response strategies for emotion.

        Args:
            emotion: Detected emotion

        Returns:
            List of strategy guidelines

        Example:
            >>> strategies = generator.get_response_strategy(EmotionType.SADNESS)
            >>> for strategy in strategies:
            ...     print(f"- {strategy}")
        """
        return self.response_strategies.get(emotion, [])

    def build_system_prompt_addition(
        self,
        emotion: EmotionType,
        intensity: float,
        emotional_context: Optional[str] = None
    ) -> str:
        """
        Build system prompt addition for emotional context.

        Args:
            emotion: Detected emotion
            intensity: Emotion intensity
            emotional_context: Optional context from state manager

        Returns:
            System prompt addition text

        Example:
            >>> addition = generator.build_system_prompt_addition(
            ...     EmotionType.SADNESS,
            ...     0.8,
            ...     "User is feeling down after a difficult day."
            ... )
        """
        tone = self.select_tone(emotion, intensity)
        strategies = self.get_response_strategy(emotion)

        parts = [
            f"\nEMOTIONAL CONTEXT:",
            f"- User emotion: {emotion.value} (intensity: {intensity:.2f})",
            f"- Response tone: {tone.value}",
        ]

        if emotional_context:
            parts.append(f"- Context: {emotional_context}")

        parts.append("\nRESPONSE GUIDELINES:")
        for strategy in strategies:
            parts.append(f"- {strategy}")

        return "\n".join(parts)

    def generate_empathetic_wrapper(
        self,
        base_response: str,
        emotion: EmotionType,
        intensity: float,
        use_prefix: bool = True
    ) -> str:
        """
        Wrap base response with empathetic framing.

        Args:
            base_response: Original response content
            emotion: Detected emotion
            intensity: Emotion intensity
            use_prefix: Whether to add empathetic prefix

        Returns:
            Wrapped response

        Example:
            >>> wrapped = generator.generate_empathetic_wrapper(
            ...     "Here's the information you requested.",
            ...     EmotionType.FEAR,
            ...     0.7
            ... )
        """
        if not use_prefix or intensity < 0.4:
            return base_response

        prefix = self.get_empathetic_prefix(emotion, intensity)

        if not prefix:
            return base_response

        return f"{prefix} {base_response}"
