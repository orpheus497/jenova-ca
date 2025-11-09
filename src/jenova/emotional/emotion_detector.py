# The JENOVA Cognitive Architecture - Emotion Detector
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 27: Emotion Detector - Detects emotional content in user input.

Uses keyword-based analysis with intensity scoring to identify emotions
from text input. Production systems could use ML classifiers for better accuracy.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EmotionType(str, Enum):
    """Primary emotion types based on Plutchik's wheel of emotions."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


@dataclass
class EmotionDetection:
    """
    Result of emotion detection.

    Attributes:
        primary_emotion: Dominant emotion detected
        intensity: Emotion strength (0.0-1.0)
        all_emotions: All detected emotions with scores
        confidence: Detection confidence (0.0-1.0)
        keywords_matched: Keywords that triggered detection
    """

    primary_emotion: EmotionType
    intensity: float
    all_emotions: Dict[EmotionType, float]
    confidence: float
    keywords_matched: List[str]


class EmotionDetector:
    """
    Detect emotions from text using keyword-based analysis.

    Uses curated keyword lists for each emotion type with weighted scoring.
    Accounts for negation, intensifiers, and diminishers.

    Example:
        >>> detector = EmotionDetector()
        >>> result = detector.detect("I'm so happy and excited!")
        >>> print(result.primary_emotion)  # EmotionType.JOY
        >>> print(result.intensity)  # 0.85
    """

    def __init__(self):
        """Initialize emotion detector with keyword patterns."""
        # Emotion keywords with intensity weights
        self.emotion_keywords = {
            EmotionType.JOY: {
                "happy": 0.8,
                "joy": 0.9,
                "excited": 0.85,
                "wonderful": 0.8,
                "great": 0.7,
                "fantastic": 0.85,
                "amazing": 0.85,
                "excellent": 0.8,
                "love": 0.9,
                "pleased": 0.75,
                "delighted": 0.85,
                "thrilled": 0.9,
                "glad": 0.75,
                "cheerful": 0.8,
                "joyful": 0.9,
                "ecstatic": 0.95,
                "elated": 0.9,
                "content": 0.7,
                "satisfied": 0.7,
                "good": 0.6,
            },
            EmotionType.SADNESS: {
                "sad": 0.8,
                "unhappy": 0.8,
                "depressed": 0.9,
                "miserable": 0.9,
                "down": 0.7,
                "disappointed": 0.75,
                "heartbroken": 0.95,
                "sorry": 0.7,
                "hurt": 0.8,
                "upset": 0.75,
                "cry": 0.85,
                "tears": 0.8,
                "grief": 0.9,
                "sorrow": 0.85,
                "lonely": 0.8,
                "hopeless": 0.9,
                "despair": 0.95,
                "gloomy": 0.75,
                "blue": 0.7,
                "melancholy": 0.8,
            },
            EmotionType.ANGER: {
                "angry": 0.85,
                "mad": 0.8,
                "furious": 0.95,
                "rage": 0.95,
                "annoyed": 0.7,
                "irritated": 0.75,
                "frustrated": 0.8,
                "hate": 0.9,
                "outraged": 0.9,
                "livid": 0.95,
                "enraged": 0.95,
                "irate": 0.85,
                "hostile": 0.85,
                "resentful": 0.8,
                "bitter": 0.8,
                "infuriated": 0.95,
                "aggravated": 0.75,
                "pissed": 0.85,
                "upset": 0.75,
            },
            EmotionType.FEAR: {
                "afraid": 0.85,
                "scared": 0.85,
                "terrified": 0.95,
                "fear": 0.85,
                "anxious": 0.8,
                "worried": 0.75,
                "nervous": 0.75,
                "panic": 0.9,
                "frightened": 0.85,
                "alarmed": 0.8,
                "apprehensive": 0.75,
                "dread": 0.85,
                "horror": 0.9,
                "paranoid": 0.85,
                "uneasy": 0.7,
                "tense": 0.7,
                "stressed": 0.75,
                "threatened": 0.8,
            },
            EmotionType.SURPRISE: {
                "surprised": 0.8,
                "shocked": 0.85,
                "amazed": 0.8,
                "astonished": 0.85,
                "stunned": 0.85,
                "startled": 0.75,
                "unexpected": 0.7,
                "wow": 0.75,
                "incredible": 0.8,
                "unbelievable": 0.8,
                "astounded": 0.85,
                "flabbergasted": 0.85,
                "dumbfounded": 0.85,
                "speechless": 0.8,
            },
            EmotionType.DISGUST: {
                "disgusted": 0.85,
                "revolted": 0.85,
                "repulsed": 0.85,
                "sickened": 0.8,
                "nauseated": 0.8,
                "gross": 0.75,
                "nasty": 0.75,
                "horrible": 0.8,
                "terrible": 0.75,
                "awful": 0.75,
                "repugnant": 0.85,
                "loathe": 0.85,
                "detest": 0.8,
                "abhor": 0.85,
                "offensive": 0.75,
            },
            EmotionType.TRUST: {
                "trust": 0.85,
                "confident": 0.8,
                "secure": 0.75,
                "reliable": 0.75,
                "safe": 0.75,
                "certain": 0.7,
                "assured": 0.75,
                "believe": 0.8,
                "faith": 0.85,
                "dependable": 0.75,
                "honest": 0.75,
                "loyal": 0.8,
                "committed": 0.75,
            },
            EmotionType.ANTICIPATION: {
                "excited": 0.8,
                "eager": 0.8,
                "hopeful": 0.75,
                "optimistic": 0.75,
                "looking forward": 0.8,
                "expecting": 0.7,
                "anticipating": 0.8,
                "waiting": 0.6,
                "prepared": 0.7,
                "ready": 0.7,
                "curious": 0.7,
                "interested": 0.7,
            },
        }

        # Intensifiers boost emotion scores
        self.intensifiers = {
            "very": 1.3,
            "extremely": 1.5,
            "incredibly": 1.5,
            "absolutely": 1.4,
            "totally": 1.3,
            "completely": 1.3,
            "so": 1.2,
            "really": 1.2,
            "super": 1.3,
            "highly": 1.2,
            "utterly": 1.4,
            "seriously": 1.2,
        }

        # Diminishers reduce emotion scores
        self.diminishers = {
            "slightly": 0.6,
            "somewhat": 0.7,
            "a bit": 0.7,
            "a little": 0.7,
            "kind of": 0.7,
            "sort of": 0.7,
            "mildly": 0.6,
            "barely": 0.5,
            "hardly": 0.5,
        }

        # Negation words flip emotion polarity
        self.negation_words = {
            "not",
            "no",
            "never",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "can't",
            "cannot",
            "shouldn't",
            "wouldn't",
        }

    def detect(self, text: str) -> EmotionDetection:
        """
        Detect emotions in text.

        Args:
            text: Input text to analyze

        Returns:
            EmotionDetection with primary emotion and scores

        Example:
            >>> result = detector.detect("I'm really angry about this!")
            >>> print(result.primary_emotion, result.intensity)
            EmotionType.ANGER 0.92
        """
        if not text or not text.strip():
            return EmotionDetection(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=0.0,
                all_emotions={},
                confidence=1.0,
                keywords_matched=[],
            )

        # Preprocess text
        text_lower = text.lower()
        words = self._tokenize(text_lower)

        # Detect emotions with scores
        emotion_scores: Dict[EmotionType, float] = {}
        keywords_matched: List[str] = []

        for emotion_type, keywords in self.emotion_keywords.items():
            score, matched = self._calculate_emotion_score(
                text_lower, words, keywords
            )
            if score > 0:
                emotion_scores[emotion_type] = score
                keywords_matched.extend(matched)

        # Determine primary emotion
        if not emotion_scores:
            return EmotionDetection(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=0.0,
                all_emotions={},
                confidence=1.0,
                keywords_matched=[],
            )

        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        emotion_type, raw_score = primary_emotion

        # Normalize intensity (0.0-1.0)
        intensity = min(1.0, raw_score)

        # Calculate confidence based on score separation
        if len(emotion_scores) > 1:
            sorted_scores = sorted(emotion_scores.values(), reverse=True)
            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        else:
            confidence = 1.0

        return EmotionDetection(
            primary_emotion=emotion_type,
            intensity=intensity,
            all_emotions=emotion_scores,
            confidence=confidence,
            keywords_matched=keywords_matched,
        )

    def _calculate_emotion_score(
        self, text: str, words: List[str], keywords: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """
        Calculate score for specific emotion.

        Args:
            text: Preprocessed text
            words: Tokenized words
            keywords: Emotion keywords with base scores

        Returns:
            (total_score, matched_keywords)
        """
        total_score = 0.0
        matched_keywords = []

        for keyword, base_score in keywords.items():
            # Check if keyword present
            if keyword not in text:
                continue

            matched_keywords.append(keyword)

            # Find keyword position
            keyword_idx = self._find_keyword_index(words, keyword)
            if keyword_idx == -1:
                # Multi-word keyword or not found
                total_score += base_score
                continue

            # Check for intensifiers/diminishers before keyword
            modifier = self._find_modifier(words, keyword_idx)
            modified_score = base_score * modifier

            # Check for negation
            if self._has_negation(words, keyword_idx):
                # Flip emotion (joy -> sadness, etc.)
                modified_score *= -0.5

            total_score += modified_score

        return total_score, matched_keywords

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of lowercase words
        """
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        return text.lower().split()

    def _find_keyword_index(self, words: List[str], keyword: str) -> int:
        """
        Find index of keyword in word list.

        Args:
            words: List of words
            keyword: Keyword to find

        Returns:
            Index or -1 if not found
        """
        keyword_words = keyword.split()

        if len(keyword_words) == 1:
            try:
                return words.index(keyword)
            except ValueError:
                return -1

        # Multi-word keyword
        for i in range(len(words) - len(keyword_words) + 1):
            if words[i : i + len(keyword_words)] == keyword_words:
                return i

        return -1

    def _find_modifier(self, words: List[str], keyword_idx: int) -> float:
        """
        Find intensifier/diminisher before keyword.

        Args:
            words: Word list
            keyword_idx: Index of emotion keyword

        Returns:
            Modifier multiplier (default 1.0)
        """
        # Check previous 3 words
        for i in range(max(0, keyword_idx - 3), keyword_idx):
            word = words[i]
            if word in self.intensifiers:
                return self.intensifiers[word]
            if word in self.diminishers:
                return self.diminishers[word]

        return 1.0

    def _has_negation(self, words: List[str], keyword_idx: int) -> bool:
        """
        Check if keyword is negated.

        Args:
            words: Word list
            keyword_idx: Index of emotion keyword

        Returns:
            True if negated
        """
        # Check previous 3 words
        for i in range(max(0, keyword_idx - 3), keyword_idx):
            if words[i] in self.negation_words:
                return True

        return False

    def get_emotion_summary(self, detection: EmotionDetection) -> str:
        """
        Generate human-readable summary of detection.

        Args:
            detection: Emotion detection result

        Returns:
            Summary string

        Example:
            >>> summary = detector.get_emotion_summary(result)
            >>> print(summary)
            "Primary: joy (85% intensity, 92% confidence)"
        """
        intensity_pct = int(detection.intensity * 100)
        confidence_pct = int(detection.confidence * 100)

        summary = (
            f"Primary: {detection.primary_emotion.value} "
            f"({intensity_pct}% intensity, {confidence_pct}% confidence)"
        )

        if len(detection.all_emotions) > 1:
            other_emotions = [
                f"{em.value} ({int(score * 100)}%)"
                for em, score in sorted(
                    detection.all_emotions.items(), key=lambda x: x[1], reverse=True
                )[1:3]  # Top 2 secondary emotions
            ]
            if other_emotions:
                summary += f" | Secondary: {', '.join(other_emotions)}"

        return summary
