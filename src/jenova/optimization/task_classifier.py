# The JENOVA Cognitive Architecture - Task Classifier
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 25: Task Classifier for automatic task type detection.

Classifies user queries into task types (general_qa, code_generation,
summarization, analysis, creative_writing) to select optimal parameters.
"""

from typing import Dict, List, Tuple, Optional
import re


class TaskClassifier:
    """
    Automatic task type detection.

    Uses keyword matching and pattern recognition to classify queries.
    In production, could use ML classifier for better accuracy.

    Task types:
        - general_qa: General questions and answers
        - code_generation: Code writing and programming
        - summarization: Text summarization
        - analysis: Data or code analysis
        - creative_writing: Stories, creative content

    Example:
        >>> classifier = TaskClassifier()
        >>> task_type = classifier.classify_task(
        ...     "Write a Python function to sort a list"
        ... )
        >>> print(task_type)  # "code_generation"
    """

    def __init__(self):
        """Initialize task classifier with keyword patterns."""
        # Define keywords for each task type
        self.task_keywords = {
            "code_generation": [
                "write",
                "code",
                "function",
                "class",
                "implement",
                "program",
                "script",
                "algorithm",
                "debug",
                "fix",
                "refactor",
                "optimize",
                "python",
                "javascript",
                "java",
                "c++",
                "rust",
                "def ",
                "async ",
                "lambda",
                "for loop",
                "while loop",
            ],
            "summarization": [
                "summarize",
                "summary",
                "tldr",
                "brief",
                "condense",
                "shorten",
                "key points",
                "main ideas",
                "in short",
                "quick overview",
            ],
            "analysis": [
                "analyze",
                "analysis",
                "examine",
                "investigate",
                "evaluate",
                "assess",
                "review",
                "compare",
                "contrast",
                "pros and cons",
                "advantages",
                "disadvantages",
                "performance",
                "metrics",
                "statistics",
            ],
            "creative_writing": [
                "write a story",
                "create a poem",
                "imagine",
                "fiction",
                "narrative",
                "character",
                "plot",
                "creative",
                "storytelling",
                "once upon",
                "describe",
                "vivid",
            ],
            # general_qa is the default fallback
        }

        # Compile regex patterns
        self.patterns = {
            task_type: self._compile_patterns(keywords)
            for task_type, keywords in self.task_keywords.items()
        }

    def _compile_patterns(self, keywords: List[str]) -> List[re.Pattern]:
        """
        Compile keyword patterns.

        Args:
            keywords: List of keywords

        Returns:
            List of compiled regex patterns
        """
        patterns = []
        for keyword in keywords:
            # Word boundary for whole words, case-insensitive
            pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def classify_task(
        self, user_input: str, context: Optional[str] = None
    ) -> str:
        """
        Classify task type from user input.

        Args:
            user_input: User query
            context: Optional context (previous messages, etc.)

        Returns:
            Task type string

        Example:
            >>> task = classifier.classify_task("Explain quantum physics")
            >>> print(task)  # "general_qa"
        """
        # Combine input and context
        text = user_input.lower()
        if context:
            text += " " + context.lower()

        # Score each task type
        scores = self._score_task_types(text)

        # Return highest scoring task type
        if not scores:
            return "general_qa"

        best_task = max(scores.items(), key=lambda x: x[1])

        # Require minimum score threshold
        if best_task[1] < 1.0:
            return "general_qa"  # Default

        return best_task[0]

    def _score_task_types(self, text: str) -> Dict[str, float]:
        """
        Score each task type based on keyword matches.

        Args:
            text: Text to analyze

        Returns:
            Dict mapping task types to scores
        """
        scores: Dict[str, float] = {}

        for task_type, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                # Count matches
                matches = len(pattern.findall(text))
                score += matches

            if score > 0:
                scores[task_type] = score

        return scores

    def get_task_characteristics(self, task_type: str) -> Dict[str, any]:
        """
        Get optimal characteristics for task type.

        Args:
            task_type: Task type

        Returns:
            Dict with optimal parameter ranges and description

        Example:
            >>> chars = classifier.get_task_characteristics("code_generation")
            >>> print(chars["temperature_range"])  # (0.2, 0.4)
        """
        characteristics = {
            "general_qa": {
                "description": "General question answering",
                "temperature_range": (0.6, 0.8),
                "max_tokens_range": (256, 512),
                "creativity": "medium",
            },
            "code_generation": {
                "description": "Code writing and programming",
                "temperature_range": (0.2, 0.4),
                "max_tokens_range": (512, 1024),
                "creativity": "low",
            },
            "summarization": {
                "description": "Text summarization and condensing",
                "temperature_range": (0.2, 0.4),
                "max_tokens_range": (128, 256),
                "creativity": "low",
            },
            "analysis": {
                "description": "Analysis and evaluation",
                "temperature_range": (0.3, 0.5),
                "max_tokens_range": (512, 1024),
                "creativity": "low-medium",
            },
            "creative_writing": {
                "description": "Creative and narrative writing",
                "temperature_range": (0.8, 1.0),
                "max_tokens_range": (512, 1536),
                "creativity": "high",
            },
        }

        return characteristics.get(task_type, characteristics["general_qa"])

    def classify_with_confidence(
        self, user_input: str, context: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Classify task with confidence score.

        Args:
            user_input: User query
            context: Optional context

        Returns:
            (task_type, confidence) tuple

        Example:
            >>> task, conf = classifier.classify_with_confidence(
            ...     "Write a sorting algorithm"
            ... )
            >>> print(f"{task} ({conf:.2f})")  # "code_generation (0.85)"
        """
        text = user_input.lower()
        if context:
            text += " " + context.lower()

        scores = self._score_task_types(text)

        if not scores:
            return "general_qa", 0.5  # Default with low confidence

        # Calculate confidence based on score distribution
        total_score = sum(scores.values())
        best_task = max(scores.items(), key=lambda x: x[1])
        task_type, best_score = best_task

        # Confidence is proportion of best score to total
        confidence = best_score / total_score if total_score > 0 else 0.5

        # Boost confidence for clear matches
        if best_score >= 3.0:
            confidence = min(1.0, confidence * 1.2)

        return task_type, confidence
