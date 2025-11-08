"""
JENOVA Cognitive Architecture - Intent Classifier Module
Copyright (c) 2024, orpheus497. All rights reserved.
Licensed under the MIT License

This module provides natural language intent classification using pattern matching
and heuristic analysis for understanding user commands.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any


logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Enumeration of recognized intent types."""

    # Code operations
    CODE_ANALYSIS = "code_analysis"
    CODE_REFACTOR = "code_refactor"
    CODE_REVIEW = "code_review"
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    CODE_DEBUG = "code_debug"

    # Git operations
    GIT_COMMIT = "git_commit"
    GIT_BRANCH = "git_branch"
    GIT_MERGE = "git_merge"
    GIT_REBASE = "git_rebase"
    GIT_STATUS = "git_status"
    GIT_DIFF = "git_diff"
    GIT_LOG = "git_log"

    # File operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    FILE_DELETE = "file_delete"
    FILE_SEARCH = "file_search"
    FILE_CREATE = "file_create"

    # Project operations
    PROJECT_BUILD = "project_build"
    PROJECT_TEST = "project_test"
    PROJECT_DEPLOY = "project_deploy"
    PROJECT_SETUP = "project_setup"
    PROJECT_ANALYZE = "project_analyze"

    # Documentation
    DOC_GENERATE = "doc_generate"
    DOC_EXPLAIN = "doc_explain"
    DOC_UPDATE = "doc_update"

    # System operations
    SYSTEM_COMMAND = "system_command"
    SYSTEM_INSTALL = "system_install"
    SYSTEM_CONFIG = "system_config"

    # Conversational
    QUESTION = "question"
    GREETING = "greeting"
    HELP = "help"
    GENERAL = "general"


@dataclass
class Intent:
    """Represents a classified intent with confidence and entities."""

    intent_type: IntentType
    confidence: float
    text: str
    entities: Dict[str, Any] = field(default_factory=dict)
    secondary_intents: List[Tuple[IntentType, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary format."""
        return {
            "intent_type": self.intent_type.value,
            "confidence": self.confidence,
            "text": self.text,
            "entities": self.entities,
            "secondary_intents": [(i.value, c) for i, c in self.secondary_intents]
        }


class IntentClassifier:
    """
    Natural language intent classifier using pattern matching and heuristics.

    This classifier analyzes user input to determine the intended action,
    supporting multiple categories of operations including code, git, file,
    and system operations.
    """

    def __init__(self):
        """Initialize the intent classifier with pattern rules."""
        self.patterns = self._build_patterns()
        self.entity_extractors = self._build_entity_extractors()

    def _build_patterns(self) -> Dict[IntentType, List[str]]:
        """
        Build regex patterns for each intent type.

        Returns:
            Dictionary mapping intent types to their regex patterns
        """
        return {
            # Code operations
            IntentType.CODE_ANALYSIS: [
                r'\b(analyz|inspect|examine|review|check)\s+.*\b(code|function|class|module)',
                r'\b(complexity|metrics|quality)\b',
                r'\bcode\s+(quality|metrics|analysis)',
                r'\banalyze\s+.*\.py',
            ],
            IntentType.CODE_REFACTOR: [
                r'\b(refactor|restructure|reorganize|improve|optimize)\s+.*\b(code|function|class)',
                r'\bclean\s+up\s+.*code',
                r'\bextract\s+(method|function|class)',
                r'\brename\s+(variable|function|class|method)',
            ],
            IntentType.CODE_REVIEW: [
                r'\b(review|critique)\s+.*\b(code|pr|pull\s+request|changes)',
                r'\bcode\s+review',
                r'\breview\s+pull\s+request',
            ],
            IntentType.CODE_GENERATION: [
                r'\b(create|generate|write|implement|add)\s+.*\b(function|class|module|script|code)',
                r'\bscaffold\s+.*',
                r'\bgenerate\s+.*code',
                r'\bcreate\s+.*\.py',
            ],
            IntentType.CODE_EXPLANATION: [
                r'\b(explain|describe|what\s+(does|is))\s+.*\b(code|function|class)',
                r'\bhow\s+does\s+.*\bwork',
                r'\bwhat.*this.*do',
            ],
            IntentType.CODE_DEBUG: [
                r'\b(debug|fix|solve|troubleshoot)\s+.*\b(error|bug|issue|problem)',
                r'\bwhy.*not\s+working',
                r'\berror\s+in\s+.*',
                r'\bexception\s+in\s+.*',
            ],

            # Git operations
            IntentType.GIT_COMMIT: [
                r'\b(commit|save)\s+.*\b(changes|files)',
                r'\bgit\s+commit',
                r'\bmake\s+.*commit',
            ],
            IntentType.GIT_BRANCH: [
                r'\b(create|switch|checkout|delete)\s+.*\b(branch)',
                r'\bgit\s+(branch|checkout)',
                r'\bnew\s+branch',
            ],
            IntentType.GIT_MERGE: [
                r'\b(merge|combine)\s+.*\b(branch|changes)',
                r'\bgit\s+merge',
            ],
            IntentType.GIT_REBASE: [
                r'\brebase\s+.*',
                r'\bgit\s+rebase',
            ],
            IntentType.GIT_STATUS: [
                r'\bgit\s+status',
                r'\b(show|check)\s+.*\b(status|changes)',
                r'\bwhat.*changed',
            ],
            IntentType.GIT_DIFF: [
                r'\bgit\s+diff',
                r'\b(show|display)\s+.*\b(diff|difference|changes)',
                r'\bcompare\s+.*',
            ],
            IntentType.GIT_LOG: [
                r'\bgit\s+(log|history)',
                r'\b(show|view)\s+.*\b(history|commits|log)',
                r'\bcommit\s+history',
            ],

            # File operations
            IntentType.FILE_READ: [
                r'\b(read|show|display|cat|view|open)\s+.*\b(file|\.py|\.txt|\.md)',
                r'\bshow\s+me\s+.*',
                r'\bwhat.*in\s+.*file',
            ],
            IntentType.FILE_WRITE: [
                r'\b(write|save|create)\s+.*\b(to\s+)?file',
                r'\bsave\s+.*to\s+.*',
            ],
            IntentType.FILE_EDIT: [
                r'\b(edit|modify|update|change)\s+.*\b(file|\.py|\.txt)',
                r'\bupdate\s+.*file',
                r'\bmodify\s+.*in\s+.*',
            ],
            IntentType.FILE_DELETE: [
                r'\b(delete|remove|rm)\s+.*\b(file|\.py|\.txt)',
                r'\bremove\s+.*file',
            ],
            IntentType.FILE_SEARCH: [
                r'\b(find|search|locate|grep)\s+.*\b(file|files|in)',
                r'\bsearch\s+for\s+.*',
                r'\bwhere\s+is\s+.*',
            ],
            IntentType.FILE_CREATE: [
                r'\b(create|make|new)\s+.*\b(file|directory|folder)',
                r'\bmkdir\s+.*',
                r'\btouch\s+.*',
            ],

            # Project operations
            IntentType.PROJECT_BUILD: [
                r'\b(build|compile|make)\s+.*\b(project|code|app)',
                r'\brun\s+(build|make)',
                r'\bnpm\s+run\s+build',
                r'\bmake\s+build',
            ],
            IntentType.PROJECT_TEST: [
                r'\b(test|run\s+tests|execute\s+tests)',
                r'\bpytest\s+.*',
                r'\bunittest\s+.*',
                r'\bnpm\s+test',
            ],
            IntentType.PROJECT_DEPLOY: [
                r'\b(deploy|publish|release)\s+.*',
                r'\bpush\s+to\s+(production|staging)',
            ],
            IntentType.PROJECT_SETUP: [
                r'\b(setup|initialize|init)\s+.*\b(project|repo|repository)',
                r'\bgit\s+init',
                r'\bnpm\s+init',
            ],
            IntentType.PROJECT_ANALYZE: [
                r'\b(analyze|audit|scan)\s+.*\b(project|codebase|repo)',
                r'\bproject\s+(analysis|audit)',
            ],

            # Documentation
            IntentType.DOC_GENERATE: [
                r'\b(generate|create|write)\s+.*\b(docs|documentation|readme)',
                r'\bdocument\s+.*',
            ],
            IntentType.DOC_EXPLAIN: [
                r'\bexplain\s+.*\b(feature|api|usage)',
                r'\bhow\s+to\s+use\s+.*',
            ],
            IntentType.DOC_UPDATE: [
                r'\b(update|modify|fix)\s+.*\b(docs|documentation|readme)',
            ],

            # System operations
            IntentType.SYSTEM_COMMAND: [
                r'\b(run|execute)\s+.*\b(command|script)',
                r'^\s*[\w/]+\s+.*',  # Command-like syntax
            ],
            IntentType.SYSTEM_INSTALL: [
                r'\b(install|setup|add)\s+.*\b(package|dependency|library)',
                r'\bpip\s+install\s+.*',
                r'\bnpm\s+install\s+.*',
            ],
            IntentType.SYSTEM_CONFIG: [
                r'\b(configure|config|setup)\s+.*',
                r'\bsettings\s+.*',
            ],

            # Conversational
            IntentType.QUESTION: [
                r'^\s*(what|why|how|when|where|who|can|could|would|should)',
                r'\?\s*$',
            ],
            IntentType.GREETING: [
                r'\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))',
            ],
            IntentType.HELP: [
                r'\b(help|assist|support)\b',
                r'\bhow\s+do\s+i\s+.*',
            ],
        }

    def _build_entity_extractors(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Build entity extraction patterns.

        Returns:
            Dictionary of entity types to their (name, pattern) tuples
        """
        return {
            'file_path': [
                ('path', r'(?:^|\s)((?:\.{0,2}/)?[\w/.-]+\.[\w]+)(?:\s|$)'),
                ('path', r'(?:^|\s)([\w/.-]+/[\w/.-]+)(?:\s|$)'),
            ],
            'git_ref': [
                ('branch', r'\bbranch\s+(\w+[\w/-]*)'),
                ('commit', r'\bcommit\s+([0-9a-f]{7,40})'),
                ('tag', r'\btag\s+([\w.-]+)'),
            ],
            'code_entity': [
                ('function', r'\bfunction\s+(\w+)'),
                ('class', r'\bclass\s+(\w+)'),
                ('method', r'\bmethod\s+(\w+)'),
                ('variable', r'\bvariable\s+(\w+)'),
            ],
            'language': [
                ('language', r'\b(python|javascript|java|cpp|c\+\+|rust|go|typescript|ruby)\b'),
            ],
        }

    def classify(self, text: str) -> Intent:
        """
        Classify the intent of user input text.

        Args:
            text: User input text to classify

        Returns:
            Intent object with classification results
        """
        if not text or not text.strip():
            return Intent(
                intent_type=IntentType.GENERAL,
                confidence=1.0,
                text=text
            )

        text_lower = text.lower().strip()

        # Score all intents
        scores = self._score_intents(text_lower)

        # Get top intent
        if not scores:
            primary_intent = IntentType.GENERAL
            confidence = 0.5
        else:
            primary_intent, confidence = max(scores.items(), key=lambda x: x[1])

        # Get secondary intents
        secondary_intents = [
            (intent, score) for intent, score in scores.items()
            if intent != primary_intent and score > 0.3
        ]
        secondary_intents.sort(key=lambda x: x[1], reverse=True)

        # Extract entities
        entities = self._extract_entities(text)

        return Intent(
            intent_type=primary_intent,
            confidence=confidence,
            text=text,
            entities=entities,
            secondary_intents=secondary_intents[:3]  # Top 3 secondary
        )

    def _score_intents(self, text: str) -> Dict[IntentType, float]:
        """
        Score all intent types for the given text.

        Args:
            text: Lowercase text to score

        Returns:
            Dictionary of intent types to confidence scores
        """
        scores = defaultdict(float)

        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        # Increase score for this intent
                        scores[intent_type] += 0.3

                        # Bonus for exact keyword matches
                        if self._has_exact_keywords(text, pattern):
                            scores[intent_type] += 0.2
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {intent_type}: {pattern} - {e}")

        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: min(1.0, v / max_score) for k, v in scores.items()}

        return dict(scores)

    def _has_exact_keywords(self, text: str, pattern: str) -> bool:
        """Check if text contains exact keywords from pattern."""
        # Extract simple words from pattern (ignoring regex syntax)
        keywords = re.findall(r'\b([a-z]{3,})\b', pattern.lower())
        return any(keyword in text for keyword in keywords)

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        for entity_type, extractors in self.entity_extractors.items():
            for name, pattern in extractors:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if entity_type not in entities:
                        entities[entity_type] = {}
                    if name not in entities[entity_type]:
                        entities[entity_type][name] = []
                    entities[entity_type][name].extend(matches)

        # Also extract quoted strings as potential entities
        quoted_strings = re.findall(r'["\']([^"\']+)["\']', text)
        if quoted_strings:
            entities['quoted_strings'] = quoted_strings

        # Extract file extensions
        extensions = re.findall(r'\.([a-z]{2,4})\b', text.lower())
        if extensions:
            entities['file_extensions'] = list(set(extensions))

        return entities

    def classify_batch(self, texts: List[str]) -> List[Intent]:
        """
        Classify multiple texts at once.

        Args:
            texts: List of texts to classify

        Returns:
            List of Intent objects
        """
        return [self.classify(text) for text in texts]

    def get_intent_keywords(self, intent_type: IntentType) -> Set[str]:
        """
        Get the keywords associated with an intent type.

        Args:
            intent_type: The intent type to get keywords for

        Returns:
            Set of keywords for the intent type
        """
        keywords = set()

        if intent_type in self.patterns:
            for pattern in self.patterns[intent_type]:
                # Extract simple words from pattern
                words = re.findall(r'\b([a-z]{3,})\b', pattern.lower())
                keywords.update(words)

        return keywords

    def add_custom_pattern(self, intent_type: IntentType, pattern: str) -> None:
        """
        Add a custom pattern for an intent type.

        Args:
            intent_type: The intent type to add pattern for
            pattern: Regex pattern to add
        """
        if intent_type not in self.patterns:
            self.patterns[intent_type] = []

        # Validate pattern
        try:
            re.compile(pattern)
            self.patterns[intent_type].append(pattern)
            logger.info(f"Added custom pattern for {intent_type.value}: {pattern}")
        except re.error as e:
            logger.error(f"Invalid regex pattern: {pattern} - {e}")
            raise ValueError(f"Invalid regex pattern: {e}")

    def remove_pattern(self, intent_type: IntentType, pattern: str) -> bool:
        """
        Remove a pattern from an intent type.

        Args:
            intent_type: The intent type to remove pattern from
            pattern: Pattern to remove

        Returns:
            True if pattern was removed, False otherwise
        """
        if intent_type in self.patterns and pattern in self.patterns[intent_type]:
            self.patterns[intent_type].remove(pattern)
            logger.info(f"Removed pattern from {intent_type.value}: {pattern}")
            return True
        return False

    def get_similar_intents(self, intent_type: IntentType, threshold: float = 0.5) -> List[IntentType]:
        """
        Get intent types that are similar to the given intent.

        Args:
            intent_type: The intent type to find similar ones for
            threshold: Minimum similarity threshold

        Returns:
            List of similar intent types
        """
        # This is a simple heuristic based on shared keywords
        keywords = self.get_intent_keywords(intent_type)
        similar = []

        for other_intent in IntentType:
            if other_intent == intent_type:
                continue

            other_keywords = self.get_intent_keywords(other_intent)
            if not keywords or not other_keywords:
                continue

            # Calculate Jaccard similarity
            intersection = len(keywords & other_keywords)
            union = len(keywords | other_keywords)
            similarity = intersection / union if union > 0 else 0

            if similarity >= threshold:
                similar.append(other_intent)

        return similar
