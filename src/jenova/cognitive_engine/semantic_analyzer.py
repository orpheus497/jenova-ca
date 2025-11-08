# The JENOVA Cognitive Architecture - Semantic Query Analyzer
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 11: Semantic understanding and query analysis.

Provides:
- Intent classification (question, command, statement, request)
- Entity recognition and linking
- Sentiment analysis
- Query expansion for better retrieval
- Rhetorical structure analysis
"""

import re
from typing import Any, Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict


class QueryIntent(Enum):
    """Types of query intents."""
    QUESTION = "question"  # Seeking information
    COMMAND = "command"  # Executing an action
    STATEMENT = "statement"  # Sharing information
    REQUEST = "request"  # Asking for help or action
    CLARIFICATION = "clarification"  # Asking for clarification
    FEEDBACK = "feedback"  # Providing feedback
    GREETING = "greeting"  # Social interaction
    UNKNOWN = "unknown"


class Sentiment(Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class Entity:
    """Represents a recognized entity."""
    text: str
    entity_type: str  # person, place, concept, tech, time, etc.
    start_pos: int
    end_pos: int
    confidence: float = 1.0


@dataclass
class SemanticAnalysis:
    """Complete semantic analysis of a query."""
    original_query: str
    intent: QueryIntent
    confidence: float
    entities: List[Entity]
    keywords: List[str]
    expanded_query: str
    sentiment: Sentiment
    topics: List[str]
    dependencies: List[Tuple[str, str, str]]  # (word, relation, word)
    rhetorical_elements: Dict[str, Any]


class SemanticAnalyzer:
    """
    Semantic analyzer for understanding user queries.

    Performs:
    - Intent classification
    - Entity recognition
    - Sentiment analysis
    - Keyword extraction
    - Query expansion
    - Topic modeling
    """

    def __init__(self, llm_interface, file_logger):
        self.llm = llm_interface
        self.file_logger = file_logger

        # Intent detection patterns
        self.question_words = {
            'what', 'when', 'where', 'who', 'whom', 'whose', 'which',
            'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are',
            'do', 'does', 'did', 'has', 'have', 'had'
        }

        self.command_words = {
            'show', 'tell', 'explain', 'describe', 'define', 'list',
            'give', 'provide', 'create', 'make', 'generate', 'write',
            'calculate', 'compute', 'find', 'search', 'analyze'
        }

        self.greeting_words = {
            'hello', 'hi', 'hey', 'greetings', 'good morning',
            'good afternoon', 'good evening', 'bye', 'goodbye',
            'see you', 'thanks', 'thank you'
        }

        # Entity type patterns
        self.tech_terms = {
            'ai', 'ml', 'algorithm', 'data', 'model', 'neural',
            'network', 'deep learning', 'machine learning',
            'api', 'database', 'framework', 'library', 'code',
            'python', 'javascript', 'java', 'c++', 'rust'
        }

        # Sentiment indicators
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'perfect',
            'amazing', 'wonderful', 'fantastic', 'love', 'like',
            'best', 'better', 'nice', 'helpful', 'useful'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor',
            'worst', 'worse', 'hate', 'dislike', 'wrong',
            'error', 'problem', 'issue', 'fail', 'failed'
        }

        # Stopwords for keyword extraction
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
            'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their'
        }

    def analyze(self, query: str) -> SemanticAnalysis:
        """
        Perform complete semantic analysis of a query.

        Args:
            query: The user query to analyze

        Returns:
            SemanticAnalysis object with all extracted information
        """
        # Normalize query
        normalized_query = query.strip()
        query_lower = normalized_query.lower()
        words = self._tokenize(query_lower)

        # Intent classification
        intent, confidence = self._classify_intent(normalized_query, words)

        # Entity recognition
        entities = self._recognize_entities(normalized_query, words)

        # Keyword extraction
        keywords = self._extract_keywords(words, entities)

        # Query expansion
        expanded_query = self._expand_query(normalized_query, keywords, entities)

        # Sentiment analysis
        sentiment = self._analyze_sentiment(words)

        # Topic extraction
        topics = self._extract_topics(keywords, entities)

        # Rhetorical elements
        rhetorical = self._analyze_rhetoric(normalized_query, words)

        return SemanticAnalysis(
            original_query=query,
            intent=intent,
            confidence=confidence,
            entities=entities,
            keywords=keywords,
            expanded_query=expanded_query,
            sentiment=sentiment,
            topics=topics,
            dependencies=[],  # Would use dependency parser in full implementation
            rhetorical_elements=rhetorical
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation except for important cases
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        return [w for w in text.split() if w]

    def _classify_intent(self, query: str, words: List[str]) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of the query.

        Returns:
            (intent, confidence) tuple
        """
        query_lower = query.lower()
        first_words = words[:3] if len(words) >= 3 else words

        # Check for commands (starts with /)
        if query.startswith('/'):
            return QueryIntent.COMMAND, 1.0

        # Check for greetings
        for word in first_words:
            if word in self.greeting_words:
                return QueryIntent.GREETING, 0.9

        # Check for questions
        if query.endswith('?'):
            return QueryIntent.QUESTION, 0.95

        # Check for question words at start
        if first_words and first_words[0] in self.question_words:
            return QueryIntent.QUESTION, 0.85

        # Check for command/request verbs
        if first_words and first_words[0] in self.command_words:
            # Distinguish between command and request
            if any(word in words for word in ['please', 'could you', 'would you']):
                return QueryIntent.REQUEST, 0.8
            return QueryIntent.COMMAND, 0.8

        # Check for feedback indicators
        feedback_indicators = ['think', 'believe', 'seems', 'appears', 'looks like']
        if any(ind in query_lower for ind in feedback_indicators):
            return QueryIntent.FEEDBACK, 0.7

        # Check for clarification
        clarification_words = {'mean', 'clarify', 'explain again', 'what do you'}
        if any(word in query_lower for word in clarification_words):
            return QueryIntent.CLARIFICATION, 0.75

        # Default to statement
        return QueryIntent.STATEMENT, 0.6

    def _recognize_entities(self, query: str, words: List[str]) -> List[Entity]:
        """
        Recognize entities in the query.

        In a full implementation, this would use NER models.
        This is a simplified pattern-based approach.
        """
        entities = []

        # Technical terms
        query_lower = query.lower()
        for term in self.tech_terms:
            if term in query_lower:
                start = query_lower.find(term)
                entities.append(Entity(
                    text=term,
                    entity_type="technology",
                    start_pos=start,
                    end_pos=start + len(term),
                    confidence=0.8
                ))

        # Capitalized words (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(capitalized_pattern, query):
            text = match.group()
            # Skip if it's at the start of the sentence (might just be capitalization)
            if match.start() > 0:
                entities.append(Entity(
                    text=text,
                    entity_type="proper_noun",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.6
                ))

        # Numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        for match in re.finditer(number_pattern, query):
            entities.append(Entity(
                text=match.group(),
                entity_type="number",
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=1.0
            ))

        # Time expressions
        time_patterns = [
            r'\b(?:yesterday|today|tomorrow|tonight)\b',
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        for pattern in time_patterns:
            for match in re.finditer(pattern, query_lower):
                entities.append(Entity(
                    text=match.group(),
                    entity_type="time",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9
                ))

        return entities

    def _extract_keywords(self, words: List[str], entities: List[Entity]) -> List[str]:
        """
        Extract important keywords from the query.

        Args:
            words: Tokenized words
            entities: Recognized entities

        Returns:
            List of keywords sorted by importance
        """
        # Remove stopwords
        keywords = [w for w in words if w not in self.stopwords and len(w) > 2]

        # Add entity text as high-priority keywords
        entity_keywords = [e.text.lower() for e in entities if e.entity_type != "number"]
        keywords.extend(entity_keywords)

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        # Sort by length (longer terms often more specific/important)
        unique_keywords.sort(key=len, reverse=True)

        return unique_keywords[:10]  # Top 10 keywords

    def _expand_query(self, query: str, keywords: List[str], entities: List[Entity]) -> str:
        """
        Expand the query with synonyms and related terms.

        Args:
            query: Original query
            keywords: Extracted keywords
            entities: Recognized entities

        Returns:
            Expanded query string
        """
        # Simple expansion - in full implementation would use word embeddings
        expansions = []

        # Add original query
        expansions.append(query)

        # Add entity-focused variations
        for entity in entities:
            if entity.entity_type == "technology":
                expansions.append(f"{entity.text} documentation")
                expansions.append(f"{entity.text} examples")

        # Add keyword combinations
        if len(keywords) >= 2:
            expansions.append(" ".join(keywords[:3]))

        # Create expanded query (combine with OR logic for search)
        expanded = " OR ".join(set(expansions))
        return expanded

    def _analyze_sentiment(self, words: List[str]) -> Sentiment:
        """Analyze sentiment of the query."""
        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)

        if positive_count == 0 and negative_count == 0:
            return Sentiment.NEUTRAL

        if positive_count > 0 and negative_count > 0:
            return Sentiment.MIXED

        if positive_count > negative_count:
            return Sentiment.POSITIVE

        if negative_count > positive_count:
            return Sentiment.NEGATIVE

        return Sentiment.NEUTRAL

    def _extract_topics(self, keywords: List[str], entities: List[Entity]) -> List[str]:
        """
        Extract topics from keywords and entities.

        Args:
            keywords: Extracted keywords
            entities: Recognized entities

        Returns:
            List of identified topics
        """
        topics = set()

        # Topics from entity types
        entity_types = {e.entity_type for e in entities}
        if 'technology' in entity_types:
            topics.add('technology')
        if 'time' in entity_types:
            topics.add('temporal')

        # Topics from keywords
        tech_keywords = {'code', 'program', 'software', 'algorithm', 'data'}
        if any(kw in keywords for kw in tech_keywords):
            topics.add('programming')

        science_keywords = {'theory', 'research', 'study', 'experiment'}
        if any(kw in keywords for kw in science_keywords):
            topics.add('science')

        business_keywords = {'market', 'business', 'company', 'revenue'}
        if any(kw in keywords for kw in business_keywords):
            topics.add('business')

        # If no topics identified, use "general"
        if not topics:
            topics.add('general')

        return list(topics)

    def _analyze_rhetoric(self, query: str, words: List[str]) -> Dict[str, Any]:
        """
        Analyze rhetorical structure of the query.

        Args:
            query: Original query
            words: Tokenized words

        Returns:
            Dictionary of rhetorical elements
        """
        rhetorical = {
            'length': len(words),
            'complexity': 'simple',
            'formality': 'casual',
            'specificity': 'general'
        }

        # Complexity based on length and structure
        if len(words) > 20:
            rhetorical['complexity'] = 'complex'
        elif len(words) > 10:
            rhetorical['complexity'] = 'moderate'

        # Formality based on language choices
        formal_indicators = {'please', 'kindly', 'would', 'could', 'may', 'might'}
        if any(word in words for word in formal_indicators):
            rhetorical['formality'] = 'formal'

        # Specificity based on entities and numbers
        if query.count('AND') > 0 or query.count('OR') > 0:
            rhetorical['specificity'] = 'very specific'
        elif len(words) > 15:
            rhetorical['specificity'] = 'specific'

        # Detect questions within statements
        question_marks = query.count('?')
        if question_marks > 1:
            rhetorical['multi_part'] = True
            rhetorical['question_count'] = question_marks

        return rhetorical
