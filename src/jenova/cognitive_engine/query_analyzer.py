##Script function and purpose: Query Analysis Module for The JENOVA Cognitive Architecture
##This module analyzes user queries to extract intent, entities, complexity, type, topics,
##entity links to Cortex nodes, query reformulations, and confidence scores for enhanced comprehension
##Phase C.2 Enhancement: Added topic modeling, entity linking, query reformulation, and confidence scoring

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from jenova.utils.json_parser import extract_json

##Enum purpose: Define query intent types for classification
class QueryIntent(Enum):
    QUESTION = "question"
    COMMAND = "command"
    CONVERSATION = "conversation"
    INFORMATION_SEEKING = "information_seeking"
    CLARIFICATION = "clarification"

##Enum purpose: Define query complexity levels for planning adaptation
class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

##Enum purpose: Define topic categories for topic modeling
class TopicCategory(Enum):
    TECHNICAL = "technical"
    PERSONAL = "personal"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"

##Dataclass purpose: Store entity link information connecting entities to Cortex nodes
@dataclass
class EntityLink:
    """Represents a link between an extracted entity and a Cortex node."""
    entity: str
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    confidence: float = 0.0
    relationship: str = "related_to"

##Dataclass purpose: Store topic modeling results with confidence scores
@dataclass
class TopicResult:
    """Represents an extracted topic with category and confidence."""
    topic: str
    category: TopicCategory = TopicCategory.UNKNOWN
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)

##Class purpose: Analyzes user queries to extract structured information for enhanced comprehension
##Enhanced in Phase C.2 with topic modeling, entity linking, query reformulation, and confidence scoring
class QueryAnalyzer:
    ##Function purpose: Initialize query analyzer with LLM interface and configuration
    def __init__(self, llm: Any, config: Dict[str, Any], json_grammar: Optional[Any] = None) -> None:
        self.llm = llm
        self.config = config
        self.json_grammar = json_grammar
        self.comprehension_config = config.get('comprehension', {})
        self.query_analysis_config = self.comprehension_config.get('query_analysis', {})
        
        ##Block purpose: Initialize C.2 enhancement settings
        self.topic_modeling_enabled = self.query_analysis_config.get('topic_modeling', True)
        self.entity_linking_enabled = self.query_analysis_config.get('entity_linking', True)
        self.reformulation_enabled = self.query_analysis_config.get('reformulation', True)
        self.confidence_scoring_enabled = self.query_analysis_config.get('confidence_scoring', True)
        
        ##Block purpose: Store reference for entity linking (set externally by CognitiveEngine)
        self._cortex = None
        self._username = None
        self.enabled = self.query_analysis_config.get('enabled', True)
    
    ##Function purpose: Set Cortex reference for entity linking (called by CognitiveEngine)
    def set_cortex(self, cortex: Any, username: Optional[str] = None) -> None:
        """Set Cortex reference for entity linking capabilities."""
        self._cortex = cortex
        if username is not None:
            self._username = username
    
    ##Function purpose: Set username for entity linking context (called per-request by CognitiveEngine)
    def set_username(self, username: str) -> None:
        """Set current username for entity linking context."""
        self._username = username
    
    ##Function purpose: Get current username for entity linking
    def get_username(self) -> Optional[str]:
        """Get current username for entity linking context."""
        return self._username
    
    ##Function purpose: Analyze query to extract intent, entities, complexity, type, topics, and reformulations
    def analyze(self, query: str) -> Dict[str, Any]:
        """Comprehensive query analysis returning structured information with C.2 enhancements."""
        
        ##Block purpose: Return default analysis if query analysis is disabled
        if not self.enabled:
            return self._default_analysis(query)
        
        ##Block purpose: Build enhanced analysis prompt with all C.2 features
        intent_options = [e.value for e in QueryIntent]
        complexity_options = [e.value for e in QueryComplexity]
        topic_options = [e.value for e in TopicCategory]
        
        prompt = f"""Analyze the following user query and extract structured information:

1. Intent: Classify the query intent as one of: {intent_options}
2. Intent Confidence: Rate confidence in intent classification (0.0 to 1.0)
3. Entities: Extract a list of key entities (people, places, concepts, temporal references, objects)
4. Complexity: Assess complexity as one of: {complexity_options}
5. Complexity Confidence: Rate confidence in complexity assessment (0.0 to 1.0)
6. Query type: Classify as one of: factual, procedural, analytical, creative, conversational
7. Type Confidence: Rate confidence in type classification (0.0 to 1.0)
8. Keywords: Extract 3-5 key terms or phrases
9. Topics: Identify main topics discussed (1-3 topics)
10. Topic Categories: Classify each topic as one of: {topic_options}
11. Alternative Phrasings: Suggest 2-3 alternative ways to phrase this query

Query: "{query}"

Respond with a valid JSON object:
{{
    "intent": "<one of {intent_options}>",
    "intent_confidence": 0.0-1.0,
    "entities": ["entity1", "entity2", ...],
    "complexity": "<one of {complexity_options}>",
    "complexity_confidence": 0.0-1.0,
    "type": "<factual|procedural|analytical|creative|conversational>",
    "type_confidence": 0.0-1.0,
    "keywords": ["keyword1", "keyword2", ...],
    "topics": [
        {{"topic": "topic_name", "category": "<one of {topic_options}>", "confidence": 0.0-1.0}}
    ],
    "reformulations": ["alternative phrasing 1", "alternative phrasing 2"]
}}"""
        
        ##Block purpose: Generate analysis using LLM with JSON grammar
        try:
            analysis_json_str = self.llm.generate(
                prompt, 
                temperature=0.2, 
                grammar=self.json_grammar
            )
            
            ##Block purpose: Parse and validate analysis JSON
            analysis_data = self._parse_analysis(analysis_json_str, query)
            
            ##Block purpose: Perform entity linking if Cortex is available
            if self.entity_linking_enabled and self._cortex is not None:
                analysis_data['entity_links'] = self._link_entities_to_cortex(
                    analysis_data.get('entities', [])
                )
            
            return analysis_data
            
        except Exception as e:
            ##Block purpose: Fallback to default analysis on error
            if hasattr(self, 'file_logger'):
                self.file_logger.log_error(f"Error during query analysis: {e}")
            return self._default_analysis(query)
    
    ##Function purpose: Parse LLM JSON response into structured format with validation and C.2 enhancements
    def _parse_analysis(self, json_str: str, original_query: str = "") -> Dict[str, Any]:
        """Parse and validate analysis JSON response with confidence scores and topics."""
        
        ##Block purpose: Extract JSON from potentially wrapped response
        try:
            data = extract_json(json_str)
        except (ValueError, json.JSONDecodeError):
            ##Block purpose: Fallback to direct JSON parsing
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return self._default_analysis(original_query)
        
        ##Block purpose: Handle None or non-dict data
        if data is None or not isinstance(data, dict):
            return self._default_analysis(original_query)
        
        ##Block purpose: Validate and normalize intent with confidence
        intent = data.get('intent', 'question')
        if intent not in [e.value for e in QueryIntent]:
            intent = 'question'
        intent_confidence = self._validate_confidence(data.get('intent_confidence', 0.5))
        
        ##Block purpose: Validate and normalize complexity with confidence
        complexity = data.get('complexity', 'simple')
        if complexity not in [e.value for e in QueryComplexity]:
            complexity = 'simple'
        complexity_confidence = self._validate_confidence(data.get('complexity_confidence', 0.5))
        
        ##Block purpose: Validate and normalize query type with confidence
        query_type = data.get('type', 'factual')
        if query_type not in ['factual', 'procedural', 'analytical', 'creative', 'conversational']:
            query_type = 'factual'
        type_confidence = self._validate_confidence(data.get('type_confidence', 0.5))
        
        ##Block purpose: Validate entities list
        entities = data.get('entities', [])
        if not isinstance(entities, list):
            entities = []
        
        ##Block purpose: Validate keywords list
        keywords = data.get('keywords', [])
        if not isinstance(keywords, list):
            keywords = []
        
        ##Block purpose: Parse and validate topics with categories
        topics = self._parse_topics(data.get('topics', []))
        
        ##Block purpose: Validate reformulations list
        reformulations = data.get('reformulations', [])
        if not isinstance(reformulations, list):
            reformulations = []
        reformulations = [r for r in reformulations if isinstance(r, str) and r.strip()]
        
        ##Block purpose: Calculate overall analysis confidence
        overall_confidence = self._calculate_overall_confidence(
            intent_confidence, complexity_confidence, type_confidence
        )
        
        return {
            'intent': intent,
            'intent_confidence': intent_confidence,
            'entities': entities,
            'complexity': complexity,
            'complexity_confidence': complexity_confidence,
            'type': query_type,
            'type_confidence': type_confidence,
            'keywords': keywords,
            'topics': topics,
            'reformulations': reformulations,
            'overall_confidence': overall_confidence,
            'entity_links': []  # Populated by entity linking step
        }
    
    ##Function purpose: Validate confidence score is within valid range
    def _validate_confidence(self, value: Any) -> float:
        """Validate and normalize confidence score to 0.0-1.0 range."""
        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            return 0.5
    
    ##Function purpose: Parse topics list from LLM response with validation
    def _parse_topics(self, topics_data: Any) -> List[Dict[str, Any]]:
        """Parse and validate topics from LLM response."""
        if not isinstance(topics_data, list):
            return []
        
        parsed_topics = []
        valid_categories = [e.value for e in TopicCategory]
        
        for topic_item in topics_data:
            if isinstance(topic_item, dict):
                topic_name = topic_item.get('topic', '')
                if not topic_name or not isinstance(topic_name, str):
                    continue
                
                category = topic_item.get('category', 'unknown')
                if category not in valid_categories:
                    category = 'unknown'
                
                confidence = self._validate_confidence(topic_item.get('confidence', 0.5))
                
                parsed_topics.append({
                    'topic': topic_name,
                    'category': category,
                    'confidence': confidence
                })
            elif isinstance(topic_item, str):
                ##Block purpose: Handle simple string topics
                parsed_topics.append({
                    'topic': topic_item,
                    'category': 'unknown',
                    'confidence': 0.5
                })
        
        return parsed_topics
    
    ##Function purpose: Calculate overall analysis confidence from component confidences
    def _calculate_overall_confidence(
        self, 
        intent_conf: float, 
        complexity_conf: float, 
        type_conf: float
    ) -> float:
        """Calculate weighted overall confidence score."""
        ##Block purpose: Use weighted average with intent having highest weight
        weights = {'intent': 0.4, 'complexity': 0.3, 'type': 0.3}
        overall = (
            intent_conf * weights['intent'] +
            complexity_conf * weights['complexity'] +
            type_conf * weights['type']
        )
        return round(overall, 3)
    
    ##Function purpose: Link extracted entities to Cortex nodes
    def _link_entities_to_cortex(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Search Cortex for nodes related to extracted entities."""
        if not self._cortex or not entities:
            return []
        
        entity_links = []
        
        for entity in entities:
            if not isinstance(entity, str) or not entity.strip():
                continue
            
            try:
                ##Block purpose: Search Cortex for matching nodes
                matching_nodes = self._search_cortex_for_entity(entity)
                
                if matching_nodes:
                    ##Block purpose: Take best match
                    best_match = matching_nodes[0]
                    entity_links.append({
                        'entity': entity,
                        'node_id': best_match.get('id'),
                        'node_type': best_match.get('type', 'unknown'),
                        'confidence': best_match.get('score', 0.0),
                        'relationship': 'related_to'
                    })
                else:
                    ##Block purpose: No match found - include as unlinked
                    entity_links.append({
                        'entity': entity,
                        'node_id': None,
                        'node_type': None,
                        'confidence': 0.0,
                        'relationship': 'unlinked'
                    })
                    
            except Exception:
                ##Block purpose: Skip entity on error
                entity_links.append({
                    'entity': entity,
                    'node_id': None,
                    'node_type': None,
                    'confidence': 0.0,
                    'relationship': 'error'
                })
        
        return entity_links
    
    ##Function purpose: Search Cortex for nodes matching an entity
    def _search_cortex_for_entity(self, entity: str) -> List[Dict[str, Any]]:
        """Search Cortex graph for nodes related to an entity."""
        if not self._cortex:
            return []
        
        matching_nodes = []
        entity_lower = entity.lower()
        
        try:
            ##Block purpose: Get all nodes from Cortex and search for matches
            if hasattr(self._cortex, 'get_all_nodes'):
                all_nodes = self._cortex.get_all_nodes(self._username)
            elif hasattr(self._cortex, 'cognitive_graph'):
                all_nodes = list(self._cortex.cognitive_graph.get(self._username, {}).values())
            else:
                return []
            
            for node in all_nodes:
                if not isinstance(node, dict):
                    continue
                
                ##Block purpose: Calculate match score based on content similarity
                score = self._calculate_entity_match_score(entity_lower, node)
                
                if score > 0.3:  # Minimum threshold for relevance
                    matching_nodes.append({
                        'id': node.get('id', ''),
                        'type': node.get('type', 'unknown'),
                        'content': node.get('content', '')[:100],
                        'score': score
                    })
            
            ##Block purpose: Sort by score descending
            matching_nodes.sort(key=lambda x: x['score'], reverse=True)
            
            return matching_nodes[:5]  # Return top 5 matches
            
        except Exception:
            return []
    
    ##Function purpose: Calculate match score between entity and Cortex node
    def _calculate_entity_match_score(self, entity: str, node: Dict[str, Any]) -> float:
        """Calculate relevance score between entity and node content."""
        score = 0.0
        
        ##Block purpose: Check content field
        content = str(node.get('content', '')).lower()
        if entity in content:
            score += 0.5
            ##Block purpose: Boost for exact word match
            if re.search(rf'\b{re.escape(entity)}\b', content):
                score += 0.3
        
        ##Block purpose: Check title/summary field if exists
        title = str(node.get('title', '')).lower()
        if entity in title:
            score += 0.4
        
        ##Block purpose: Check keywords if available
        keywords = node.get('keywords', [])
        if isinstance(keywords, list):
            for kw in keywords:
                if entity in str(kw).lower():
                    score += 0.2
                    break
        
        return min(1.0, score)
    
    ##Function purpose: Generate query reformulations using heuristics
    def generate_reformulations(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate alternative phrasings of the query based on analysis."""
        reformulations = []
        
        ##Block purpose: Check if reformulations already exist from LLM
        if analysis.get('reformulations'):
            return analysis['reformulations']
        
        if not self.reformulation_enabled:
            return []
        
        ##Block purpose: Generate heuristic-based reformulations
        query_type = analysis.get('type', 'factual')
        keywords = analysis.get('keywords', [])
        entities = analysis.get('entities', [])
        
        ##Block purpose: Type-based reformulation
        if query_type == 'question' and not query.endswith('?'):
            reformulations.append(query.rstrip('.!') + '?')
        
        ##Block purpose: Keyword-focused reformulation
        if keywords:
            keyword_str = ', '.join(keywords[:3])
            reformulations.append(f"Tell me about {keyword_str}")
        
        ##Block purpose: Entity-focused reformulation
        if entities:
            entity_str = ' and '.join(entities[:2])
            if 'how' in query.lower():
                reformulations.append(f"Explain how {entity_str} works")
            elif 'what' in query.lower():
                reformulations.append(f"What is {entity_str}")
            else:
                reformulations.append(f"Information about {entity_str}")
        
        ##Block purpose: Simplification reformulation for complex queries
        if analysis.get('complexity') in ['complex', 'very_complex']:
            if keywords:
                reformulations.append(f"Explain {keywords[0]} simply")
        
        return reformulations[:3]  # Return max 3 reformulations
    
    ##Function purpose: Get analysis summary as human-readable string
    def get_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable summary of query analysis."""
        intent = analysis.get('intent', 'unknown')
        intent_conf = analysis.get('intent_confidence', 0)
        complexity = analysis.get('complexity', 'unknown')
        query_type = analysis.get('type', 'unknown')
        overall_conf = analysis.get('overall_confidence', 0)
        
        topics = analysis.get('topics', [])
        topic_str = ', '.join([t.get('topic', '') for t in topics[:3]]) if topics else 'none identified'
        
        entity_count = len(analysis.get('entities', []))
        linked_count = sum(1 for el in analysis.get('entity_links', []) if el.get('node_id'))
        
        return (
            f"Intent: {intent} ({intent_conf:.0%} confident) | "
            f"Type: {query_type} | Complexity: {complexity} | "
            f"Topics: {topic_str} | "
            f"Entities: {entity_count} ({linked_count} linked) | "
            f"Overall confidence: {overall_conf:.0%}"
        )
    
    ##Function purpose: Generate default analysis when analysis is disabled or fails
    def _default_analysis(self, query: str) -> Dict[str, Any]:
        """Returns default analysis structure with C.2 enhancements."""
        
        ##Block purpose: Simple heuristic-based default analysis
        query_lower = query.lower()
        
        ##Block purpose: Detect intent from query patterns with confidence
        if query.startswith('/') or any(cmd in query_lower for cmd in ['do', 'make', 'create', 'show', 'list']):
            intent = 'command'
            intent_confidence = 0.7
        elif '?' in query:
            intent = 'question'
            intent_confidence = 0.8
        else:
            intent = 'conversation'
            intent_confidence = 0.5
        
        ##Block purpose: Estimate complexity from query length and structure
        word_count = len(query.split())
        if word_count < 5:
            complexity = 'simple'
            complexity_confidence = 0.8
        elif word_count < 15:
            complexity = 'moderate'
            complexity_confidence = 0.7
        elif word_count < 30:
            complexity = 'complex'
            complexity_confidence = 0.6
        else:
            complexity = 'very_complex'
            complexity_confidence = 0.6
        
        ##Block purpose: Infer type from query patterns
        query_type = 'factual'
        type_confidence = 0.5
        if any(word in query_lower for word in ['how to', 'steps', 'procedure']):
            query_type = 'procedural'
            type_confidence = 0.7
        elif any(word in query_lower for word in ['analyze', 'compare', 'evaluate']):
            query_type = 'analytical'
            type_confidence = 0.6
        elif any(word in query_lower for word in ['write', 'create', 'imagine', 'story']):
            query_type = 'creative'
            type_confidence = 0.6
        
        ##Block purpose: Extract simple topics from query
        topics = self._extract_default_topics(query)
        
        ##Block purpose: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            intent_confidence, complexity_confidence, type_confidence
        )
        
        return {
            'intent': intent,
            'intent_confidence': intent_confidence,
            'entities': [],
            'complexity': complexity,
            'complexity_confidence': complexity_confidence,
            'type': query_type,
            'type_confidence': type_confidence,
            'keywords': query.split()[:5],  # First 5 words as keywords
            'topics': topics,
            'reformulations': [],
            'overall_confidence': overall_confidence,
            'entity_links': []
        }
    
    ##Function purpose: Extract basic topics from query using heuristics
    def _extract_default_topics(self, query: str) -> List[Dict[str, Any]]:
        """Extract topics from query using simple heuristics."""
        topics = []
        query_lower = query.lower()
        
        ##Block purpose: Technical topic detection
        tech_keywords = ['code', 'programming', 'software', 'algorithm', 'api', 'database', 'system']
        if any(kw in query_lower for kw in tech_keywords):
            topics.append({'topic': 'technology', 'category': 'technical', 'confidence': 0.6})
        
        ##Block purpose: Personal topic detection
        personal_keywords = ['i', 'my', 'me', 'feel', 'think', 'believe', 'want']
        if any(kw in query_lower.split() for kw in personal_keywords):
            topics.append({'topic': 'personal', 'category': 'personal', 'confidence': 0.5})
        
        ##Block purpose: Procedural topic detection
        procedural_keywords = ['how to', 'steps', 'guide', 'tutorial', 'process']
        if any(kw in query_lower for kw in procedural_keywords):
            topics.append({'topic': 'procedures', 'category': 'procedural', 'confidence': 0.7})
        
        ##Block purpose: Temporal topic detection
        temporal_keywords = ['when', 'today', 'yesterday', 'tomorrow', 'last', 'next', 'time']
        if any(kw in query_lower for kw in temporal_keywords):
            topics.append({'topic': 'time-related', 'category': 'temporal', 'confidence': 0.6})
        
        if not topics:
            topics.append({'topic': 'general', 'category': 'unknown', 'confidence': 0.4})
        
        return topics
