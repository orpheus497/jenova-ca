##Script function and purpose: Query Analysis Module for The JENOVA Cognitive Architecture
##This module analyzes user queries to extract intent, entities, complexity, and type for enhanced comprehension

import json
from typing import Dict, List, Any, Optional
from enum import Enum
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

##Class purpose: Analyzes user queries to extract structured information for enhanced comprehension
class QueryAnalyzer:
    ##Function purpose: Initialize query analyzer with LLM interface and configuration
    def __init__(self, llm: Any, config: Dict[str, Any], json_grammar: Optional[Any] = None) -> None:
        self.llm = llm
        self.config = config
        self.json_grammar = json_grammar
        self.comprehension_config = config.get('comprehension', {})
        self.query_analysis_config = self.comprehension_config.get('query_analysis', {})
        self.enabled = self.query_analysis_config.get('enabled', True)
    
    ##Function purpose: Analyze query to extract intent, entities, complexity, and type
    def analyze(self, query: str) -> Dict[str, Any]:
        """Comprehensive query analysis returning structured information."""
        
        ##Block purpose: Return default analysis if query analysis is disabled
        if not self.enabled:
            return self._default_analysis(query)
        
        ##Block purpose: Build analysis prompt with query and analysis requirements
        intent_options = [e.value for e in QueryIntent]
        complexity_options = [e.value for e in QueryComplexity]
        
        prompt = f"""Analyze the following user query and extract structured information:

1. Intent: Classify the query intent as one of: {intent_options}
2. Entities: Extract a list of key entities (people, places, concepts, temporal references, objects)
3. Complexity: Assess complexity as one of: {complexity_options}
4. Query type: Classify as one of: factual, procedural, analytical, creative, conversational
5. Keywords: Extract 3-5 key terms or phrases

Query: "{query}"

Respond with a valid JSON object:
{{
    "intent": "<one of {intent_options}>",
    "entities": ["entity1", "entity2", ...],
    "complexity": "<one of {complexity_options}>",
    "type": "<factual|procedural|analytical|creative|conversational>",
    "keywords": ["keyword1", "keyword2", ...]
}}"""
        
        ##Block purpose: Generate analysis using LLM with JSON grammar
        try:
            analysis_json_str = self.llm.generate(
                prompt, 
                temperature=0.2, 
                grammar=self.json_grammar
            )
            
            ##Block purpose: Parse and validate analysis JSON
            analysis_data = self._parse_analysis(analysis_json_str)
            return analysis_data
            
        except Exception as e:
            ##Block purpose: Fallback to default analysis on error
            if hasattr(self, 'file_logger'):
                self.file_logger.log_error(f"Error during query analysis: {e}")
            return self._default_analysis(query)
    
    ##Function purpose: Parse LLM JSON response into structured format with validation
    def _parse_analysis(self, json_str: str) -> Dict[str, Any]:
        """Parse and validate analysis JSON response."""
        
        ##Block purpose: Extract JSON from potentially wrapped response
        try:
            data = extract_json(json_str)
        except (ValueError, json.JSONDecodeError):
            ##Block purpose: Fallback to direct JSON parsing
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return self._default_analysis("")
        
        ##Block purpose: Validate and normalize analysis data
        intent = data.get('intent', 'question')
        if intent not in [e.value for e in QueryIntent]:
            intent = 'question'
        
        complexity = data.get('complexity', 'simple')
        if complexity not in [e.value for e in QueryComplexity]:
            complexity = 'simple'
        
        query_type = data.get('type', 'factual')
        if query_type not in ['factual', 'procedural', 'analytical', 'creative', 'conversational']:
            query_type = 'factual'
        
        entities = data.get('entities', [])
        if not isinstance(entities, list):
            entities = []
        
        keywords = data.get('keywords', [])
        if not isinstance(keywords, list):
            keywords = []
        
        return {
            'intent': intent,
            'entities': entities,
            'complexity': complexity,
            'type': query_type,
            'keywords': keywords
        }
    
    ##Function purpose: Generate default analysis when analysis is disabled or fails
    def _default_analysis(self, query: str) -> Dict[str, Any]:
        """Returns default analysis structure."""
        
        ##Block purpose: Simple heuristic-based default analysis
        query_lower = query.lower()
        
        ##Block purpose: Detect intent from query patterns
        if query.startswith('/') or any(cmd in query_lower for cmd in ['do', 'make', 'create', 'show', 'list']):
            intent = 'command'
        elif '?' in query:
            intent = 'question'
        else:
            intent = 'conversation'
        
        ##Block purpose: Estimate complexity from query length and structure
        word_count = len(query.split())
        if word_count < 5:
            complexity = 'simple'
        elif word_count < 15:
            complexity = 'moderate'
        elif word_count < 30:
            complexity = 'complex'
        else:
            complexity = 'very_complex'
        
        return {
            'intent': intent,
            'entities': [],
            'complexity': complexity,
            'type': 'factual',
            'keywords': query.split()[:5]  # First 5 words as keywords
        }
