##Script function and purpose: Context Relevance Scoring for The JENOVA Cognitive Architecture
##This module scores context items based on multiple relevance factors for prioritization

from typing import List, Dict, Any, Tuple, Optional
import re

##Class purpose: Scores context items for relevance and prioritization based on multiple factors
class ContextScorer:
    ##Function purpose: Initialize scorer with configuration and optional embedding model
    def __init__(self, config: Dict[str, Any], embedding_model: Optional[Any] = None) -> None:
        self.config = config
        self.embedding_model = embedding_model
        self.comprehension_config = config.get('comprehension', {})
        self.scoring_config = self.comprehension_config.get('context_scoring', {})
        self.enabled = self.scoring_config.get('enabled', True)
        
        ##Block purpose: Load scoring weights from configuration with defaults
        self.weights = self.scoring_config.get('weights', {
            'semantic_similarity': 0.4,
            'entity_overlap': 0.3,
            'keyword_match': 0.2,
            'query_type_match': 0.1
        })
    
    ##Function purpose: Score context items based on multiple factors and return sorted results
    def score_context(self, context_items: List[str], query: str, query_analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Returns context items with relevance scores, sorted by score (descending)."""
        
        ##Block purpose: Return unsorted items if scoring is disabled
        if not self.enabled:
            return [(item, 1.0) for item in context_items]
        
        scored_items = []
        
        ##Block purpose: Calculate relevance score for each context item
        for item in context_items:
            score = self._calculate_relevance_score(item, query, query_analysis)
            scored_items.append((item, score))
        
        ##Block purpose: Sort by relevance score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items
    
    ##Function purpose: Calculate multi-factor relevance score for a context item
    def _calculate_relevance_score(self, item: str, query: str, query_analysis: Dict[str, Any]) -> float:
        """Multi-factor relevance scoring combining multiple signals."""
        
        ##Block purpose: Calculate individual factor scores
        scores = {
            'semantic_similarity': self._semantic_similarity(item, query),
            'entity_overlap': self._entity_overlap(item, query_analysis.get('entities', [])),
            'keyword_match': self._keyword_match(item, query_analysis.get('keywords', [])),
            'query_type_match': self._query_type_match(item, query_analysis.get('type', 'factual'))
        }
        
        ##Block purpose: Weighted combination of scores
        total_score = sum(
            scores[factor] * self.weights.get(factor, 0) 
            for factor in scores
        )
        
        ##Block purpose: Normalize to 0-1 range
        return min(max(total_score, 0.0), 1.0)
    
    ##Function purpose: Calculate semantic similarity score between context item and query
    def _semantic_similarity(self, item: str, query: str) -> float:
        """Calculate semantic similarity using embeddings if available, otherwise use word overlap."""
        
        ##Block purpose: Use embedding model if available for accurate semantic similarity
        if self.embedding_model:
            try:
                item_embedding = self.embedding_model.encode([item], convert_to_numpy=True)[0]
                query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
                
                ##Block purpose: Calculate cosine similarity using numpy if available
                try:
                    import numpy as np
                    dot_product = np.dot(item_embedding, query_embedding)
                    norm_item = np.linalg.norm(item_embedding)
                    norm_query = np.linalg.norm(query_embedding)
                    
                    if norm_item > 0 and norm_query > 0:
                        similarity = dot_product / (norm_item * norm_query)
                        ##Block purpose: Normalize from [-1, 1] to [0, 1]
                        return (similarity + 1) / 2
                except ImportError:
                    ##Block purpose: Fallback to manual calculation if numpy not available
                    dot_product = sum(a * b for a, b in zip(item_embedding, query_embedding))
                    norm_item = sum(a * a for a in item_embedding) ** 0.5
                    norm_query = sum(b * b for b in query_embedding) ** 0.5
                    
                    if norm_item > 0 and norm_query > 0:
                        similarity = dot_product / (norm_item * norm_query)
                        return (similarity + 1) / 2
            except Exception:
                pass
        
        ##Block purpose: Fallback to word overlap similarity
        return self._word_overlap_similarity(item, query)
    
    ##Function purpose: Calculate word overlap similarity as fallback
    def _word_overlap_similarity(self, item: str, query: str) -> float:
        """Simple word overlap-based similarity."""
        
        ##Block purpose: Normalize and tokenize text
        item_words = set(re.findall(r'\b\w+\b', item.lower()))
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        if not query_words:
            return 0.0
        
        ##Block purpose: Calculate Jaccard similarity
        intersection = len(item_words & query_words)
        union = len(item_words | query_words)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    ##Function purpose: Calculate entity overlap score between context item and query entities
    def _entity_overlap(self, item: str, entities: List[str]) -> float:
        """Calculate overlap between context item and extracted entities."""
        
        if not entities:
            return 0.0
        
        item_lower = item.lower()
        matches = 0
        
        ##Block purpose: Count entity matches in context item
        for entity in entities:
            entity_lower = entity.lower()
            ##Block purpose: Check for whole word matches
            if re.search(r'\b' + re.escape(entity_lower) + r'\b', item_lower):
                matches += 1
        
        ##Block purpose: Normalize by total entity count
        return matches / len(entities) if entities else 0.0
    
    ##Function purpose: Calculate keyword match score
    def _keyword_match(self, item: str, keywords: List[str]) -> float:
        """Calculate keyword match score."""
        
        if not keywords:
            return 0.0
        
        item_lower = item.lower()
        matches = 0
        
        ##Block purpose: Count keyword matches in context item
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in item_lower:
                matches += 1
        
        ##Block purpose: Normalize by total keyword count
        return matches / len(keywords) if keywords else 0.0
    
    ##Function purpose: Calculate query type match score based on context item characteristics
    def _query_type_match(self, item: str, query_type: str) -> float:
        """Calculate score based on how well context item matches query type."""
        
        item_lower = item.lower()
        
        ##Block purpose: Type-specific heuristics
        if query_type == 'factual':
            ##Block purpose: Factual queries benefit from declarative statements
            factual_indicators = ['is', 'was', 'are', 'were', 'fact', 'information', 'data']
            return 1.0 if any(indicator in item_lower for indicator in factual_indicators) else 0.5
        
        elif query_type == 'procedural':
            ##Block purpose: Procedural queries benefit from step-by-step content
            procedural_indicators = ['step', 'procedure', 'process', 'how to', 'method', 'way']
            return 1.0 if any(indicator in item_lower for indicator in procedural_indicators) else 0.5
        
        elif query_type == 'analytical':
            ##Block purpose: Analytical queries benefit from comparative or evaluative content
            analytical_indicators = ['compare', 'analyze', 'evaluate', 'difference', 'similarity', 'why']
            return 1.0 if any(indicator in item_lower for indicator in analytical_indicators) else 0.5
        
        elif query_type == 'creative':
            ##Block purpose: Creative queries benefit from diverse, open-ended content
            return 0.7  # Neutral score for creative queries
        
        else:
            ##Block purpose: Default score for conversational or unknown types
            return 0.5
