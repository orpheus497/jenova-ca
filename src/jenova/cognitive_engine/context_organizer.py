##Script function and purpose: Context Organization for The JENOVA Cognitive Architecture
##This module organizes context into categories and matrix structures for improved organization

from typing import List, Dict, Any, Optional
from collections import defaultdict
import json
from jenova.utils.grammar_loader import load_json_grammar

##Class purpose: Organizes context into structured categories and matrices
class ContextOrganizer:
    ##Function purpose: Initialize organizer with LLM and config
    def __init__(self, llm: Any, config: Dict[str, Any], file_logger: Any) -> None:
        self.llm = llm
        self.config = config
        self.file_logger = file_logger
        self.organizing_config = config.get('organizing', {})
        self.context_org_config = self.organizing_config.get('context_organization', {})
        self.enabled = self.context_org_config.get('enabled', True)
        self.categorization_enabled = self.context_org_config.get('categorization', True)
        self.tier_classification_enabled = self.context_org_config.get('tier_classification', True)
        
        ##Block purpose: Load JSON grammar using centralized utility
        self.json_grammar = load_json_grammar(file_logger=file_logger)
    
    ##Function purpose: Organize context into categorized matrix
    def organize_context(self, context_items: List[str], query: str) -> Dict[str, Any]:
        """Organizes context into topic-based categories and relevance tiers."""
        
        ##Block purpose: Return flat list if organization is disabled (backward compatibility)
        if not self.enabled:
            return {
                'categorized': {},
                'tiers': {'high': [], 'medium': [], 'low': []},
                'matrix': {'high_priority': [], 'medium_priority': [], 'low_priority': [], 'structure': 'flat'},
                'flat_list': context_items
            }
        
        if not context_items:
            return {
                'categorized': {},
                'tiers': {'high': [], 'medium': [], 'low': []},
                'matrix': {'high_priority': [], 'medium_priority': [], 'low_priority': [], 'structure': 'flat'},
                'flat_list': []
            }
        
        ##Block purpose: Categorize context items by topic if enabled
        if self.categorization_enabled:
            categorized = self._categorize_by_topic(context_items, query)
        else:
            ##Block purpose: Create single category if categorization disabled
            categorized = {'general': context_items}
        
        ##Block purpose: Classify by relevance tier if enabled
        if self.tier_classification_enabled:
            tiered = self._classify_relevance_tiers(categorized, query)
        else:
            ##Block purpose: Put all items in medium tier if tier classification disabled
            tiered = {
                'high': [],
                'medium': list(categorized.values())[0] if categorized else [],
                'low': []
            }
        
        ##Block purpose: Create context matrix structure
        matrix = self._create_context_matrix(tiered)
        
        return {
            'categorized': categorized,
            'tiers': tiered,
            'matrix': matrix,
            'flat_list': context_items  # Backward compatibility
        }
    
    ##Function purpose: Categorize context items by topic
    def _categorize_by_topic(self, items: List[str], query: str) -> Dict[str, List[str]]:
        """Groups context items by topic/category using LLM."""
        
        ##Block purpose: Use simple keyword-based categorization for small item sets
        if len(items) <= 3:
            return self._simple_categorization(items, query)
        
        try:
            ##Block purpose: Create prompt for LLM-based categorization
            items_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])
            prompt = f"""Categorize the following context items by topic. Group related items together.
Each item should be assigned to exactly one category. Use concise category names (1-3 words).

Query: "{query}"

Context Items:
{items_str}

Respond with a valid JSON object where keys are category names and values are arrays of item numbers (1-indexed).
Example: {{"Python Programming": [1, 3], "Data Structures": [2, 4]}}

JSON Response:"""
            
            ##Block purpose: Generate categorization using LLM
            response = self.llm.generate(prompt, temperature=0.3, grammar=self.json_grammar)
            
            ##Block purpose: Parse JSON response
            try:
                from jenova.utils.json_parser import extract_json
                categories_json = extract_json(response)
                categories_dict = json.loads(categories_json)
                
                ##Block purpose: Convert item numbers back to actual items
                categorized = {}
                for category, item_indices in categories_dict.items():
                    categorized[category] = []
                    for idx in item_indices:
                        if isinstance(idx, int) and 1 <= idx <= len(items):
                            categorized[category].append(items[idx - 1])
                
                ##Block purpose: Ensure all items are categorized (handle any missed items)
                categorized_items = set()
                for category_items in categorized.values():
                    categorized_items.update(category_items)
                
                uncategorized = [item for item in items if item not in categorized_items]
                if uncategorized:
                    categorized['Other'] = uncategorized
                
                self.file_logger.log_info(f"Categorized {len(items)} items into {len(categorized)} categories")
                return categorized
                
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                self.file_logger.log_error(f"Error parsing categorization JSON: {e}")
                return self._simple_categorization(items, query)
                
        except Exception as e:
            self.file_logger.log_error(f"Error during topic categorization: {e}")
            return self._simple_categorization(items, query)
    
    ##Function purpose: Simple keyword-based categorization fallback
    def _simple_categorization(self, items: List[str], query: str) -> Dict[str, List[str]]:
        """Fallback categorization using keyword matching."""
        
        ##Block purpose: Extract keywords from query
        query_words = set(query.lower().split())
        
        ##Block purpose: Group items by keyword overlap
        categorized = defaultdict(list)
        for item in items:
            item_words = set(item.lower().split())
            overlap = query_words & item_words
            
            if overlap:
                ##Block purpose: Use most common overlapping word as category
                category = max(overlap, key=lambda w: query.lower().count(w))
                categorized[category.title()].append(item)
            else:
                categorized['Other'].append(item)
        
        return dict(categorized)
    
    ##Function purpose: Classify items by relevance tier
    def _classify_relevance_tiers(self, categorized: Dict[str, List[str]], query: str) -> Dict[str, List[str]]:
        """Classifies items into high/medium/low relevance tiers."""
        
        tiers = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        ##Block purpose: Classify each category based on relevance heuristics
        for category, items in categorized.items():
            category_lower = category.lower()
            
            ##Block purpose: Calculate category relevance score
            category_words = set(category_lower.split())
            word_overlap = len(query_words & category_words)
            category_size = len(items)
            
            ##Block purpose: High tier: Direct matches, small focused categories
            if word_overlap > 0 and category_size <= 3:
                tiers['high'].extend(items)
            ##Block purpose: Medium tier: Related categories, moderate size
            elif word_overlap > 0 or category_size <= 5:
                tiers['medium'].extend(items)
            ##Block purpose: Low tier: Distant categories, large categories
            else:
                tiers['low'].extend(items)
        
        self.file_logger.log_info(f"Classified context into tiers: High={len(tiers['high'])}, Medium={len(tiers['medium'])}, Low={len(tiers['low'])}")
        return tiers
    
    ##Function purpose: Create context matrix structure
    def _create_context_matrix(self, tiered: Dict[str, List[str]]) -> Dict[str, Any]:
        """Creates hierarchical matrix structure."""
        return {
            'high_priority': tiered.get('high', []),
            'medium_priority': tiered.get('medium', []),
            'low_priority': tiered.get('low', []),
            'structure': 'hierarchical'
        }
