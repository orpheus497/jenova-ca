##Script function and purpose: Cognitive Cortex Integration Layer for The JENOVA Cognitive Architecture
##This module provides unified knowledge representation and coordinates feedback loops between Memory and Cortex systems

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from jenova.utils.cache import CacheManager

##Class purpose: Coordinates integration between Memory systems and Cortex graph
class IntegrationLayer:
    ##Function purpose: Initialize integration layer with Memory and Cortex references
    def __init__(self, cortex: Any, memory_search: Any, config: Dict[str, Any], file_logger: Any, cache_manager: Optional[CacheManager] = None) -> None:
        self.cortex = cortex
        self.memory_search = memory_search
        self.config = config
        self.file_logger = file_logger
        self.integration_enabled = config.get('cortex', {}).get('integration', {}).get('enabled', True)
        self.cache_manager = cache_manager  # Optional cache manager for performance optimization
        
    ##Function purpose: Find Cortex nodes semantically related to a memory item
    def find_related_cortex_nodes(self, memory_content: str, username: str, max_nodes: int = 5) -> List[Dict[str, Any]]:
        """Finds Cortex nodes semantically related to a memory item using content similarity."""
        if not self.integration_enabled or not self.cortex:
            return []
        
        ##Block purpose: Check cache first if cache manager is available
        if self.cache_manager:
            cache_key = CacheManager.make_key('node_search', username, memory_content[:100], max_nodes)
            cached_result = self.cache_manager.node_search_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        ##Block purpose: Get all user nodes from Cortex
        user_nodes = [n for n in self.cortex.graph["nodes"].values() if n.get("user") == username]
        
        if not user_nodes:
            return []
        
        ##Block purpose: Use semantic memory to find similar nodes
        try:
            node_contents = [node['content'] for node in user_nodes]
            similar_nodes = self.memory_search.semantic_memory.search_documents(
                query=memory_content,
                documents=node_contents,
                n_results=max_nodes
            )
            
            ##Block purpose: Map similarity results back to full node data
            result_nodes = []
            for content, distance in similar_nodes:
                # Find the node with matching content
                for node in user_nodes:
                    if node['content'] == content:
                        result_nodes.append({
                            'node': node,
                            'similarity_score': 1.0 / (1.0 + distance),
                            'distance': distance
                        })
                        break
            
            ##Block purpose: Cache result if cache manager is available
            if self.cache_manager:
                cache_key = CacheManager.make_key('node_search', username, memory_content[:100], max_nodes)
                self.cache_manager.node_search_cache.set(cache_key, result_nodes)
            
            return result_nodes
        except Exception as e:
            self.file_logger.log_error(f"Error finding related Cortex nodes: {e}")
            return []
    
    ##Function purpose: Get centrality score for memory content from Cortex
    def get_centrality_score(self, memory_content: str, username: str) -> float:
        """Retrieves centrality score from Cortex graph for memory content."""
        if not self.integration_enabled or not self.cortex:
            return 0.0
        
        ##Block purpose: Check cache first if cache manager is available
        if self.cache_manager:
            cache_key = CacheManager.make_key('centrality', username, memory_content[:100])
            cached_score = self.cache_manager.centrality_cache.get(cache_key)
            if cached_score is not None:
                return cached_score
        
        ##Block purpose: Find related nodes and calculate average centrality
        related_nodes = self.find_related_cortex_nodes(memory_content, username, max_nodes=3)
        
        if not related_nodes:
            return 0.0
        
        ##Block purpose: Calculate weighted average centrality
        total_centrality = 0.0
        total_weight = 0.0
        
        for item in related_nodes:
            node = item['node']
            similarity = item['similarity_score']
            centrality = node.get('metadata', {}).get('centrality', 0.0)
            
            total_centrality += centrality * similarity
            total_weight += similarity
        
        if total_weight == 0:
            return 0.0
        
        avg_centrality = total_centrality / total_weight
        
        ##Block purpose: Normalize to 0-1 range (assuming max centrality ~10)
        normalized = min(avg_centrality / 10.0, 1.0)
        
        ##Block purpose: Cache result if cache manager is available
        if self.cache_manager:
            cache_key = CacheManager.make_key('centrality', username, memory_content[:100])
            self.cache_manager.centrality_cache.set(cache_key, normalized)
        
        return normalized
    
    ##Function purpose: Expand context using Cortex relationships
    def expand_context_with_relationships(self, context_items: List[str], username: str, max_expansion: int = 3) -> List[str]:
        """Expands context by following Cortex graph relationships."""
        if not self.integration_enabled or not self.cortex:
            return context_items
        
        expanded_context = list(context_items)
        added_content = set(context_items)
        
        ##Block purpose: For each context item, find related nodes and add their content
        for item in context_items[:5]:  # Limit initial items to avoid explosion
            related_nodes = self.find_related_cortex_nodes(item, username, max_nodes=max_expansion)
            
            for node_item in related_nodes:
                node = node_item['node']
                node_content = node.get('content', '')
                
                ##Block purpose: Add node content if not already present
                if node_content and node_content not in added_content:
                    expanded_context.append(node_content)
                    added_content.add(node_content)
                    
                    if len(expanded_context) >= len(context_items) + max_expansion:
                        break
            
            if len(expanded_context) >= len(context_items) + max_expansion:
                break
        
        return expanded_context
    
    ##Function purpose: Create unified knowledge representation
    def create_unified_knowledge_map(self, username: str) -> Dict[str, Any]:
        """Creates a unified representation combining Memory and Cortex knowledge."""
        if not self.integration_enabled:
            return {}
        
        knowledge_map = {
            'memory_items': [],
            'cortex_nodes': [],
            'cross_references': [],
            'knowledge_gaps': []
        }
        
        ##Block purpose: Sample memory items
        try:
            sample_query = "general knowledge"
            memory_items = self.memory_search.search_all(sample_query, username)
            knowledge_map['memory_items'] = memory_items[:10]  # Limit to 10
        except Exception as e:
            self.file_logger.log_error(f"Error sampling memory items: {e}")
        
        ##Block purpose: Get Cortex nodes
        try:
            cortex_nodes = [n for n in self.cortex.graph["nodes"].values() if n.get("user") == username]
            knowledge_map['cortex_nodes'] = cortex_nodes[:20]  # Limit to 20
        except Exception as e:
            self.file_logger.log_error(f"Error getting Cortex nodes: {e}")
        
        ##Block purpose: Find cross-references between Memory and Cortex
        try:
            for memory_item in knowledge_map['memory_items'][:5]:  # Limit to avoid performance issues
                related_nodes = self.find_related_cortex_nodes(memory_item, username, max_nodes=2)
                if related_nodes:
                    knowledge_map['cross_references'].append({
                        'memory_item': memory_item,
                        'related_nodes': [n['node']['id'] for n in related_nodes]
                    })
        except Exception as e:
            self.file_logger.log_error(f"Error finding cross-references: {e}")
        
        return knowledge_map
    
    ##Function purpose: Provide feedback from Memory to Cortex
    def feedback_memory_to_cortex(self, memory_content: str, memory_type: str, username: str) -> Optional[str]:
        """Provides feedback from Memory system to Cortex, creating connections if needed."""
        if not self.integration_enabled or not self.cortex:
            return None
        
        ##Block purpose: Find related Cortex nodes
        related_nodes = self.find_related_cortex_nodes(memory_content, username, max_nodes=3)
        
        if not related_nodes:
            return None
        
        ##Block purpose: Create memory reference node in Cortex if highly related
        best_match = related_nodes[0]
        if best_match['similarity_score'] > 0.7:  # Threshold for strong relationship
            try:
                memory_node_id = self.cortex.add_node(
                    node_type='memory_reference',
                    content=f"Memory reference: {memory_content[:100]}...",
                    user=username,
                    linked_to=[best_match['node']['id']],
                    metadata={
                        'memory_type': memory_type,
                        'similarity_score': best_match['similarity_score']
                    }
                )
                self.file_logger.log_info(f"Created memory reference node {memory_node_id} linked to Cortex node {best_match['node']['id']}")
                return memory_node_id
            except Exception as e:
                self.file_logger.log_error(f"Error creating memory reference node: {e}")
        
        return None
    
    ##Function purpose: Provide feedback from Cortex to Memory
    def feedback_cortex_to_memory(self, cortex_node_id: str, username: str) -> None:
        """Provides feedback from Cortex to Memory system, enhancing memory with graph context."""
        if not self.integration_enabled or not self.cortex:
            return
        
        ##Block purpose: Get Cortex node
        node = self.cortex.get_node(cortex_node_id)
        if not node:
            return
        
        ##Block purpose: Get related nodes from graph
        related_node_ids = [
            link['target'] for link in self.cortex.graph['links']
            if link['source'] == cortex_node_id
        ]
        
        ##Block purpose: Enhance memory with relationship context
        # This could be used to update memory metadata with relationship information
        # For now, we log the relationship information
        self.file_logger.log_info(
            f"Cortex node {cortex_node_id} has {len(related_node_ids)} related nodes. "
            f"Centrality: {node.get('metadata', {}).get('centrality', 0)}"
        )
    
    ##Function purpose: Check knowledge consistency between Memory and Cortex
    def check_knowledge_consistency(self, username: str) -> Dict[str, Any]:
        """Checks for consistency and gaps between Memory and Cortex knowledge."""
        if not self.integration_enabled:
            return {'consistent': True, 'gaps': []}
        
        consistency_report = {
            'consistent': True,
            'gaps': [],
            'duplications': [],
            'recommendations': []
        }
        
        ##Block purpose: Sample memory and Cortex to find inconsistencies
        try:
            sample_memories = self.memory_search.search_all("general knowledge", username)[:5]
            cortex_nodes = [n for n in self.cortex.graph["nodes"].values() if n.get("user") == username][:10]
            
            ##Block purpose: Check for knowledge gaps (high centrality nodes not in memory)
            high_centrality_nodes = [
                n for n in cortex_nodes
                if n.get('metadata', {}).get('centrality', 0) > 2.0
            ]
            
            for node in high_centrality_nodes:
                related_memories = self.find_related_cortex_nodes(node['content'], username, max_nodes=1)
                if not related_memories or related_memories[0]['similarity_score'] < 0.5:
                    consistency_report['gaps'].append({
                        'type': 'high_centrality_node_not_in_memory',
                        'node_id': node['id'],
                        'content': node['content'][:100]
                    })
            
            ##Block purpose: Check for duplications
            for memory in sample_memories:
                related_nodes = self.find_related_cortex_nodes(memory, username, max_nodes=1)
                if related_nodes and related_nodes[0]['similarity_score'] > 0.9:
                    consistency_report['duplications'].append({
                        'memory_content': memory[:100],
                        'related_node_id': related_nodes[0]['node']['id']
                    })
            
            ##Block purpose: Generate recommendations
            if consistency_report['gaps']:
                consistency_report['recommendations'].append(
                    "Consider adding high-centrality Cortex nodes to memory for better retrieval"
                )
            
            if consistency_report['duplications']:
                consistency_report['recommendations'].append(
                    "Consider consolidating duplicate knowledge between Memory and Cortex"
                )
            
            consistency_report['consistent'] = len(consistency_report['gaps']) == 0 and len(consistency_report['duplications']) == 0
            
        except Exception as e:
            self.file_logger.log_error(f"Error checking knowledge consistency: {e}")
        
        return consistency_report
