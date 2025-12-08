##Script function and purpose: Memory Search for The JENOVA Cognitive Architecture
##This module coordinates searches across all memory types and provides unified search interface

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from jenova.memory.semantic import SemanticMemory
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.cognitive_engine.context_scorer import ContextScorer
from jenova.cognitive_engine.context_organizer import ContextOrganizer
from jenova.utils.cache import CacheManager

##Class purpose: Coordinates searches across all memory types and provides unified interface
class MemorySearch:
    ##Function purpose: Initialize memory search with all memory systems
    def __init__(self, semantic_memory: SemanticMemory, episodic_memory: EpisodicMemory, procedural_memory: ProceduralMemory, config: Dict[str, Any], file_logger: Any, cortex: Optional[Any] = None, integration_layer: Optional[Any] = None, llm: Optional[Any] = None, embedding_model: Optional[Any] = None, cache_manager: Optional[CacheManager] = None) -> None:
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.insight_manager = None # Will be set later
        self.config = config
        self.file_logger = file_logger
        self.cortex = cortex  # Optional Cortex integration for centrality scoring
        self.integration_layer = integration_layer  # Integration layer for unified knowledge representation
        self.llm = llm  # Optional LLM for context organization
        self.embedding_model = embedding_model  # Optional embedding model for context scoring
        self.cache_manager = cache_manager  # Optional cache manager for performance optimization
        
        ##Block purpose: Initialize context scorer for relevance scoring
        self.context_scorer = ContextScorer(config, embedding_model)
        
        ##Block purpose: Initialize context organizer if LLM is available
        self.context_organizer = None
        if llm:
            self.context_organizer = ContextOrganizer(llm, config, file_logger)

        if self.config.get('memory', {}).get('preload_memories', False):
            self._preload_memories()

    ##Function purpose: Pre-load memory collections into RAM for faster access (performance optimization)
    def _preload_memories(self) -> None:
        self.file_logger.log_info("Pre-loading memories into RAM...")
        try:
            import threading
            threads = []
            collections = [self.semantic_memory.collection, self.episodic_memory.collection, self.procedural_memory.collection]
            for collection in collections:
                thread = threading.Thread(target=collection.get)
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            self.file_logger.log_info("Memories pre-loaded successfully.")
        except Exception as e:
            self.file_logger.log_error(f"Error pre-loading memories: {e}")

    ##Function purpose: Search all memory types and return unified, ranked results with enhanced scoring
    def search_all(self, query: str, username: str, query_analysis: Optional[Dict[str, Any]] = None) -> List[str]:
        self.file_logger.log_info(f"Searching all memories for user '{username}' with query: '{query}'")
        memory_search_config = self.config.get('memory_search', {})
        semantic_n_results = memory_search_config.get('semantic_n_results', 5)
        episodic_n_results = memory_search_config.get('episodic_n_results', 3)
        procedural_n_results = memory_search_config.get('procedural_n_results', 3)
        insight_n_results = memory_search_config.get('insight_n_results', 5)

        try:
            # Retrieve from structured memories
            semantic_results = self.semantic_memory.search_collection(query, username, n_results=semantic_n_results)
            self.file_logger.log_info(f"Found {len(semantic_results)} semantic results.")
        except Exception as e:
            self.file_logger.log_error(f"Error during semantic memory search: {e}")
            semantic_results = []

        try:
            episodic_results = self.episodic_memory.recall_relevant_episodes(query, username, n_results=episodic_n_results)
            self.file_logger.log_info(f"Found {len(episodic_results)} episodic results.")
        except Exception as e:
            self.file_logger.log_error(f"Error during episodic memory search: {e}")
            episodic_results = []

        try:
            procedural_results = self.procedural_memory.search(query, username, n_results=procedural_n_results)
            self.file_logger.log_info(f"Found {len(procedural_results)} procedural results.")
        except Exception as e:
            self.file_logger.log_error(f"Error during procedural memory search: {e}")
            procedural_results = []
        
        # Retrieve relevant learned insights
        try:
            insight_results = self.search_insights(query, username, max_insights=insight_n_results)
            self.file_logger.log_info(f"Found {len(insight_results)} relevant insights.")
        except Exception as e:
            self.file_logger.log_error(f"Error during insight search: {e}")
            insight_results = []

        ##Block purpose: Use multi-factor ranking if enabled, otherwise use simple distance-based ranking
        organizing_config = self.config.get('organizing', {})
        ranking_enabled = organizing_config.get('ranking', {}).get('enabled', False)
        
        if ranking_enabled:
            ##Block purpose: Apply multi-factor ranking algorithm
            ranked_results = self._rank_memory_results(
                semantic_results, episodic_results, procedural_results, 
                insight_results, query, username
            )
            ranked_docs = [doc for doc, score, metadata in ranked_results]
        else:
            ##Block purpose: Fallback to simple distance-based ranking (backward compatibility)
            vector_results = semantic_results + episodic_results + procedural_results
            vector_results.sort(key=lambda x: x[1])
            ranked_docs = [doc for doc, dist in vector_results]
            ##Block purpose: Prioritize insights (extract documents from tuples)
            insight_docs = [doc for doc, dist in insight_results] if insight_results else []
            ranked_docs = insight_docs + ranked_docs

        ##Block purpose: Expand context using Cortex relationships if integration layer is available
        integration_config = self.config.get('cortex', {}).get('integration', {})
        if integration_config.get('relationship_aware_retrieval', False) and self.integration_layer:
            try:
                expanded_context = self.integration_layer.expand_context_with_relationships(
                    ranked_docs, username, max_expansion=integration_config.get('max_expansion', 3)
                )
                ranked_docs = expanded_context
                self.file_logger.log_info(f"Expanded context with Cortex relationships: {len(ranked_docs)} items")
            except Exception as e:
                self.file_logger.log_error(f"Error expanding context with relationships: {e}")

        ##Block purpose: Apply context scoring if query analysis is available for enhanced relevance
        if query_analysis is not None:
            scored_context = self.context_scorer.score_context(ranked_docs, query, query_analysis)
            ranked_docs = [item for item, score in scored_context]
            self.file_logger.log_info(f"Applied context scoring. Top scores: {[f'{score:.2f}' for _, score in scored_context[:3]]}")

        ##Block purpose: Organize context into categories and tiers if context organizer is available
        organized_context = None
        if self.context_organizer:
            try:
                organized_context = self.context_organizer.organize_context(ranked_docs, query)
                ##Block purpose: Use organized context matrix for prioritized retrieval
                if organized_context.get('matrix', {}).get('structure') == 'hierarchical':
                    ##Block purpose: Prioritize high-priority items, then medium, then low
                    matrix = organized_context['matrix']
                    ranked_docs = matrix.get('high_priority', []) + matrix.get('medium_priority', []) + matrix.get('low_priority', [])
                    self.file_logger.log_info(f"Organized context: High={len(matrix.get('high_priority', []))}, Medium={len(matrix.get('medium_priority', []))}, Low={len(matrix.get('low_priority', []))}")
            except Exception as e:
                self.file_logger.log_error(f"Error organizing context: {e}")
                ##Block purpose: Fallback to flat list on error
                organized_context = None

        self.file_logger.log_info(f"Final context length: {len(ranked_docs)}")
        
        return ranked_docs[:10] # Return a combined list of the most relevant context

    ##Function purpose: Search insights using semantic similarity
    def search_insights(self, query: str, username: str, max_insights: int = 3) -> List[Tuple[str, float]]:
        """Uses semantic search to find the most relevant insights for a given query."""
        self.file_logger.log_info(f"Searching insights for user '{username}' with query: '{query}'")
        all_insights = self.insight_manager.get_all_insights(username)
        if not all_insights:
            self.file_logger.log_info("No insights found for user.")
            return []
        self.file_logger.log_info(f"Found {len(all_insights)} total insights for user.")

        insight_contents = [f"Learned Insight on '{insight['topic']}': {insight['content']}" for insight in all_insights]
        
        try:
            relevant_insights = self.semantic_memory.search_documents(query, documents=insight_contents, n_results=max_insights)
            self.file_logger.log_info(f"Found {len(relevant_insights)} relevant insights after semantic search.")
        except Exception as e:
            self.file_logger.log_error(f"Error during insight semantic search: {e}")
            return []
        
        return relevant_insights  # Return (doc, distance) tuples for consistency

    ##Function purpose: Rank memory results using multi-factor algorithm with batch optimization
    def _rank_memory_results(self, semantic_results: List[Tuple[str, float]], 
                            episodic_results: List[Tuple[str, float]], 
                            procedural_results: List[Tuple[str, float]],
                            insight_results: List[Tuple[str, float]],
                            query: str, username: str) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Multi-factor ranking combining relevance, recency, centrality, type, confidence."""
        
        ranking_config = self.config.get('organizing', {}).get('ranking', {})
        weights = ranking_config.get('factors', {
            'relevance_weight': 0.4,
            'recency_weight': 0.2,
            'centrality_weight': 0.2,
            'type_weight': 0.1,
            'confidence_weight': 0.1
        })
        
        ##Block purpose: Cleanup cache if needed (periodic maintenance)
        if self.cache_manager:
            self.cache_manager.cleanup_if_needed()
        
        all_results = []
        
        ##Block purpose: Batch process all results for better cache utilization
        all_docs_to_score = []
        all_docs_to_score.extend([(doc, distance, 'semantic') for doc, distance in semantic_results])
        all_docs_to_score.extend([(doc, distance, 'episodic') for doc, distance in episodic_results])
        all_docs_to_score.extend([(doc, distance, 'procedural') for doc, distance in procedural_results])
        all_docs_to_score.extend([(doc, distance, 'insight') for doc, distance in insight_results])
        
        ##Block purpose: Process all documents in batch
        for doc, distance, mem_type in all_docs_to_score:
            score = self._calculate_combined_score(
                doc, distance, mem_type, query, username, weights
            )
            all_results.append((doc, score, {'type': mem_type, 'distance': distance}))
        
        ##Block purpose: Sort by combined score (descending - higher is better)
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    ##Function purpose: Calculate combined score from multiple factors
    def _calculate_combined_score(self, doc: str, distance: float, mem_type: str, 
                                 query: str, username: str, weights: Dict[str, float]) -> float:
        """Calculate weighted combination of relevance, recency, centrality, type, confidence."""
        
        ##Block purpose: Normalize distance to relevance score (0-1, higher is better)
        relevance_score = 1.0 / (1.0 + distance)
        
        ##Block purpose: Get recency score (if available from metadata)
        recency_score = self._get_recency_score(doc, mem_type, username)
        
        ##Block purpose: Get centrality score from Cortex
        centrality_score = self._get_centrality_score(doc, username)
        
        ##Block purpose: Get type priority score
        memory_prioritization = self.config.get('organizing', {}).get('memory_prioritization', {
            'insight': 1.0,
            'semantic': 0.8,
            'episodic': 0.6,
            'procedural': 0.5
        })
        type_priority = memory_prioritization.get(mem_type, 0.5)
        
        ##Block purpose: Get confidence score (for semantic memories)
        confidence_score = self._get_confidence_score(doc, mem_type, username)
        
        ##Block purpose: Weighted combination of all factors
        combined_score = (
            relevance_score * weights.get('relevance_weight', 0.4) +
            recency_score * weights.get('recency_weight', 0.2) +
            centrality_score * weights.get('centrality_weight', 0.2) +
            type_priority * weights.get('type_weight', 0.1) +
            confidence_score * weights.get('confidence_weight', 0.1)
        )
        
        return combined_score

    ##Function purpose: Get recency score based on timestamp metadata
    def _get_recency_score(self, doc: str, mem_type: str, username: str) -> float:
        """Calculate recency score based on temporal decay (0-1, higher is more recent)."""
        ##Block purpose: Check cache first if cache manager is available
        if self.cache_manager:
            cache_key = CacheManager.make_key('recency', username, mem_type, doc[:100])
            cached_score = self.cache_manager.recency_cache.get(cache_key)
            if cached_score is not None:
                return cached_score
        
        try:
            ##Block purpose: Query metadata for timestamp
            if mem_type == 'semantic':
                results = self.semantic_memory.collection.query(
                    query_texts=[doc], n_results=1, where={"username": username}
                )
            elif mem_type == 'episodic':
                results = self.episodic_memory.collection.query(
                    query_texts=[doc], n_results=1, where={"username": username}
                )
            elif mem_type == 'procedural':
                results = self.procedural_memory.collection.query(
                    query_texts=[doc], n_results=1, where={"username": username}
                )
            else:
                return 0.5  # Default for insights (no timestamp metadata)
            
            if not results.get('metadatas') or not results['metadatas'][0]:
                return 0.5  # Default if no metadata
            
            metadata = results['metadatas'][0][0] if results['metadatas'][0] else {}
            timestamp_str = metadata.get('timestamp')
            
            if not timestamp_str:
                return 0.5  # Default if no timestamp
            
            ##Block purpose: Calculate temporal decay (exponential decay over 30 days)
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now() - timestamp).total_seconds() / 86400  # Convert to days
                decay_half_life = 30.0  # Half-life of 30 days
                recency_score = 0.5 ** (age_days / decay_half_life)
                recency_score = min(max(recency_score, 0.0), 1.0)  # Clamp to [0, 1]
                
                ##Block purpose: Cache result if cache manager is available
                if hasattr(self, 'cache_manager') and self.cache_manager:
                    try:
                        from jenova.utils.cache import CacheManager
                        cache_key = CacheManager.make_key('recency', username, mem_type, doc[:100])
                        self.cache_manager.recency_cache.set(cache_key, recency_score)
                    except Exception:
                        pass  # Continue without caching if cache fails
                
                return recency_score
            except (ValueError, TypeError):
                return 0.5  # Default if timestamp parsing fails
                
        except Exception as e:
            self.file_logger.log_error(f"Error calculating recency score: {e}")
            return 0.5  # Default on error

    ##Function purpose: Get centrality score from Cortex graph
    def _get_centrality_score(self, doc: str, username: str) -> float:
        """Retrieves centrality score from Cortex graph (0-1, normalized)."""
        ##Block purpose: Use integration layer if available for better semantic matching
        if self.integration_layer:
            try:
                return self.integration_layer.get_centrality_score(doc, username)
            except Exception as e:
                self.file_logger.log_error(f"Error getting centrality score from integration layer: {e}")
        
        ##Block purpose: Fallback to direct Cortex access if no integration layer
        if not self.cortex:
            return 0.0  # No Cortex integration
        
        try:
            ##Block purpose: Find related nodes in Cortex by searching graph
            related_nodes = self._find_related_cortex_nodes(doc, username)
            
            if not related_nodes:
                return 0.0
            
            ##Block purpose: Calculate average centrality from related nodes
            total_centrality = sum(node.get('metadata', {}).get('centrality', 0) for node in related_nodes)
            avg_centrality = total_centrality / len(related_nodes) if related_nodes else 0.0
            
            ##Block purpose: Normalize to 0-1 range (assuming max centrality ~10, adjust as needed)
            normalized = min(avg_centrality / 10.0, 1.0)
            return normalized
            
        except Exception as e:
            self.file_logger.log_error(f"Error calculating centrality score: {e}")
            return 0.0  # Default on error

    ##Function purpose: Find nodes in Cortex related to memory item
    def _find_related_cortex_nodes(self, doc: str, username: str) -> List[Dict[str, Any]]:
        """Finds Cortex nodes semantically related to memory item."""
        ##Block purpose: Use integration layer if available for semantic similarity search
        if self.integration_layer:
            try:
                related_node_items = self.integration_layer.find_related_cortex_nodes(doc, username, max_nodes=5)
                return [item['node'] for item in related_node_items]
            except Exception as e:
                self.file_logger.log_error(f"Error finding related nodes via integration layer: {e}")
        
        ##Block purpose: Fallback to simple keyword matching if no integration layer
        if not self.cortex:
            return []
        
        try:
            ##Block purpose: Search Cortex graph for nodes with similar content
            all_nodes = self.cortex.get_all_nodes_by_type('insight', username)
            all_nodes.extend(self.cortex.get_all_nodes_by_type('assumption', username))
            
            ##Block purpose: Simple content matching (could be enhanced with semantic similarity)
            related_nodes = []
            doc_lower = doc.lower()
            for node in all_nodes:
                node_content = node.get('content', '').lower()
                ##Block purpose: Check for keyword overlap (simple heuristic)
                if any(word in node_content for word in doc_lower.split() if len(word) > 3):
                    related_nodes.append(node)
            
            return related_nodes[:5]  # Limit to top 5 related nodes
            
        except Exception as e:
            self.file_logger.log_error(f"Error finding related Cortex nodes: {e}")
            return []

    ##Function purpose: Get confidence score for semantic memories
    def _get_confidence_score(self, doc: str, mem_type: str, username: str) -> float:
        """Retrieves confidence score from semantic memory metadata (0-1)."""
        if mem_type != 'semantic':
            return 0.5  # Default for non-semantic memories
        
        try:
            ##Block purpose: Query semantic memory metadata for confidence
            results = self.semantic_memory.collection.query(
                query_texts=[doc], n_results=1, where={"username": username}
            )
            
            if not results.get('metadatas') or not results['metadatas'][0]:
                return 0.5  # Default if no metadata
            
            metadata = results['metadatas'][0][0] if results['metadatas'][0] else {}
            confidence = metadata.get('confidence')
            
            if confidence is None:
                return 0.5  # Default if no confidence score
            
            ##Block purpose: Ensure confidence is in [0, 1] range
            return min(max(float(confidence), 0.0), 1.0)
            
        except Exception as e:
            self.file_logger.log_error(f"Error getting confidence score: {e}")
            return 0.5  # Default on error
