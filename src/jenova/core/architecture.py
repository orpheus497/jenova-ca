# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root
# directory of this source tree.

"""
JENOVA Cognitive Architecture - Main Entry Point

This module provides the CognitiveArchitecture class, the primary interface
for integrating JENOVA's cognitive capabilities into any AI system.

The architecture is designed to be:
    - Pluggable: Works with any LLM, vector store, or embedding model
    - Modular: Use only the components you need
    - Stateful: Maintains persistent memory across sessions
    - Extensible: Easy to add new cognitive capabilities

Quick Start:
    >>> from jenova.core import CognitiveArchitecture
    >>>
    >>> # Create with default components (uses local LLM and ChromaDB)
    >>> arch = CognitiveArchitecture.create_default(user_data_path="~/.my_app/data")
    >>>
    >>> # Process a query
    >>> response = arch.think("What did we discuss yesterday?", user="john")
    >>>
    >>> # Store a fact
    >>> arch.remember("John prefers Python over JavaScript", user="john", memory_type="semantic")

Custom Integration:
    >>> from jenova.core import CognitiveArchitecture
    >>> from jenova.core.interfaces import LLMAdapter, MemoryBackend
    >>>
    >>> # Create with custom components
    >>> arch = CognitiveArchitecture(
    ...     llm=MyOpenAIAdapter(),
    ...     memory_backend=MyPineconeBackend(),
    ...     embedding_provider=MyEmbeddingProvider(),
    ... )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jenova.core.interfaces import (CognitiveComponent, ConfigProvider,
                                    EmbeddingProvider, InsightGenerator,
                                    KnowledgeGraph, LLMAdapter, Logger,
                                    MemoryBackend, MemoryEntry, MemoryType,
                                    ReasoningEngine, SearchResult)


@dataclass
class CognitiveConfig:
    """
    Configuration for the Cognitive Architecture.

    This dataclass holds all configuration options for the architecture.
    Use this to customize behavior without modifying code.

    Attributes:
        memory_cache_size: Number of items to cache in memory (default: 100)
        insight_interval: Generate insights every N turns (default: 5)
        assumption_interval: Generate assumptions every N turns (default: 7)
        reflection_interval: Reflect on graph every N turns (default: 20)
        max_context_items: Maximum context items for RAG (default: 10)
        rerank_enabled: Enable LLM-based re-ranking (default: True)
        rerank_timeout: Timeout for re-ranking in seconds (default: 15)
        llm_timeout: Default LLM timeout in seconds (default: 120)
        planning_timeout: Timeout for planning phase (default: 60)
    """
    memory_cache_size: int = 100
    insight_interval: int = 5
    assumption_interval: int = 7
    reflection_interval: int = 20
    max_context_items: int = 10
    rerank_enabled: bool = True
    rerank_timeout: int = 15
    llm_timeout: int = 120
    planning_timeout: int = 60
    enable_distributed: bool = False
    user_data_path: Optional[str] = None


class CognitiveArchitecture(CognitiveComponent):
    """
    The JENOVA Cognitive Architecture - A pluggable cognitive framework for AI systems.

    This class provides the main interface for integrating cognitive capabilities
    into any AI application. It manages:

    - Multi-layered memory (episodic, semantic, procedural, insights)
    - Cognitive graph for knowledge relationships
    - RAG-based response generation
    - Insight and assumption learning
    - Reflective reasoning

    The architecture follows the cognitive cycle:
    1. RETRIEVE: Search memories for relevant context
    2. PLAN: Generate execution plan based on context
    3. EXECUTE: Generate response using RAG
    4. REFLECT: Extract insights and update knowledge

    Example:
        >>> # Quick setup with defaults
        >>> arch = CognitiveArchitecture.create_default()
        >>>
        >>> # Process user query
        >>> response = arch.think("Hello, how are you?", user="alice")
        >>>
        >>> # Access individual components
        >>> arch.memory.search("python programming", user="alice")
        >>> arch.cortex.reflect(user="alice")

    Attributes:
        config: Architecture configuration
        llm: LLM adapter for text generation
        memory: Memory backend for storage and retrieval
        embedding: Embedding provider for vector operations
        cortex: Knowledge graph for cognitive relationships
        reasoning: Reasoning engine for cognitive cycle
        insight_gen: Insight generator for learning
        logger: Logger for debugging and monitoring
    """

    def __init__(
        self,
        llm: Optional[LLMAdapter] = None,
        memory_backend: Optional[MemoryBackend] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        cortex: Optional[KnowledgeGraph] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        insight_generator: Optional[InsightGenerator] = None,
        logger: Optional[Logger] = None,
        config: Optional[CognitiveConfig] = None,
    ):
        """
        Initialize the Cognitive Architecture.

        Args:
            llm: LLM adapter for text generation (required for most operations)
            memory_backend: Storage backend for memories
            embedding_provider: Provider for text embeddings
            cortex: Knowledge graph implementation
            reasoning_engine: Custom reasoning engine
            insight_generator: Custom insight generator
            logger: Logging implementation
            config: Architecture configuration
        """
        self.config = config or CognitiveConfig()
        self._llm = llm
        self._memory = memory_backend
        self._embedding = embedding_provider
        self._cortex = cortex
        self._reasoning = reasoning_engine
        self._insight_gen = insight_generator
        self._logger = logger

        # Conversation state
        self._history: Dict[str, List[str]] = {}
        self._turn_count: Dict[str, int] = {}

        # Track initialization
        self._initialized = False

    @classmethod
    def create_default(
        cls,
        user_data_path: Optional[str] = None,
        model_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        config: Optional[CognitiveConfig] = None,
    ) -> "CognitiveArchitecture":
        """
        Create a CognitiveArchitecture with default components.

        This factory method creates a fully functional architecture using
        the built-in JENOVA components:
        - llama-cpp-python for local LLM inference
        - ChromaDB for vector storage
        - sentence-transformers for embeddings

        Args:
            user_data_path: Path for user data storage (default: ~/.jenova-ai)
            model_path: Path to GGUF model file (auto-detected if not provided)
            embedding_model: Sentence-transformer model name
            config: Optional configuration override

        Returns:
            Configured CognitiveArchitecture instance

        Raises:
            FileNotFoundError: If model file not found
            ImportError: If required dependencies not installed
        """
        if config is None:
            config = CognitiveConfig()

        if user_data_path:
            config.user_data_path = user_data_path

        # Import default implementations
        # These are lazy imports to avoid circular dependencies
        from jenova.core.adapters import (create_default_cortex,
                                          create_default_embedding,
                                          create_default_llm,
                                          create_default_logger,
                                          create_default_memory)

        # Create components
        logger = create_default_logger()
        embedding = create_default_embedding(embedding_model)
        llm = create_default_llm(model_path, logger)
        memory = create_default_memory(user_data_path, embedding)
        cortex = create_default_cortex(user_data_path, llm, logger)

        return cls(
            llm=llm,
            memory_backend=memory,
            embedding_provider=embedding,
            cortex=cortex,
            logger=logger,
            config=config,
        )

    # =========================================================================
    # Properties for Component Access
    # =========================================================================

    @property
    def llm(self) -> Optional[LLMAdapter]:
        """Get the LLM adapter."""
        return self._llm

    @property
    def memory(self) -> Optional[MemoryBackend]:
        """Get the memory backend."""
        return self._memory

    @property
    def embedding(self) -> Optional[EmbeddingProvider]:
        """Get the embedding provider."""
        return self._embedding

    @property
    def cortex(self) -> Optional[KnowledgeGraph]:
        """Get the knowledge graph (cortex)."""
        return self._cortex

    @property
    def reasoning(self) -> Optional[ReasoningEngine]:
        """Get the reasoning engine."""
        return self._reasoning

    @property
    def insight_gen(self) -> Optional[InsightGenerator]:
        """Get the insight generator."""
        return self._insight_gen

    @property
    def logger(self) -> Optional[Logger]:
        """Get the logger."""
        return self._logger

    # =========================================================================
    # Core Cognitive Methods
    # =========================================================================

    def think(
        self,
        query: str,
        user: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Process a query through the full cognitive cycle.

        This is the main entry point for cognitive processing. It:
        1. Retrieves relevant context from all memory layers
        2. Generates an execution plan
        3. Executes the plan to generate a response
        4. Optionally triggers insight generation

        Args:
            query: User's input/question
            user: User identifier for personalization
            context: Optional pre-retrieved context (skips retrieval)
            **kwargs: Additional parameters passed to reasoning engine

        Returns:
            Generated response

        Example:
            >>> response = arch.think(
            ...     "What were the key points from our last meeting?",
            ...     user="alice"
            ... )
        """
        if self._reasoning:
            return self._reasoning.think(query, user, context, **kwargs)

        # Fallback: Simple RAG implementation
        if not self._llm:
            raise RuntimeError("No LLM adapter configured")

        # Track conversation
        if user not in self._history:
            self._history[user] = []
            self._turn_count[user] = 0

        self._turn_count[user] += 1

        # Retrieve context if not provided
        if context is None:
            context = self.retrieve(query, user)

        # Generate response
        if context:
            response = self._llm.generate_with_context(query, context)
        else:
            response = self._llm.generate(query)

        # Update history
        self._history[user].append(f"{user}: {query}")
        self._history[user].append(f"Assistant: {response}")

        # Limit history size
        max_history = 10
        if len(self._history[user]) > max_history * 2:
            self._history[user] = self._history[user][-(max_history * 2):]

        # Store interaction in episodic memory
        if self._memory:
            entry = MemoryEntry(
                id=f"episode_{user}_{self._turn_count[user]}",
                content=f"{user}: {query}\nAssistant: {response}",
                memory_type=MemoryType.EPISODIC,
                user=user,
            )
            self._memory.store(entry)

        # Check for insight generation
        if (
            self._insight_gen
            and self._turn_count[user] % self.config.insight_interval == 0
        ):
            self._generate_insight(user)

        return response

    def retrieve(
        self,
        query: str,
        user: str,
        n_results: int = None,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[str]:
        """
        Retrieve relevant context from memory.

        Searches across all memory layers and returns the most relevant
        context for the given query.

        Args:
            query: Search query
            user: User identifier
            n_results: Maximum results (default: config.max_context_items)
            memory_types: Specific memory types to search (default: all)

        Returns:
            List of relevant context strings

        Example:
            >>> context = arch.retrieve("Python best practices", user="alice")
            >>> for item in context:
            ...     print(item)
        """
        if not self._memory:
            return []

        n_results = n_results or self.config.max_context_items

        if memory_types is None:
            memory_types = [
                MemoryType.SEMANTIC,
                MemoryType.EPISODIC,
                MemoryType.PROCEDURAL]

        all_results: List[SearchResult] = []

        for mem_type in memory_types:
            results = self._memory.search(
                query,
                n_results=n_results,
                memory_type=mem_type,
                user=user,
            )
            all_results.extend(results)

        # Sort by score (higher is better)
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Return top N content strings
        return [r.entry.content for r in all_results[:n_results]]

    def remember(
        self,
        content: str,
        user: str,
        memory_type: Union[str, MemoryType] = MemoryType.SEMANTIC,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store information in memory.

        Args:
            content: Content to remember
            user: User identifier
            memory_type: Type of memory (semantic, procedural, etc.)
            metadata: Additional metadata

        Returns:
            Memory entry ID

        Example:
            >>> # Store a fact
            >>> arch.remember(
            ...     "The project deadline is December 15th",
            ...     user="alice",
            ...     memory_type="semantic"
            ... )
            >>>
            >>> # Store a procedure
            >>> arch.remember(
            ...     "To deploy: 1. Run tests 2. Build 3. Deploy to staging",
            ...     user="alice",
            ...     memory_type="procedural"
            ... )
        """
        if not self._memory:
            raise RuntimeError("No memory backend configured")

        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        import uuid
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user=user,
            metadata=metadata or {},
        )

        return self._memory.store(entry)

    def reflect(self, user: str) -> List[str]:
        """
        Trigger deep reflection on the knowledge graph.

        This analyzes the cognitive graph structure, links orphan nodes,
        generates meta-insights, and prunes outdated knowledge.

        Args:
            user: User identifier

        Returns:
            List of messages describing reflection results

        Example:
            >>> messages = arch.reflect(user="alice")
            >>> for msg in messages:
            ...     print(msg)
        """
        if not self._cortex:
            return ["Knowledge graph not configured"]

        return self._cortex.reflect(user)

    def learn_insight(
        self,
        insight: str,
        topic: str,
        user: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Manually store an insight.

        Args:
            insight: The insight content
            topic: Topic/category for the insight
            user: User identifier
            metadata: Additional metadata

        Returns:
            Insight ID
        """
        if self._insight_gen:
            return self._insight_gen.store_insight(
                insight, topic, user, metadata)

        # Fallback: Store as semantic memory
        if self._memory:
            entry = MemoryEntry(
                id=f"insight_{user}_{topic}",
                content=f"[{topic}] {insight}",
                memory_type=MemoryType.INSIGHT,
                user=user,
                metadata={"topic": topic, **(metadata or {})},
            )
            return self._memory.store(entry)

        raise RuntimeError("No insight storage configured")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _generate_insight(self, user: str) -> None:
        """Generate insight from recent conversation."""
        if not self._insight_gen or user not in self._history:
            return

        result = self._insight_gen.generate_insight(
            self._history[user][-10:],  # Last 10 messages
            user,
        )

        if result:
            self._insight_gen.store_insight(
                result["insight"],
                result["topic"],
                user,
            )

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def initialize(self) -> None:
        """Initialize all components."""
        self._initialized = True
        if self._logger:
            self._logger.info("CognitiveArchitecture initialized")

    def start(self) -> None:
        """Start the architecture (for background tasks)."""
        if self._logger:
            self._logger.info("CognitiveArchitecture started")

    def stop(self) -> None:
        """Stop the architecture gracefully."""
        if self._logger:
            self._logger.info("CognitiveArchitecture stopped")

    def dispose(self) -> None:
        """Release all resources."""
        self._history.clear()
        self._turn_count.clear()
        if self._logger:
            self._logger.info("CognitiveArchitecture disposed")

    def health_check(self) -> Dict[str, Any]:
        """Check health of all components."""
        health = {
            "healthy": True,
            "components": {},
        }

        if self._llm:
            health["components"]["llm"] = {"available": True}

        if self._memory:
            try:
                count = self._memory.count()
                health["components"]["memory"] = {
                    "available": True, "entries": count}
            except Exception as e:
                health["components"]["memory"] = {
                    "available": False, "error": str(e)}
                health["healthy"] = False

        if self._cortex:
            health["components"]["cortex"] = {"available": True}

        return health

    # =========================================================================
    # Conversation Management
    # =========================================================================

    def get_history(self, user: str) -> List[str]:
        """Get conversation history for a user."""
        return self._history.get(user, [])

    def clear_history(self, user: str) -> None:
        """Clear conversation history for a user."""
        if user in self._history:
            self._history[user] = []
            self._turn_count[user] = 0

    def get_turn_count(self, user: str) -> int:
        """Get the turn count for a user."""
        return self._turn_count.get(user, 0)
