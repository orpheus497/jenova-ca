# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Base interface definitions for the JENOVA Cognitive Architecture.

This module defines the abstract base classes and protocols that form the
foundation of the pluggable cognitive architecture. All interfaces are designed
to be implementation-agnostic, allowing integration with any LLM, vector store,
or application framework.

The interfaces follow these principles:
    1. Minimal dependencies - No external libraries required
    2. Type-safe - Full type hints for IDE support
    3. Extensible - Easy to add new capabilities
    4. Testable - Designed for easy mocking and testing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
NodeT = TypeVar("NodeT")
LinkT = TypeVar("LinkT")


# =============================================================================
# Data Classes for Memory
# =============================================================================


class MemoryType(Enum):
    """Types of memory in the cognitive architecture."""
    EPISODIC = "episodic"      # Conversations and events
    SEMANTIC = "semantic"       # Facts and knowledge
    PROCEDURAL = "procedural"   # How-to and procedures
    INSIGHT = "insight"         # Learned insights
    ASSUMPTION = "assumption"   # Formed assumptions


@dataclass
class MemoryEntry:
    """
    A single entry in any memory system.

    This is the universal format for storing and retrieving memories.
    Memory backends should convert their internal format to/from this.

    Attributes:
        id: Unique identifier for the entry
        content: The actual content/text of the memory
        memory_type: Type of memory (episodic, semantic, procedural, insight)
        user: User this memory belongs to
        metadata: Additional structured data
        timestamp: When this memory was created
        embedding: Optional pre-computed embedding vector
    """
    id: str
    content: str
    memory_type: MemoryType
    user: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """
    A single search result from memory.

    Attributes:
        entry: The memory entry that matched
        score: Relevance/similarity score (0-1, higher is better)
        distance: Raw distance from query (lower is closer)
    """
    entry: MemoryEntry
    score: float
    distance: float = 0.0


# =============================================================================
# LLM Interfaces
# =============================================================================


@runtime_checkable
class LLMAdapter(Protocol):
    """
    Protocol for LLM adapters.

    Implement this interface to integrate any LLM (local or API-based)
    with the cognitive architecture.

    Example:
        >>> class OpenAIAdapter(LLMAdapter):
        ...     def generate(self, prompt, **kwargs):
        ...         response = openai.chat.completions.create(
        ...             model="gpt-4",
        ...             messages=[{"role": "user", "content": prompt}],
        ...             **kwargs
        ...         )
        ...         return response.choices[0].message.content
        ...
        ...     def generate_with_context(self, prompt, context, **kwargs):
        ...         full_prompt = f"Context:\\n{context}\\n\\nQuestion: {prompt}"
        ...         return self.generate(full_prompt, **kwargs)
    """

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional LLM-specific parameters

        Returns:
            Generated text response
        """
        ...

    def generate_with_context(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text with retrieved context (RAG).

        Args:
            prompt: The user's query
            context: List of relevant context strings from memory
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated response grounded in context
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    Implement this to use any embedding model with the memory systems.

    Example:
        >>> class SentenceTransformerProvider(EmbeddingProvider):
        ...     def __init__(self, model_name="all-MiniLM-L6-v2"):
        ...         from sentence_transformers import SentenceTransformer
        ...         self.model = SentenceTransformer(model_name)
        ...         self._dimension = self.model.get_sentence_embedding_dimension()
        ...
        ...     def embed(self, texts):
        ...         return self.model.encode(texts).tolist()
        ...
        ...     @property
        ...     def dimension(self):
        ...         return self._dimension
    """

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        ...

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query for similarity search.

        Some models use different embeddings for queries vs documents.
        Override this if your model requires it.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        ...


# =============================================================================
# Memory Interfaces
# =============================================================================


class MemoryBackend(ABC):
    """
    Abstract base class for memory storage backends.

    Implement this to use any vector database (ChromaDB, Pinecone, Weaviate, etc.)
    or custom storage solution.

    Example:
        >>> class PineconeBackend(MemoryBackend):
        ...     def __init__(self, index_name, embedding_provider):
        ...         import pinecone
        ...         self.index = pinecone.Index(index_name)
        ...         self.embedding_provider = embedding_provider
        ...
        ...     def store(self, entry):
        ...         embedding = entry.embedding or self.embedding_provider.embed(entry.content)[0]
        ...         self.index.upsert(vectors=[(entry.id, embedding, entry.metadata)])
        ...         return entry.id
        ...
        ...     def search(self, query, n_results=5, **kwargs):
        ...         query_embedding = self.embedding_provider.embed_query(query)
        ...         results = self.index.query(vector=query_embedding, top_k=n_results)
        ...         return [self._to_search_result(r) for r in results.matches]
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry.

        Args:
            entry: The memory entry to store

        Returns:
            The ID of the stored entry
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[MemoryType] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query text
            n_results: Maximum number of results
            memory_type: Filter by memory type
            user: Filter by user
            **kwargs: Backend-specific parameters

        Returns:
            List of search results ordered by relevance
        """
        pass

    @abstractmethod
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific entry by ID.

        Args:
            entry_id: The entry ID to retrieve

        Returns:
            The memory entry if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """
        Delete an entry by ID.

        Args:
            entry_id: The entry ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def count(self, memory_type: Optional[MemoryType] = None, user: Optional[str] = None) -> int:
        """
        Count entries in memory.

        Args:
            memory_type: Filter by memory type
            user: Filter by user

        Returns:
            Number of matching entries
        """
        pass

    def bulk_store(self, entries: List[MemoryEntry]) -> List[str]:
        """
        Store multiple entries efficiently.

        Default implementation calls store() for each entry.
        Override for batch optimization.

        Args:
            entries: List of entries to store

        Returns:
            List of stored entry IDs
        """
        return [self.store(entry) for entry in entries]

    def clear(self, memory_type: Optional[MemoryType] = None, user: Optional[str] = None) -> int:
        """
        Clear entries from memory.

        Args:
            memory_type: Only clear this type (None = all types)
            user: Only clear this user's entries (None = all users)

        Returns:
            Number of entries deleted
        """
        raise NotImplementedError("Bulk clear not implemented for this backend")


# =============================================================================
# Logging Interface
# =============================================================================


@runtime_checkable
class Logger(Protocol):
    """
    Protocol for logging in the cognitive architecture.

    Implement this to integrate with your application's logging system.

    Example:
        >>> class PythonLogger(Logger):
        ...     def __init__(self, name="jenova"):
        ...         import logging
        ...         self.logger = logging.getLogger(name)
        ...
        ...     def info(self, message, **kwargs):
        ...         self.logger.info(message, extra=kwargs)
        ...
        ...     def warning(self, message, **kwargs):
        ...         self.logger.warning(message, extra=kwargs)
        ...
        ...     def error(self, message, error=None, **kwargs):
        ...         self.logger.error(message, exc_info=error, extra=kwargs)
        ...
        ...     def debug(self, message, **kwargs):
        ...         self.logger.debug(message, extra=kwargs)
    """

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        ...

    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log an error message."""
        ...

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        ...


# =============================================================================
# Configuration Interface
# =============================================================================


@runtime_checkable
class ConfigProvider(Protocol):
    """
    Protocol for configuration providers.

    Implement this to load configuration from any source (YAML, JSON, env vars, etc.)
    """

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (dot notation supported, e.g., "memory.cache_size")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        ...

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name

        Returns:
            Dictionary of section configuration
        """
        ...


# =============================================================================
# Cognitive Component Interfaces
# =============================================================================


class CognitiveComponent(ABC):
    """
    Base class for all cognitive components.

    All major components (memory systems, reasoning engines, etc.) should
    inherit from this to ensure consistent lifecycle management.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component."""
        pass

    def start(self) -> None:
        """Start the component (for components with background tasks)."""
        pass

    def stop(self) -> None:
        """Stop the component gracefully."""
        pass

    def dispose(self) -> None:
        """Release all resources."""
        pass

    def health_check(self) -> Dict[str, Any]:
        """
        Check component health.

        Returns:
            Dictionary with health information:
            - "healthy": bool
            - "message": str (optional)
            - Additional component-specific data
        """
        return {"healthy": True}


class ReasoningEngine(ABC):
    """
    Abstract base class for reasoning engines.

    The reasoning engine implements the core cognitive cycle:
    Retrieve → Plan → Execute → Reflect
    """

    @abstractmethod
    def think(
        self,
        query: str,
        user: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Process a query through the cognitive cycle.

        Args:
            query: User's input/question
            user: User identifier
            context: Optional pre-retrieved context
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        pass

    @abstractmethod
    def plan(self, query: str, context: List[str]) -> str:
        """
        Generate an execution plan for the query.

        Args:
            query: The query to plan for
            context: Retrieved context

        Returns:
            Execution plan as text
        """
        pass

    @abstractmethod
    def execute(self, query: str, plan: str, context: List[str]) -> str:
        """
        Execute the plan to generate a response.

        Args:
            query: Original query
            plan: Generated plan
            context: Retrieved context

        Returns:
            Generated response
        """
        pass


class InsightGenerator(ABC):
    """
    Abstract base class for insight generation.

    Insights are learned knowledge extracted from conversations and
    experiences that should be remembered for future interactions.
    """

    @abstractmethod
    def generate_insight(
        self,
        conversation: List[str],
        user: str
    ) -> Optional[Dict[str, str]]:
        """
        Generate an insight from conversation.

        Args:
            conversation: Recent conversation history
            user: User identifier

        Returns:
            Dictionary with "topic" and "insight" keys, or None if no insight
        """
        pass

    @abstractmethod
    def store_insight(
        self,
        insight: str,
        topic: str,
        user: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an insight in the knowledge base.

        Args:
            insight: The insight content
            topic: Topic/category for the insight
            user: User identifier
            metadata: Additional metadata

        Returns:
            Insight ID
        """
        pass

    @abstractmethod
    def get_relevant_insights(
        self,
        query: str,
        user: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant insights for a query.

        Args:
            query: Search query
            user: User identifier
            max_results: Maximum results to return

        Returns:
            List of relevant insights with metadata
        """
        pass


class KnowledgeGraph(ABC, Generic[NodeT, LinkT]):
    """
    Abstract base class for knowledge graph implementations.

    The knowledge graph stores interconnected cognitive nodes representing
    insights, assumptions, and learned knowledge.
    """

    @abstractmethod
    def add_node(
        self,
        node_type: str,
        content: str,
        user: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a node to the graph.

        Args:
            node_type: Type of node (insight, assumption, document, etc.)
            content: Node content
            user: User identifier
            metadata: Additional metadata

        Returns:
            Node ID
        """
        pass

    @abstractmethod
    def add_link(
        self,
        source_id: str,
        target_id: str,
        relationship: str
    ) -> None:
        """
        Add a link between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Relationship type (related_to, elaborates_on, etc.)
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[NodeT]:
        """
        Get a node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node if found, None otherwise
        """
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        relationship: Optional[str] = None
    ) -> List[Tuple[NodeT, str]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node ID
            relationship: Filter by relationship type

        Returns:
            List of (node, relationship) tuples
        """
        pass

    @abstractmethod
    def reflect(self, user: str) -> List[str]:
        """
        Perform deep reflection on the graph.

        This analyzes the graph structure, links orphan nodes,
        generates meta-insights, and prunes outdated knowledge.

        Args:
            user: User identifier

        Returns:
            List of messages describing reflection results
        """
        pass


# =============================================================================
# Factory Protocol
# =============================================================================


@runtime_checkable
class ComponentFactory(Protocol):
    """
    Protocol for creating cognitive components.

    Use this to customize component creation without modifying core code.
    """

    def create_memory_backend(
        self,
        memory_type: MemoryType,
        **kwargs
    ) -> MemoryBackend:
        """Create a memory backend for the specified type."""
        ...

    def create_llm_adapter(self, **kwargs) -> LLMAdapter:
        """Create an LLM adapter."""
        ...

    def create_embedding_provider(self, **kwargs) -> EmbeddingProvider:
        """Create an embedding provider."""
        ...

    def create_reasoning_engine(self, **kwargs) -> ReasoningEngine:
        """Create a reasoning engine."""
        ...
