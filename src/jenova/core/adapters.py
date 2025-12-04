# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root
# directory of this source tree.

"""
Default adapters for the JENOVA Cognitive Architecture.

This module provides factory functions that create default implementations
of the core interfaces using JENOVA's built-in components. These are used
by CognitiveArchitecture.create_default() but can also be used individually.

The adapters wrap existing JENOVA components to conform to the new interfaces,
enabling backward compatibility while supporting the pluggable architecture.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jenova.core.interfaces import (EmbeddingProvider, KnowledgeGraph,
                                    LLMAdapter, Logger, MemoryBackend,
                                    MemoryEntry, MemoryType, SearchResult)

# =============================================================================
# Logger Adapter
# =============================================================================


class DefaultLogger:
    """Default logger using Python's logging module."""

    def __init__(self, name: str = "jenova"):
        import logging
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def info(self, message: str, **kwargs) -> None:
        self._logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._logger.warning(message, extra=kwargs)

    def error(
            self,
            message: str,
            error: Optional[Exception] = None,
            **kwargs) -> None:
        if error:
            self._logger.error(message, exc_info=error, extra=kwargs)
        else:
            self._logger.error(message, extra=kwargs)

    def debug(self, message: str, **kwargs) -> None:
        self._logger.debug(message, extra=kwargs)


def create_default_logger() -> Logger:
    """Create a default logger instance."""
    return DefaultLogger()


# =============================================================================
# Embedding Provider Adapter
# =============================================================================


class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, query: str) -> List[float]:
        return self.embed(query)[0]


def create_default_embedding(
        model_name: str = "all-MiniLM-L6-v2") -> EmbeddingProvider:
    """Create a default embedding provider using sentence-transformers."""
    return SentenceTransformerProvider(model_name)


# =============================================================================
# LLM Adapter
# =============================================================================


class LlamaCppAdapter:
    """LLM adapter for llama-cpp-python models."""

    def __init__(
            self,
            model_path: str,
            logger: Optional[Logger] = None,
            **kwargs):
        from llama_cpp import Llama

        self._logger = logger
        self._model_path = model_path

        # Default parameters
        default_params = {
            "n_ctx": 4096,
            "n_threads": 4,
            "n_gpu_layers": -1,  # Use all GPU layers if available
            "verbose": False,
        }
        default_params.update(kwargs)

        if logger:
            logger.info(f"Loading model from {model_path}")

        self._llm = Llama(model_path=model_path, **default_params)

        if logger:
            logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        result = self._llm(
            prompt,
            max_tokens=max_tokens or 512,
            temperature=temperature,
            **kwargs
        )
        return result["choices"][0]["text"]

    def generate_with_context(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        # Build RAG prompt
        context_str = "\n".join(f"- {c}" for c in context)
        full_prompt = f"""Context:
{context_str}

Question: {prompt}

Answer based on the context provided:"""

        return self.generate(full_prompt, temperature, max_tokens, **kwargs)


def create_default_llm(
    model_path: Optional[str] = None,
    logger: Optional[Logger] = None,
    **kwargs
) -> LLMAdapter:
    """
    Create a default LLM adapter using llama-cpp-python.

    Args:
        model_path: Path to GGUF model file. If not provided, searches common locations.
        logger: Optional logger
        **kwargs: Additional parameters for Llama

    Returns:
        LLM adapter instance

    Raises:
        FileNotFoundError: If no model found
    """
    if model_path is None:
        # Search common model locations
        search_paths = [
            "/usr/local/share/models",
            str(Path.home() / ".jenova-ai" / "models"),
            "./models",
        ]

        for base_path in search_paths:
            if os.path.exists(base_path):
                for file in os.listdir(base_path):
                    if file.endswith(".gguf"):
                        model_path = os.path.join(base_path, file)
                        break
            if model_path:
                break

        if not model_path:
            raise FileNotFoundError(
                "No GGUF model found. Please provide model_path or place a .gguf file "
                f"in one of: {search_paths}"
            )

    return LlamaCppAdapter(model_path, logger, **kwargs)


# =============================================================================
# Memory Backend Adapter
# =============================================================================


class ChromaDBBackend:
    """Memory backend using ChromaDB."""

    def __init__(
        self,
        persist_directory: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "jenova_memory",
    ):
        import chromadb

        self._persist_dir = persist_directory
        self._embedding = embedding_provider

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(path=persist_directory)

        # Create embedding function wrapper
        self._ef = self._create_embedding_function()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
        )

    def _create_embedding_function(self):
        """Create a ChromaDB-compatible embedding function."""
        from chromadb.api.types import EmbeddingFunction

        provider = self._embedding

        class ProviderEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input_texts: List[str]) -> List[List[float]]:
                return provider.embed(input_texts)

        return ProviderEmbeddingFunction()

    def store(self, entry: MemoryEntry) -> str:
        metadata = {
            "memory_type": entry.memory_type.value,
            "user": entry.user,
            "timestamp": entry.timestamp,
            **entry.metadata,
        }

        # Sanitize metadata (ChromaDB requires string values for some fields)
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif v is None:
                sanitized[k] = ""
            else:
                sanitized[k] = str(v)

        self._collection.add(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[sanitized],
        )

        return entry.id

    def search(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[MemoryType] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        if self._collection.count() == 0:
            return []

        # Build where clause
        where = {}
        if memory_type:
            where["memory_type"] = memory_type.value
        if user:
            where["user"] = user

        # Limit results to collection size
        n_results = min(n_results, self._collection.count())

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where if where else None,
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {
                }
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert distance to score (lower distance = higher score)
                score = 1.0 / (1.0 + distance)

                entry = MemoryEntry(
                    id=results["ids"][0][i], content=doc, memory_type=MemoryType(
                        metadata.get(
                            "memory_type", "semantic")), user=metadata.get(
                        "user", ""), metadata=metadata, timestamp=metadata.get(
                        "timestamp", ""), )

                search_results.append(SearchResult(
                    entry=entry,
                    score=score,
                    distance=distance,
                ))

        return search_results

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        result = self._collection.get(ids=[entry_id])

        if not result["documents"]:
            return None

        metadata = result["metadatas"][0] if result["metadatas"] else {}

        return MemoryEntry(
            id=entry_id,
            content=result["documents"][0],
            memory_type=MemoryType(metadata.get("memory_type", "semantic")),
            user=metadata.get("user", ""),
            metadata=metadata,
            timestamp=metadata.get("timestamp", ""),
        )

    def delete(self, entry_id: str) -> bool:
        try:
            self._collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    def count(
        self,
        memory_type: Optional[MemoryType] = None,
        user: Optional[str] = None
    ) -> int:
        if memory_type is None and user is None:
            return self._collection.count()

        # Need to query with filter
        where = {}
        if memory_type:
            where["memory_type"] = memory_type.value
        if user:
            where["user"] = user

        # ChromaDB doesn't have a count with filter, so we query all
        results = self._collection.get(where=where if where else None)
        return len(results["ids"])


def create_default_memory(
    persist_directory: Optional[str] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> MemoryBackend:
    """
    Create a default memory backend using ChromaDB.

    Args:
        persist_directory: Where to store the database (default: ~/.jenova-ai/memory)
        embedding_provider: Embedding provider (created if not provided)

    Returns:
        Memory backend instance
    """
    if persist_directory is None:
        persist_directory = str(Path.home() / ".jenova-ai" / "memory")

    if embedding_provider is None:
        embedding_provider = create_default_embedding()

    return ChromaDBBackend(persist_directory, embedding_provider)


# =============================================================================
# Knowledge Graph (Cortex) Adapter
# =============================================================================


class CortexAdapter:
    """Adapter wrapping the existing Cortex implementation."""

    def __init__(
        self,
        cortex_root: str,
        llm: LLMAdapter,
        logger: Optional[Logger] = None,
    ):
        from jenova.cortex.cortex import Cortex

        # Create a minimal config for Cortex
        config = {
            "cortex": {
                "relationship_weights": {
                    "related_to": 1.0,
                    "elaborates_on": 1.5,
                    "conflicts_with": 0.5,
                    "supports": 1.2,
                    "synthesizes": 2.0,
                },
                "pruning": {
                    "enabled": True,
                    "prune_interval": 10,
                    "max_age_days": 90,
                    "min_centrality": 0.1,
                },
            }
        }

        # Create wrapper for file_logger
        class FileLoggerWrapper:
            def __init__(self, logger):
                self._logger = logger

            def log_info(self, msg):
                if self._logger:
                    self._logger.info(msg)

            def log_warning(self, msg):
                if self._logger:
                    self._logger.warning(msg)

            def log_error(self, msg):
                if self._logger:
                    self._logger.error(msg)

        # Create wrapper for llm
        class LLMWrapper:
            def __init__(self, adapter):
                self._adapter = adapter

            def generate(self, prompt, temperature=0.7):
                return self._adapter.generate(prompt, temperature=temperature)

        self._cortex = Cortex(
            config=config,
            ui_logger=None,
            file_logger=FileLoggerWrapper(logger),
            llm=LLMWrapper(llm),
            cortex_root=cortex_root,
        )

    def add_node(
        self,
        node_type: str,
        content: str,
        user: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._cortex.add_node(
            node_type=node_type,
            content=content,
            user=user,
            metadata=metadata,
        )

    def add_link(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
    ) -> None:
        self._cortex.add_link(source_id, target_id, relationship)

    def get_node(self, node_id: str):
        return self._cortex.get_node(node_id)

    def get_neighbors(self, node_id: str, relationship: Optional[str] = None):
        # Get all links involving this node
        neighbors = []
        node = self._cortex.get_node(node_id)
        if not node:
            return neighbors

        for link in self._cortex.graph.get("links", []):
            if link.source_id == node_id:
                if relationship is None or link.relationship == relationship:
                    target = self._cortex.get_node(link.target_id)
                    if target:
                        neighbors.append((target, link.relationship))
            elif link.target_id == node_id:
                if relationship is None or link.relationship == relationship:
                    source = self._cortex.get_node(link.source_id)
                    if source:
                        neighbors.append((source, link.relationship))

        return neighbors

    def reflect(self, user: str) -> List[str]:
        return self._cortex.reflect(user)


def create_default_cortex(
    cortex_root: Optional[str] = None,
    llm: Optional[LLMAdapter] = None,
    logger: Optional[Logger] = None,
) -> KnowledgeGraph:
    """
    Create a default knowledge graph using the Cortex implementation.

    Args:
        cortex_root: Directory for cortex storage (default: ~/.jenova-ai/cortex)
        llm: LLM adapter for analysis (required)
        logger: Optional logger

    Returns:
        Knowledge graph instance
    """
    if cortex_root is None:
        cortex_root = str(Path.home() / ".jenova-ai" / "cortex")

    if llm is None:
        raise ValueError("LLM adapter required for Cortex")

    return CortexAdapter(cortex_root, llm, logger)
