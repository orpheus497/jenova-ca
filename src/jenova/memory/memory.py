##Script function and purpose: Unified Memory class wrapping ChromaDB for vector storage
"""
Unified Memory Class

Single Memory class that handles all memory types (episodic, semantic, procedural).
Replaces the legacy copy-paste pattern of three nearly-identical classes.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import structlog
from chromadb.config import Settings

from jenova.exceptions import MemorySearchError, MemoryStoreError
from jenova.memory.types import MemoryResult, MemoryType

##Sec: Import username validation for security (PATCH-001)
from jenova.utils.cache import TTLCache
from jenova.utils.validation import validate_username

if TYPE_CHECKING:
    from chromadb.api.types import EmbeddingFunction

##Step purpose: Initialize module logger
logger = structlog.get_logger(__name__)


##Class purpose: Unified memory interface wrapping ChromaDB collection
class Memory:
    """
    Unified memory interface for vector storage.

    Wraps a ChromaDB collection for a specific memory type.
    Use MemoryType enum to specify which type of memory.
    """

    ##Method purpose: Initialize memory with type and storage path
    def __init__(
        self,
        memory_type: MemoryType,
        storage_path: Path,
        embedding_function: EmbeddingFunction[list[str]] | None = None,
    ) -> None:
        """
        Initialize memory.

        Args:
            memory_type: Type of memory (episodic, semantic, procedural)
            storage_path: Path for persistent storage
            embedding_function: Optional custom embedding function for vector generation.
                                If None, ChromaDB uses its default embedding.
        """
        ##Step purpose: Store configuration
        self.memory_type = memory_type
        self.storage_path = storage_path
        self._embedding_function = embedding_function

        ##Action purpose: Initialize ChromaDB client with persistent storage
        self._client = chromadb.PersistentClient(
            path=str(storage_path),
            settings=Settings(anonymized_telemetry=False),
        )

        ##Action purpose: Get or create collection for this memory type
        ##Condition purpose: Pass embedding function only if provided
        if embedding_function is not None:
            self._collection = self._client.get_or_create_collection(
                name=memory_type.value,
                metadata={"memory_type": memory_type.value},
                embedding_function=embedding_function,
            )
        else:
            self._collection = self._client.get_or_create_collection(
                name=memory_type.value,
                metadata={"memory_type": memory_type.value},
            )

        ##Update: Add search result cache for performance (P1-005)
        self._search_cache: TTLCache[str, list[MemoryResult]] = TTLCache(
            max_size=100,
            default_ttl=300,  # 5 minutes
        )

    ##Method purpose: Add content to memory with metadata
    def add(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Add content to memory.

        Args:
            content: Text content to store
            metadata: Optional metadata dict

        Returns:
            ID of stored memory

        Raises:
            MemoryStoreError: If storage fails
        """
        ##Step purpose: Generate unique ID
        doc_id = str(uuid.uuid4())

        ##Error purpose: Wrap ChromaDB errors
        try:
            ##Action purpose: Add document to collection
            self._collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata] if metadata else None,
            )
        except Exception as e:
            raise MemoryStoreError(content[:100], str(e)) from e

        ##Update: Invalidate search cache when new content is added (P1-005)
        self._search_cache.clear()

        return doc_id

    ##Method purpose: Search memory for relevant content
    def search(
        self,
        query: str,
        n_results: int = 5,
        username: str | None = None,
    ) -> list[MemoryResult]:
        """
        Search memory for relevant content.

        Args:
            query: Search query text
            n_results: Maximum results to return
            username: Optional username to filter results (for multi-user support)

        Returns:
            List of MemoryResult, sorted by relevance

        Raises:
            MemorySearchError: If search fails
        """
        ##Update: Check cache first (P1-005)
        cache_key_parts = [query, str(n_results), username or ""]
        cache_key = hashlib.sha256("|".join(cache_key_parts).encode()).hexdigest()

        cached_result = self._search_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(
                "memory_search_cache_hit",
                query_preview=query[:50],
                memory_type=self.memory_type.value,
            )
            return cached_result

        ##Error purpose: Wrap ChromaDB errors
        try:
            ##Step purpose: Build where clause for user filtering
            where_clause: dict[str, str] | None = None
            if username is not None:
                ##Sec: Validate username before database operations (PATCH-001)
                safe_username = validate_username(username)
                where_clause = {"username": safe_username}

            ##Action purpose: Query collection with optional user filter
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
            )
        except Exception as e:
            raise MemorySearchError(query, str(e)) from e

        ##Condition purpose: Handle empty results
        if not results["ids"] or not results["ids"][0]:
            return []

        ##Step purpose: Convert to MemoryResult objects
        memory_results: list[MemoryResult] = []

        ##Loop purpose: Build result list from raw ChromaDB output
        for i in range(len(results["ids"][0])):
            ##Step purpose: Calculate score from distance (lower distance = higher score)
            ##Fix: Add nested None checks for ChromaDB result structure
            distances = results["distances"]
            distance = distances[0][i] if distances and distances[0] else 0.0
            score = 1.0 / (1.0 + distance)

            documents = results["documents"]
            metadatas = results["metadatas"]

            memory_results.append(
                MemoryResult(
                    id=results["ids"][0][i],
                    content=documents[0][i] if documents and documents[0] else "",
                    score=score,
                    memory_type=self.memory_type,
                    metadata=metadatas[0][i] if metadatas and metadatas[0] else {},
                )
            )

        ##Update: Cache the result (P1-005)
        self._search_cache.set(cache_key, memory_results)

        return memory_results

    ##Method purpose: Get memory by ID
    def get(self, memory_id: str) -> MemoryResult | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: ID of memory to retrieve

        Returns:
            MemoryResult if found, None otherwise
        """
        ##Action purpose: Query by ID
        results = self._collection.get(ids=[memory_id])

        ##Condition purpose: Check if result exists
        if not results["ids"]:
            return None

        return MemoryResult(
            id=results["ids"][0],
            content=results["documents"][0] if results["documents"] else "",
            score=1.0,
            memory_type=self.memory_type,
            metadata=results["metadatas"][0] if results["metadatas"] else {},
        )

    ##Method purpose: Delete memory by ID
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        ##Condition purpose: Check if exists before deleting
        if self.get(memory_id) is None:
            return False

        ##Action purpose: Delete from collection
        self._collection.delete(ids=[memory_id])
        return True

    ##Method purpose: Get total count of memories
    def count(self) -> int:
        """Get total number of memories stored."""
        return self._collection.count()

    ##Method purpose: Clear all memories of this type
    def clear(self) -> None:
        """Delete all memories in this collection."""
        ##Action purpose: Delete and recreate collection
        self._client.delete_collection(self.memory_type.value)
        self._collection = self._client.get_or_create_collection(
            name=self.memory_type.value,
            metadata={"memory_type": self.memory_type.value},
        )
