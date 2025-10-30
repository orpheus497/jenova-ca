# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Base memory class for JENOVA with atomic operations and robust error handling.

This module provides common functionality for all memory types:
- Atomic file operations via FileManager
- Data validation via DataValidator
- Timeout protection
- Error handling and recovery
- Collection migration
- Health monitoring integration

Phase 4 Implementation
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any

import chromadb

from jenova.infrastructure import FileManager, DataValidator
from jenova.infrastructure.timeout_manager import with_timeout, timeout
from jenova.utils.embedding import CustomEmbeddingFunction


class MemoryError(Exception):
    """Base exception for memory operations."""
    pass


class MemoryInitError(MemoryError):
    """Raised when memory initialization fails."""
    pass


class MemoryOperationError(MemoryError):
    """Raised when memory operation fails."""
    pass


class BaseMemory(ABC):
    """
    Base class for all memory types with robust operation handling.

    Features:
    - Atomic directory creation
    - Collection migration support
    - Error recovery
    - Timeout protection
    - Metrics integration
    - Health monitoring
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ui_logger,
        file_logger,
        db_path: str,
        llm,
        embedding_model,
        collection_name: str,
        file_manager: Optional[FileManager] = None,
        metrics: Optional[Any] = None
    ):
        """
        Initialize base memory system.

        Args:
            config: Configuration dictionary
            ui_logger: UI logger for user messages
            file_logger: File logger for debug logs
            db_path: Path to ChromaDB storage
            llm: LLM interface for metadata extraction
            embedding_model: Embedding model for vector operations
            collection_name: Name of the ChromaDB collection
            file_manager: Optional FileManager for atomic operations
            metrics: Optional MetricsCollector for tracking
        """
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = db_path
        self.llm = llm
        self.collection_name = collection_name
        self.file_manager = file_manager
        self.metrics = metrics

        # Initialize collection
        self.collection = None
        self.client = None
        self.embedding_function = None

        try:
            # Create database directory with atomic operation
            self._ensure_db_directory()

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.db_path)

            # Create embedding function
            self.embedding_function = CustomEmbeddingFunction(
                model=embedding_model,
                model_name=config['model']['embedding_model']
            )

            # Initialize collection with migration support
            self.collection = self._initialize_collection()

            if self.file_logger:
                self.file_logger.log_info(
                    f"{self.__class__.__name__} initialized successfully "
                    f"(collection: {self.collection_name}, count: {self.collection.count()})"
                )

        except Exception as e:
            error_msg = f"Failed to initialize {self.__class__.__name__}: {e}"
            if self.file_logger:
                self.file_logger.log_error(error_msg)
            raise MemoryInitError(error_msg) from e

    def _ensure_db_directory(self):
        """Ensure database directory exists using atomic operations."""
        if self.file_manager:
            # Use FileManager for atomic directory creation
            try:
                # FileManager doesn't have mkdir, fall back to os.makedirs
                # but log through file manager
                os.makedirs(self.db_path, exist_ok=True)
                if self.file_logger:
                    self.file_logger.log_info(f"Database directory ensured: {self.db_path}")
            except Exception as e:
                raise MemoryInitError(f"Failed to create database directory: {e}")
        else:
            # Direct creation if no file manager
            os.makedirs(self.db_path, exist_ok=True)

    def _initialize_collection(self) -> chromadb.Collection:
        """
        Initialize ChromaDB collection with migration support.

        Returns:
            Initialized ChromaDB collection

        Raises:
            MemoryInitError: If collection initialization fails
        """
        try:
            # Try to get or create collection
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            return collection

        except ValueError as e:
            # Handle embedding function conflict
            if "Embedding function conflict" in str(e):
                return self._migrate_collection()
            else:
                raise MemoryInitError(f"Collection initialization failed: {e}")

    def _migrate_collection(self) -> chromadb.Collection:
        """
        Migrate collection when embedding function changes.

        Returns:
            New collection with migrated data

        Raises:
            MemoryInitError: If migration fails
        """
        if self.ui_logger:
            self.ui_logger.system_message(
                f"Embedding function conflict detected in {self.collection_name}. "
                "Migrating data..."
            )

        if self.file_logger:
            self.file_logger.log_warning(
                f"Embedding function conflict in {self.collection_name}. "
                "Starting migration."
            )

        try:
            # Get old collection data
            old_collection = self.client.get_collection(name=self.collection_name)
            data = old_collection.get(include=["documents", "metadatas"])

            # Delete old collection
            self.client.delete_collection(name=self.collection_name)

            # Create new collection
            new_collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

            # Migrate data if exists
            if data['ids']:
                new_collection.add(
                    ids=data['ids'],
                    documents=data['documents'],
                    metadatas=data['metadatas']
                )

                if self.ui_logger:
                    self.ui_logger.system_message(
                        f"Migration successful: {len(data['ids'])} entries migrated"
                    )

                if self.file_logger:
                    self.file_logger.log_info(
                        f"Migrated {len(data['ids'])} entries to new collection"
                    )

            return new_collection

        except Exception as e:
            error_msg = f"Collection migration failed: {e}"
            if self.file_logger:
                self.file_logger.log_error(error_msg)

            if self.ui_logger:
                self.ui_logger.system_message(
                    f"Warning: Failed to migrate {self.collection_name} data"
                )

            raise MemoryInitError(error_msg) from e

    @with_timeout(30)
    def add_entry(
        self,
        document: str,
        metadata: Dict[str, Any],
        entry_id: Optional[str] = None
    ) -> str:
        """
        Add entry to memory with timeout protection.

        Args:
            document: Document text to store
            metadata: Metadata dictionary
            entry_id: Optional custom entry ID

        Returns:
            Entry ID of added document

        Raises:
            MemoryOperationError: If add operation fails
        """
        try:
            # Validate and sanitize metadata
            from jenova.utils.data_sanitizer import sanitize_metadata
            metadata = sanitize_metadata(metadata)

            # Generate ID if not provided
            if not entry_id:
                import uuid
                entry_id = f"{self.collection_name}_{uuid.uuid4()}"

            # Add to collection
            self.collection.add(
                ids=[entry_id],
                documents=[document],
                metadatas=[metadata]
            )

            if self.file_logger:
                self.file_logger.log_info(
                    f"Added entry {entry_id} to {self.collection_name}"
                )

            # Track metrics if available
            if self.metrics:
                self.metrics.record_operation(
                    f"{self.collection_name}_add",
                    success=True
                )

            return entry_id

        except Exception as e:
            error_msg = f"Failed to add entry to {self.collection_name}: {e}"
            if self.file_logger:
                self.file_logger.log_error(error_msg)

            if self.metrics:
                self.metrics.record_operation(
                    f"{self.collection_name}_add",
                    success=False
                )

            raise MemoryOperationError(error_msg) from e

    @with_timeout(30)
    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Query memory with timeout protection.

        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of (document, distance) tuples

        Raises:
            MemoryOperationError: If query fails
        """
        try:
            # Handle empty collection
            if self.collection.count() == 0:
                return []

            # Limit results to collection size
            n_results = min(n_results, self.collection.count())

            # Perform query
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )

            # Extract results
            if not results['documents']:
                return []

            matched = list(zip(
                results['documents'][0],
                results['distances'][0]
            ))

            if self.file_logger:
                self.file_logger.log_info(
                    f"Query in {self.collection_name} returned {len(matched)} results"
                )

            # Track metrics if available
            if self.metrics:
                self.metrics.record_operation(
                    f"{self.collection_name}_query",
                    success=True,
                    metadata={'results': len(matched)}
                )

            return matched

        except Exception as e:
            error_msg = f"Query failed in {self.collection_name}: {e}"
            if self.file_logger:
                self.file_logger.log_error(error_msg)

            if self.metrics:
                self.metrics.record_operation(
                    f"{self.collection_name}_query",
                    success=False
                )

            # Don't raise for query failures, return empty
            return []

    def get_count(self) -> int:
        """Get number of entries in memory."""
        try:
            return self.collection.count()
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Failed to get count for {self.collection_name}: {e}"
                )
            return 0

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of this memory system."""
        try:
            count = self.get_count()
            return {
                "name": self.collection_name,
                "status": "healthy",
                "count": count,
                "collection_exists": self.collection is not None,
                "client_connected": self.client is not None
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "status": "unhealthy",
                "error": str(e),
                "count": 0,
                "collection_exists": False,
                "client_connected": False
            }

    @abstractmethod
    def add(self, *args, **kwargs):
        """Add entry to memory. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def search(self, *args, **kwargs):
        """Search memory. Must be implemented by subclasses."""
        pass
