# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Memory Deduplication Engine for JENOVA Cognitive Architecture.

This module provides content-based deduplication to eliminate redundant memory entries,
reducing storage footprint and improving retrieval efficiency.

Phase 20 Feature #3: Advanced Memory Deduplication
- Content-based deduplication using xxhash (fast) or SHA256 (fallback)
- Reference counting for shared content blocks
- Automatic garbage collection of orphaned entries
- Integration with compression tiers
- Thread-safe concurrent access

Benefits:
- Eliminate duplicate memories across time periods
- Reduce storage by 30-50% for redundant data
- Faster backup and restore operations
- Improved cache efficiency
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import json

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ContentBlock:
    """
    Represents a deduplicated content block.

    Thread-safety: All access should be protected by DeduplicationEngine._lock.
    """
    content_hash: str
    data: bytes
    reference_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize size from data if not set."""
        if self.size_bytes == 0:
            self.size_bytes = len(self.data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (data excluded for size)."""
        return {
            "content_hash": self.content_hash,
            "reference_count": self.reference_count,
            "first_seen": self.first_seen.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "size_bytes": self.size_bytes,
            "metadata": self.metadata.copy(),
        }


@dataclass
class DedupReference:
    """
    Reference to a deduplicated content block.

    Used to track which entries reference which content blocks.
    """
    entry_id: str
    content_hash: str
    created_at: datetime = field(default_factory=datetime.now)


class DeduplicationEngine:
    """
    Manages content-based deduplication for memory entries.

    Thread-safety: All public methods are thread-safe using internal RLock.

    Features:
    - Content-based deduplication using fast hashing (xxhash or SHA256)
    - Reference counting for garbage collection
    - Automatic cleanup of orphaned blocks
    - Integration with compression tiers
    - Statistics and monitoring
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize deduplication engine.

        Args:
            storage_dir: Optional directory to persist dedup index
        """
        self._lock = threading.RLock()
        self._content_blocks: Dict[str, ContentBlock] = {}
        self._references: Dict[str, Set[str]] = defaultdict(set)  # content_hash -> set of entry_ids
        self._entry_to_hash: Dict[str, str] = {}  # entry_id -> content_hash
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.xxhash_available = XXHASH_AVAILABLE

        if not self.xxhash_available:
            logger.warning("xxhash not available, using SHA256 (slower). Install with: pip install xxhash")

        # Statistics
        self._stats = {
            "total_blocks": 0,
            "total_references": 0,
            "bytes_stored": 0,
            "bytes_saved": 0,
            "dedup_ratio": 0.0,
        }

    def hash_content(self, data: bytes) -> str:
        """
        Calculate hash of content for deduplication.

        Uses xxhash if available (fast), otherwise SHA256 (slower).

        Args:
            data: Data to hash

        Returns:
            Hex digest of hash
        """
        if self.xxhash_available:
            return xxhash.xxh64(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()

    def store_content(
        self,
        entry_id: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, bool]:
        """
        Store content with deduplication.

        Thread-safe: Can be called from multiple threads.

        Args:
            entry_id: Unique identifier for this entry
            data: Content data
            metadata: Optional metadata for the content

        Returns:
            Tuple of (content_hash, is_duplicate)
            - content_hash: Hash of the content
            - is_duplicate: True if content was already stored
        """
        content_hash = self.hash_content(data)

        with self._lock:
            # Check if we already have this content
            is_duplicate = content_hash in self._content_blocks

            if not is_duplicate:
                # New content block
                block = ContentBlock(
                    content_hash=content_hash,
                    data=data,
                    reference_count=1,
                    size_bytes=len(data),
                    metadata=metadata or {}
                )
                self._content_blocks[content_hash] = block
                self._stats["total_blocks"] += 1
                self._stats["bytes_stored"] += len(data)
            else:
                # Existing content, increment reference
                block = self._content_blocks[content_hash]
                block.reference_count += 1
                block.last_accessed = datetime.now()
                self._stats["bytes_saved"] += len(data)

            # Track reference
            self._references[content_hash].add(entry_id)
            self._entry_to_hash[entry_id] = content_hash
            self._stats["total_references"] += 1

            # Update dedup ratio
            self._update_dedup_ratio()

            if is_duplicate:
                logger.debug(
                    f"Deduplicated entry {entry_id}: hash={content_hash[:8]}... "
                    f"(refs={block.reference_count}, saved={len(data)} bytes)"
                )

            return content_hash, is_duplicate

    def retrieve_content(self, entry_id: str) -> Optional[bytes]:
        """
        Retrieve content by entry ID.

        Thread-safe: Returns snapshot of data.

        Args:
            entry_id: Entry identifier

        Returns:
            Content data or None if not found
        """
        with self._lock:
            content_hash = self._entry_to_hash.get(entry_id)
            if not content_hash:
                return None

            block = self._content_blocks.get(content_hash)
            if not block:
                logger.error(f"Orphaned reference: entry {entry_id} -> hash {content_hash}")
                return None

            # Update access time
            block.last_accessed = datetime.now()

            return block.data

    def retrieve_by_hash(self, content_hash: str) -> Optional[bytes]:
        """
        Retrieve content directly by hash.

        Thread-safe: Returns snapshot of data.

        Args:
            content_hash: Content hash

        Returns:
            Content data or None if not found
        """
        with self._lock:
            block = self._content_blocks.get(content_hash)
            if not block:
                return None

            block.last_accessed = datetime.now()
            return block.data

    def remove_reference(self, entry_id: str) -> bool:
        """
        Remove a reference to content.

        Decrements reference count and performs garbage collection if needed.

        Thread-safe: Can be called from multiple threads.

        Args:
            entry_id: Entry identifier

        Returns:
            True if reference was removed, False if not found
        """
        with self._lock:
            content_hash = self._entry_to_hash.get(entry_id)
            if not content_hash:
                return False

            # Remove reference
            self._references[content_hash].discard(entry_id)
            del self._entry_to_hash[entry_id]
            self._stats["total_references"] -= 1

            # Decrement reference count
            block = self._content_blocks.get(content_hash)
            if block:
                block.reference_count -= 1

                # Garbage collect if no more references
                if block.reference_count <= 0:
                    self._remove_block(content_hash)
                    logger.debug(f"Garbage collected block {content_hash[:8]}...")

            self._update_dedup_ratio()
            return True

    def _remove_block(self, content_hash: str) -> None:
        """
        Remove a content block (internal, called under lock).

        Args:
            content_hash: Hash of block to remove
        """
        block = self._content_blocks.get(content_hash)
        if block:
            self._stats["bytes_stored"] -= block.size_bytes
            self._stats["total_blocks"] -= 1
            del self._content_blocks[content_hash]

            # Clean up reference tracking
            if content_hash in self._references:
                del self._references[content_hash]

    def garbage_collect(self, aggressive: bool = False) -> int:
        """
        Perform garbage collection to remove orphaned blocks.

        Thread-safe: Can be called from multiple threads.

        Args:
            aggressive: If True, also remove blocks with zero references
                       even if they're in the registry (defensive cleanup)

        Returns:
            Number of blocks removed
        """
        with self._lock:
            blocks_to_remove = []

            for content_hash, block in self._content_blocks.items():
                # Check for orphaned blocks
                if block.reference_count <= 0 or (
                    aggressive and len(self._references.get(content_hash, set())) == 0
                ):
                    blocks_to_remove.append(content_hash)

            # Remove blocks
            for content_hash in blocks_to_remove:
                self._remove_block(content_hash)

            count = len(blocks_to_remove)
            if count > 0:
                logger.info(f"Garbage collected {count} orphaned content blocks")
                self._update_dedup_ratio()

            return count

    def _update_dedup_ratio(self) -> None:
        """Update deduplication ratio statistics (internal, called under lock)."""
        total_size = self._stats["bytes_stored"] + self._stats["bytes_saved"]
        if total_size > 0:
            self._stats["dedup_ratio"] = self._stats["bytes_saved"] / total_size
        else:
            self._stats["dedup_ratio"] = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.

        Thread-safe: Returns snapshot of stats.

        Returns:
            Dictionary with deduplication statistics
        """
        with self._lock:
            return {
                "total_blocks": self._stats["total_blocks"],
                "total_references": self._stats["total_references"],
                "bytes_stored": self._stats["bytes_stored"],
                "bytes_saved": self._stats["bytes_saved"],
                "total_bytes": self._stats["bytes_stored"] + self._stats["bytes_saved"],
                "dedup_ratio": round(self._stats["dedup_ratio"] * 100, 2),
                "avg_references_per_block": (
                    round(self._stats["total_references"] / self._stats["total_blocks"], 2)
                    if self._stats["total_blocks"] > 0 else 0.0
                ),
            }

    def get_block_info(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a content block.

        Thread-safe: Returns snapshot of block info.

        Args:
            content_hash: Hash of the block

        Returns:
            Block information dictionary or None
        """
        with self._lock:
            block = self._content_blocks.get(content_hash)
            if not block:
                return None

            return block.to_dict()

    def list_blocks(
        self,
        min_references: Optional[int] = None,
        max_age_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all content blocks with optional filters.

        Thread-safe: Returns snapshot of blocks.

        Args:
            min_references: Only include blocks with >= this many references
            max_age_days: Only include blocks accessed within this many days

        Returns:
            List of block information dictionaries
        """
        with self._lock:
            result = []
            now = datetime.now()

            for block in self._content_blocks.values():
                # Apply filters
                if min_references and block.reference_count < min_references:
                    continue

                if max_age_days:
                    age = (now - block.last_accessed).days
                    if age > max_age_days:
                        continue

                result.append(block.to_dict())

            return result

    def get_entry_references(self, content_hash: str) -> Set[str]:
        """
        Get all entry IDs that reference a content hash.

        Thread-safe: Returns snapshot of references.

        Args:
            content_hash: Hash to query

        Returns:
            Set of entry IDs
        """
        with self._lock:
            return self._references.get(content_hash, set()).copy()

    def export_index(self) -> Dict[str, Any]:
        """
        Export deduplication index for persistence.

        Thread-safe: Returns snapshot of index.

        Returns:
            Dictionary with index data (excludes content blocks)
        """
        with self._lock:
            return {
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "statistics": self.get_statistics(),
                "blocks": [
                    {
                        "content_hash": hash_val,
                        "reference_count": block.reference_count,
                        "size_bytes": block.size_bytes,
                        "first_seen": block.first_seen.isoformat(),
                        "last_accessed": block.last_accessed.isoformat(),
                    }
                    for hash_val, block in self._content_blocks.items()
                ],
                "entry_mapping": self._entry_to_hash.copy(),
            }

    def save_index(self, file_path: Optional[Path] = None) -> None:
        """
        Save deduplication index to disk.

        Thread-safe: Can be called from multiple threads.

        Args:
            file_path: Path to save index (default: storage_dir/dedup_index.json)
        """
        if file_path is None:
            if self.storage_dir is None:
                raise ValueError("No storage directory configured")
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.storage_dir / "dedup_index.json"

        index_data = self.export_index()

        with open(file_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Saved deduplication index to {file_path}")

    def clear(self) -> None:
        """
        Clear all deduplication data.

        Thread-safe: Can be called from multiple threads.

        WARNING: This will remove all content blocks and references.
        """
        with self._lock:
            self._content_blocks.clear()
            self._references.clear()
            self._entry_to_hash.clear()
            self._stats = {
                "total_blocks": 0,
                "total_references": 0,
                "bytes_stored": 0,
                "bytes_saved": 0,
                "dedup_ratio": 0.0,
            }
            logger.info("Cleared all deduplication data")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.storage_dir:
            try:
                self.save_index()
            except Exception as e:
                logger.error(f"Failed to save dedup index on exit: {e}")
        return False
