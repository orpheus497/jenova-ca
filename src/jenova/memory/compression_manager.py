# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Memory Compression Manager for JENOVA Cognitive Architecture.

This module provides memory compression capabilities to reduce storage footprint
and enable long-term knowledge retention with automatic tiering based on access patterns.

Phase 20 Feature #3: Advanced Memory Compression & Archival
- LZ4 compression for recent memories (fast access, ~2-3x compression)
- Zstandard compression for archives (high ratio, ~5-10x compression)
- Automatic tiering based on access patterns
- Deduplication using xxhash

Benefits:
- Store 10x more memories in same space
- Faster backup operations
- Efficient long-term archival
- Reduced I/O overhead
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import json

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    import hashlib

logger = logging.getLogger(__name__)


class CompressionTier(Enum):
    """Compression tiers for memory data."""
    HOT = "hot"  # Recently accessed, LZ4 (fast)
    WARM = "warm"  # Occasionally accessed, LZ4 (fast)
    COLD = "cold"  # Rarely accessed, Zstandard (high ratio)
    ARCHIVED = "archived"  # Very old, Zstandard (max compression)


class CompressionManager:
    """
    Manages compression and tiering of memory data.

    Automatically selects compression strategy based on access patterns:
    - HOT: Uncompressed or LZ4 (recently accessed within 7 days)
    - WARM: LZ4 (accessed within 30 days)
    - COLD: Zstandard level 3 (accessed within 90 days)
    - ARCHIVED: Zstandard level 19 (accessed 90+ days ago)
    """

    def __init__(
        self,
        hot_days: int = 7,
        warm_days: int = 30,
        cold_days: int = 90,
    ):
        """
        Initialize compression manager.

        Args:
            hot_days: Days to keep in HOT tier
            warm_days: Days to keep in WARM tier
            cold_days: Days before moving to ARCHIVED tier
        """
        self.hot_days = hot_days
        self.warm_days = warm_days
        self.cold_days = cold_days

        # Check availability
        self.lz4_available = LZ4_AVAILABLE
        self.zstd_available = ZSTD_AVAILABLE
        self.xxhash_available = XXHASH_AVAILABLE

        if not self.lz4_available:
            logger.warning("LZ4 not available. Install with: pip install lz4")

        if not self.zstd_available:
            logger.warning("Zstandard not available. Install with: pip install zstandard")

        if not self.xxhash_available:
            logger.warning("xxhash not available, using hashlib (slower)")

    def get_tier(self, last_access: datetime) -> CompressionTier:
        """
        Determine compression tier based on last access time.

        Args:
            last_access: Last access timestamp

        Returns:
            Appropriate compression tier
        """
        age = datetime.now() - last_access

        if age <= timedelta(days=self.hot_days):
            return CompressionTier.HOT
        elif age <= timedelta(days=self.warm_days):
            return CompressionTier.WARM
        elif age <= timedelta(days=self.cold_days):
            return CompressionTier.COLD
        else:
            return CompressionTier.ARCHIVED

    def compress_lz4(self, data: bytes) -> bytes:
        """
        Compress data using LZ4 (fast compression).

        Args:
            data: Data to compress

        Returns:
            Compressed data

        Raises:
            RuntimeError: If LZ4 not available
        """
        if not self.lz4_available:
            raise RuntimeError("LZ4 not available")

        return lz4.frame.compress(data)

    def decompress_lz4(self, compressed: bytes) -> bytes:
        """
        Decompress LZ4 data.

        Args:
            compressed: Compressed data

        Returns:
            Decompressed data

        Raises:
            RuntimeError: If LZ4 not available
        """
        if not self.lz4_available:
            raise RuntimeError("LZ4 not available")

        return lz4.frame.decompress(compressed)

    def compress_zstd(self, data: bytes, level: int = 3) -> bytes:
        """
        Compress data using Zstandard (high compression ratio).

        Args:
            data: Data to compress
            level: Compression level (1-22, default: 3)

        Returns:
            Compressed data

        Raises:
            RuntimeError: If Zstandard not available
        """
        if not self.zstd_available:
            raise RuntimeError("Zstandard not available")

        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)

    def decompress_zstd(self, compressed: bytes) -> bytes:
        """
        Decompress Zstandard data.

        Args:
            compressed: Compressed data

        Returns:
            Decompressed data

        Raises:
            RuntimeError: If Zstandard not available
        """
        if not self.zstd_available:
            raise RuntimeError("Zstandard not available")

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(compressed)

    def compress(
        self,
        data: bytes,
        tier: CompressionTier
    ) -> tuple[bytes, str]:
        """
        Compress data according to tier.

        Args:
            data: Data to compress
            tier: Compression tier

        Returns:
            Tuple of (compressed data, compression method)
        """
        if tier == CompressionTier.HOT:
            # HOT tier: no compression for fastest access
            return data, "none"

        elif tier == CompressionTier.WARM:
            # WARM tier: LZ4 for fast compression/decompression
            if self.lz4_available:
                return self.compress_lz4(data), "lz4"
            else:
                return data, "none"

        elif tier == CompressionTier.COLD:
            # COLD tier: Zstandard level 3 (balanced)
            if self.zstd_available:
                return self.compress_zstd(data, level=3), "zstd-3"
            elif self.lz4_available:
                return self.compress_lz4(data), "lz4"
            else:
                return data, "none"

        else:  # ARCHIVED
            # ARCHIVED tier: Zstandard level 19 (max compression)
            if self.zstd_available:
                return self.compress_zstd(data, level=19), "zstd-19"
            elif self.lz4_available:
                return self.compress_lz4(data), "lz4"
            else:
                return data, "none"

    def decompress(
        self,
        compressed: bytes,
        method: str
    ) -> bytes:
        """
        Decompress data according to method.

        Args:
            compressed: Compressed data
            method: Compression method used

        Returns:
            Decompressed data
        """
        if method == "none":
            return compressed
        elif method == "lz4":
            return self.decompress_lz4(compressed)
        elif method.startswith("zstd"):
            return self.decompress_zstd(compressed)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def hash_content(self, data: bytes) -> str:
        """
        Calculate hash of data for deduplication.

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

    def get_compression_stats(
        self,
        original_size: int,
        compressed_size: int
    ) -> Dict[str, Any]:
        """
        Calculate compression statistics.

        Args:
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes

        Returns:
            Dictionary with compression statistics
        """
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        savings_percent = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0.0

        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": round(ratio, 2),
            "savings_percent": round(savings_percent, 1),
            "savings_bytes": original_size - compressed_size,
        }


def compress_memory_entry(
    entry: Dict[str, Any],
    manager: CompressionManager,
    last_access: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Compress a memory entry based on access pattern.

    Args:
        entry: Memory entry to compress
        manager: CompressionManager instance
        last_access: Last access time (default: now)

    Returns:
        Compressed entry with metadata
    """
    if last_access is None:
        last_access = datetime.now()

    # Determine tier
    tier = manager.get_tier(last_access)

    # Serialize entry
    data = json.dumps(entry, ensure_ascii=False).encode('utf-8')
    original_size = len(data)

    # Compress
    compressed, method = manager.compress(data, tier)
    compressed_size = len(compressed)

    # Calculate hash for deduplication
    content_hash = manager.hash_content(data)

    # Get stats
    stats = manager.get_compression_stats(original_size, compressed_size)

    return {
        "compressed_data": compressed,
        "compression_method": method,
        "compression_tier": tier.value,
        "content_hash": content_hash,
        "last_access": last_access.isoformat(),
        "stats": stats,
    }


def decompress_memory_entry(
    compressed_entry: Dict[str, Any],
    manager: CompressionManager
) -> Dict[str, Any]:
    """
    Decompress a memory entry.

    Args:
        compressed_entry: Compressed entry
        manager: CompressionManager instance

    Returns:
        Original entry
    """
    compressed_data = compressed_entry["compressed_data"]
    method = compressed_entry["compression_method"]

    # Decompress
    data = manager.decompress(compressed_data, method)

    # Deserialize
    return json.loads(data.decode('utf-8'))
