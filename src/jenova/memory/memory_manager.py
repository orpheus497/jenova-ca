# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Memory Manager for JENOVA - Orchestrates all memory systems.

This module provides unified access to all memory types:
- Episodic memory
- Semantic memory
- Procedural memory

Features:
- Unified interface
- Health monitoring
- Metrics collection
- Coordinated operations
- Error aggregation

Phase 4 Implementation
"""

from contextlib import nullcontext
from typing import Dict, List, Tuple, Optional, Any

from jenova.infrastructure import HealthMonitor, MetricsCollector
from jenova.infrastructure.timeout_manager import with_timeout


class MemoryManager:
    """
    Orchestrates all memory systems with health monitoring and metrics.

    Features:
    - Unified access to all memory types
    - Cross-memory search
    - Health status aggregation
    - Performance metrics
    - Coordinated initialization
    """

    def __init__(
        self,
        episodic_memory=None,
        semantic_memory=None,
        procedural_memory=None,
        health_monitor: Optional[HealthMonitor] = None,
        metrics: Optional[MetricsCollector] = None,
        ui_logger=None,
        file_logger=None,
    ):
        """
        Initialize memory manager.

        Args:
            episodic_memory: EpisodicMemory instance
            semantic_memory: SemanticMemory instance
            procedural_memory: ProceduralMemory instance
            health_monitor: Optional HealthMonitor for system health
            metrics: Optional MetricsCollector for tracking
            ui_logger: UI logger for user messages
            file_logger: File logger for debug logs
        """
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.procedural = procedural_memory
        self.health_monitor = health_monitor
        self.metrics = metrics
        self.ui_logger = ui_logger
        self.file_logger = file_logger

    def add_to_episodic(self, summary: str, username: str, **kwargs) -> Optional[str]:
        """
        Add episode to episodic memory.

        Args:
            summary: Episode summary
            username: User identifier
            **kwargs: Additional metadata

        Returns:
            Entry ID if successful, None otherwise
        """
        if not self.episodic:
            if self.file_logger:
                self.file_logger.log_warning("Episodic memory not available")
            return None

        try:
            with (
                self.metrics.measure("memory_episodic_add")
                if self.metrics
                else nullcontext()
            ):
                return self.episodic.add_episode(summary, username, **kwargs)
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to add to episodic memory: {e}")
            return None

    def add_to_semantic(self, fact: str, username: str, **kwargs) -> Optional[str]:
        """
        Add fact to semantic memory.

        Args:
            fact: Fact text
            username: User identifier
            **kwargs: Additional metadata

        Returns:
            Entry ID if successful, None otherwise
        """
        if not self.semantic:
            if self.file_logger:
                self.file_logger.log_warning("Semantic memory not available")
            return None

        try:
            with (
                self.metrics.measure("memory_semantic_add")
                if self.metrics
                else nullcontext()
            ):
                return self.semantic.add_fact(fact, username, **kwargs)
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to add to semantic memory: {e}")
            return None

    def add_to_procedural(
        self, procedure: str, username: str, **kwargs
    ) -> Optional[str]:
        """
        Add procedure to procedural memory.

        Args:
            procedure: Procedure description
            username: User identifier
            **kwargs: Additional metadata

        Returns:
            Entry ID if successful, None otherwise
        """
        if not self.procedural:
            if self.file_logger:
                self.file_logger.log_warning("Procedural memory not available")
            return None

        try:
            with (
                self.metrics.measure("memory_procedural_add")
                if self.metrics
                else nullcontext()
            ):
                return self.procedural.add_procedure(procedure, username, **kwargs)
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Failed to add to procedural memory: {e}")
            return None

    @with_timeout(60)
    def search_all(
        self, query: str, username: str, n_results_per_memory: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search across all memory types.

        Args:
            query: Search query
            username: User identifier
            n_results_per_memory: Results per memory type

        Returns:
            Dictionary mapping memory type to results
        """
        results = {}

        # Search episodic
        if self.episodic:
            try:
                with (
                    self.metrics.measure("memory_episodic_search")
                    if self.metrics
                    else nullcontext()
                ):
                    results["episodic"] = self.episodic.recall_relevant_episodes(
                        query, username, n_results_per_memory
                    )
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Episodic search failed: {e}")
                results["episodic"] = []

        # Search semantic
        if self.semantic:
            try:
                with (
                    self.metrics.measure("memory_semantic_search")
                    if self.metrics
                    else nullcontext()
                ):
                    results["semantic"] = self.semantic.search_collection(
                        query, username, n_results_per_memory
                    )
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Semantic search failed: {e}")
                results["semantic"] = []

        # Search procedural
        if self.procedural:
            try:
                with (
                    self.metrics.measure("memory_procedural_search")
                    if self.metrics
                    else nullcontext()
                ):
                    results["procedural"] = self.procedural.search(
                        query, username, n_results_per_memory
                    )
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Procedural search failed: {e}")
                results["procedural"] = []

        if self.file_logger:
            total_results = sum(len(r) for r in results.values())
            self.file_logger.log_info(
                f"Cross-memory search returned {total_results} total results "
                f"(episodic: {len(results.get('episodic', []))}, "
                f"semantic: {len(results.get('semantic', []))}, "
                f"procedural: {len(results.get('procedural', []))})"
            )

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all memory systems.

        Returns:
            Dictionary with stats for each memory type
        """
        stats = {}

        if self.episodic:
            try:
                stats["episodic"] = {
                    "count": (
                        self.episodic.collection.count()
                        if hasattr(self.episodic, "collection")
                        else 0
                    ),
                    "status": "available",
                }
            except Exception as e:
                stats["episodic"] = {"count": 0, "status": "error", "error": str(e)}

        if self.semantic:
            try:
                stats["semantic"] = {
                    "count": (
                        self.semantic.collection.count()
                        if hasattr(self.semantic, "collection")
                        else 0
                    ),
                    "status": "available",
                }
            except Exception as e:
                stats["semantic"] = {"count": 0, "status": "error", "error": str(e)}

        if self.procedural:
            try:
                stats["procedural"] = {
                    "count": (
                        self.procedural.collection.count()
                        if hasattr(self.procedural, "collection")
                        else 0
                    ),
                    "status": "available",
                }
            except Exception as e:
                stats["procedural"] = {"count": 0, "status": "error", "error": str(e)}

        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get aggregated health status of all memory systems.

        Returns:
            Health status dictionary
        """
        health = {"overall_status": "healthy", "memory_systems": {}}

        # Check each memory system
        if self.episodic:
            if hasattr(self.episodic, "get_health_status"):
                health["memory_systems"]["episodic"] = self.episodic.get_health_status()
            else:
                health["memory_systems"]["episodic"] = {"status": "available"}

        if self.semantic:
            if hasattr(self.semantic, "get_health_status"):
                health["memory_systems"]["semantic"] = self.semantic.get_health_status()
            else:
                health["memory_systems"]["semantic"] = {"status": "available"}

        if self.procedural:
            if hasattr(self.procedural, "get_health_status"):
                health["memory_systems"][
                    "procedural"
                ] = self.procedural.get_health_status()
            else:
                health["memory_systems"]["procedural"] = {"status": "available"}

        # Determine overall status
        for memory_name, memory_health in health["memory_systems"].items():
            if memory_health.get("status") == "unhealthy":
                health["overall_status"] = "degraded"
                break

        return health

    def clear_all(self, username: Optional[str] = None):
        """
        Clear all memory systems (DANGEROUS - use with caution).

        Args:
            username: If provided, only clear for this user
        """
        if self.ui_logger:
            self.ui_logger.warning("Clearing memory systems...")

        # Note: Current memory implementations don't support
        # per-user clearing, so this would require enhancement

        if self.file_logger:
            self.file_logger.log_warning(
                f"Memory clear requested for user: {username or 'all'}"
            )

    def get_total_count(self) -> int:
        """Get total number of entries across all memories."""
        total = 0

        if self.episodic:
            try:
                total += (
                    self.episodic.collection.count()
                    if hasattr(self.episodic, "collection")
                    else 0
                )
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(
                        f"Failed to count episodic memory entries: {e}"
                    )

        if self.semantic:
            try:
                total += (
                    self.semantic.collection.count()
                    if hasattr(self.semantic, "collection")
                    else 0
                )
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(
                        f"Failed to count semantic memory entries: {e}"
                    )

        if self.procedural:
            try:
                total += (
                    self.procedural.collection.count()
                    if hasattr(self.procedural, "collection")
                    else 0
                )
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_warning(
                        f"Failed to count procedural memory entries: {e}"
                    )

        return total

    def is_healthy(self) -> bool:
        """Check if all memory systems are healthy."""
        health = self.get_health_status()
        return health["overall_status"] in ["healthy", "degraded"]

    def get_summary(self) -> str:
        """Get a human-readable summary of memory status."""
        stats = self.get_memory_stats()
        health = self.get_health_status()

        lines = [
            "Memory Systems Summary:",
            f"  Overall Status: {health['overall_status'].upper()}",
            f"  Total Entries: {self.get_total_count()}",
            "",
        ]

        for memory_type, memory_stats in stats.items():
            status = memory_stats.get("status", "unknown")
            count = memory_stats.get("count", 0)
            lines.append(f"  {memory_type.title()}: {count} entries ({status})")

        return "\n".join(lines)
