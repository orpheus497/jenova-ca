# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Memory subsystems for JENOVA Cognitive Architecture.

This package provides:
- Episodic memory (experiences and episodes)
- Semantic memory (facts and knowledge)
- Procedural memory (procedures and how-to knowledge)
- Base memory class with atomic operations (Phase 4)
- Memory manager for orchestration (Phase 4)
"""

# Original memory implementations
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .procedural import ProceduralMemory

# Phase 4: Enhanced memory infrastructure
from .base_memory import BaseMemory, MemoryError, MemoryInitError, MemoryOperationError
from .memory_manager import MemoryManager

__all__ = [
    # Original implementations
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    # Phase 4: Base infrastructure
    "BaseMemory",
    "MemoryError",
    "MemoryInitError",
    "MemoryOperationError",
    "MemoryManager",
]

__version__ = "4.2.0"
__phase__ = "Phase 4: Memory Layer (Foundation)"
