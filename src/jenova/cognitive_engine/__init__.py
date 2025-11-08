# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 5: Enhanced Cognitive Engine Package

Improvements:
- Timeout protection on all long operations
- Better error handling and recovery
- RAG system with LRU caching
- Optional/configurable memory re-ranking
- Improved scheduler with logging
"""

from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.cognitive_engine.scheduler import CognitiveScheduler
from jenova.cognitive_engine.rag_system import RAGSystem, LRUCache
from jenova.cognitive_engine.memory_search import MemorySearch

__version__ = "4.2.0"
__phase__ = "Phase 5: Cognitive Engine Enhancements"

__all__ = [
    "CognitiveEngine",
    "CognitiveScheduler",
    "RAGSystem",
    "LRUCache",
    "MemorySearch",
]
