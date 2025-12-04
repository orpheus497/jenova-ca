# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root
# directory of this source tree.

"""
Core interfaces for the JENOVA Cognitive Architecture.

This module provides abstract base classes and protocols that define the
pluggable interface for the cognitive architecture. These interfaces allow
the architecture to be integrated with any LLM backend, memory system, or
application framework.

Usage:
    Implement these interfaces to integrate JENOVA with your AI system:

    1. LLMAdapter - Wrap any LLM (OpenAI, Anthropic, local models)
    2. MemoryBackend - Use any vector store (ChromaDB, Pinecone, Weaviate)
    3. EmbeddingProvider - Use any embedding model (sentence-transformers, OpenAI)
    4. Logger - Integrate with your logging framework

Example:
    >>> from jenova.core.interfaces import LLMAdapter, MemoryBackend
    >>> from jenova.core import CognitiveArchitecture
    >>>
    >>> class MyLLM(LLMAdapter):
    ...     def generate(self, prompt, **kwargs):
    ...         return my_llm_call(prompt)
    ...
    >>> architecture = CognitiveArchitecture(llm=MyLLM())
"""

from jenova.core.interfaces.base import (  # LLM Interfaces; Memory Interfaces; Logging Interface; Configuration Interface; Cognitive Component Interfaces
    CognitiveComponent, ConfigProvider, EmbeddingProvider, InsightGenerator,
    KnowledgeGraph, LLMAdapter, Logger, MemoryBackend, MemoryEntry, MemoryType,
    ReasoningEngine, SearchResult)

__all__ = [
    # LLM Interfaces
    "LLMAdapter",
    "EmbeddingProvider",
    # Memory Interfaces
    "MemoryBackend",
    "MemoryEntry",
    "MemoryType",
    "SearchResult",
    # Logging Interface
    "Logger",
    # Configuration Interface
    "ConfigProvider",
    # Cognitive Component Interfaces
    "CognitiveComponent",
    "ReasoningEngine",
    "InsightGenerator",
    "KnowledgeGraph",
]
