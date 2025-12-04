# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root
# directory of this source tree.

"""
JENOVA Cognitive Architecture

A comprehensive, pluggable cognitive framework for building stateful,
learning-capable AI systems. JENOVA can be used as a complete terminal-based
AI assistant or integrated as a modular cognitive layer in other applications.

=== As a Modular Cognitive Layer ===

    # Import the cognitive architecture
    from jenova import CognitiveArchitecture

    # Create with default components (local LLM + ChromaDB)
    arch = CognitiveArchitecture.create_default()

    # Process queries through the cognitive cycle
    response = arch.think("What did we discuss yesterday?", user="alice")

    # Store knowledge
    arch.remember("User prefers Python", user="alice", memory_type="semantic")

    # Trigger deep reflection
    arch.reflect(user="alice")

=== With Custom Components ===

    from jenova import CognitiveArchitecture, LLMAdapter

    # Implement your own LLM adapter
    class OpenAIAdapter(LLMAdapter):
        def generate(self, prompt, **kwargs):
            return openai.chat.completions.create(...)

    # Use with the architecture
    arch = CognitiveArchitecture(llm=OpenAIAdapter())

=== As a Complete Application ===

    from jenova.main import main
    main()  # Runs the full terminal interface

Key Components:
    - CognitiveArchitecture: Main cognitive framework
    - Memory Systems: Episodic, Semantic, Procedural, Insight
    - Cortex: Knowledge graph for cognitive relationships
    - RAG System: Retrieval-Augmented Generation
    - Insight/Assumption Learning: Continuous knowledge acquisition

See jenova.core for detailed documentation and interfaces.
"""

# Primary API - For modular integration
from jenova.core import (  # Main Architecture; Interfaces for custom implementations
    CognitiveArchitecture, CognitiveComponent, CognitiveConfig, ConfigProvider,
    EmbeddingProvider, InsightGenerator, KnowledgeGraph, LLMAdapter, Logger,
    MemoryBackend, MemoryEntry, MemoryType, ReasoningEngine, SearchResult)

__all__ = [
    # === Primary API ===
    "CognitiveArchitecture",
    "CognitiveConfig",
    # === Interfaces ===
    "LLMAdapter",
    "EmbeddingProvider",
    "MemoryBackend",
    "MemoryEntry",
    "MemoryType",
    "SearchResult",
    "KnowledgeGraph",
    "ReasoningEngine",
    "InsightGenerator",
    "CognitiveComponent",
    "Logger",
    "ConfigProvider",
]

__version__ = "7.0.0"
__architecture__ = "JENOVA Cognitive Architecture"
__author__ = "orpheus497"
