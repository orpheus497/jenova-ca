# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 21: Full Project Remediation & Modernization
# Core Application Module - Structured application initialization and lifecycle

"""
JENOVA Cognitive Architecture - Core Module

This module provides the foundational architecture for building stateful,
learning-capable AI systems. It can be used as a complete application or
integrated as a modular cognitive layer in other AI programs.

=== Quick Start ===

    # Create a cognitive architecture with default components
    >>> from jenova.core import CognitiveArchitecture
    >>> arch = CognitiveArchitecture.create_default()
    >>> response = arch.think("Hello, how are you?", user="alice")

=== Pluggable Integration ===

    # Use with custom LLM and memory backend
    >>> from jenova.core import CognitiveArchitecture
    >>> from jenova.core.interfaces import LLMAdapter, MemoryBackend
    >>>
    >>> class MyLLM(LLMAdapter):
    ...     def generate(self, prompt, **kwargs):
    ...         return my_api_call(prompt)
    ...
    >>> arch = CognitiveArchitecture(llm=MyLLM())

=== Full Application ===

    # Run the complete JENOVA terminal application
    >>> from jenova.core import Application
    >>> app = Application()
    >>> app.run()

Key Components:
    - CognitiveArchitecture: Main cognitive framework (pluggable)
    - CognitiveConfig: Configuration for the architecture
    - Application: Full terminal-based application
    - DependencyContainer: Dependency injection for components
    - ComponentLifecycle: Component lifecycle management

Interfaces (for custom implementations):
    - LLMAdapter: Wrap any LLM (OpenAI, Anthropic, local)
    - MemoryBackend: Use any vector store
    - EmbeddingProvider: Use any embedding model
    - KnowledgeGraph: Custom knowledge graph implementation
"""

# Full Application - For running as standalone application
from jenova.core.application import Application
# Core Architecture - Main entry point for modular use
from jenova.core.architecture import CognitiveArchitecture, CognitiveConfig
from jenova.core.bootstrap import ApplicationBootstrapper
from jenova.core.container import DependencyContainer
# Interfaces - For custom implementations
from jenova.core.interfaces import (  # LLM Interfaces; Memory Interfaces; Component Interfaces; Utility Interfaces
    CognitiveComponent, ConfigProvider, EmbeddingProvider, InsightGenerator,
    KnowledgeGraph, LLMAdapter, Logger, MemoryBackend, MemoryEntry, MemoryType,
    ReasoningEngine, SearchResult)
from jenova.core.lifecycle import ComponentLifecycle, LifecyclePhase

__all__ = [
    # === Primary API ===
    "CognitiveArchitecture",
    "CognitiveConfig",
    # === Full Application ===
    "Application",
    "ApplicationBootstrapper",
    "DependencyContainer",
    "ComponentLifecycle",
    "LifecyclePhase",
    # === Interfaces for Custom Implementations ===
    # LLM
    "LLMAdapter",
    "EmbeddingProvider",
    # Memory
    "MemoryBackend",
    "MemoryEntry",
    "MemoryType",
    "SearchResult",
    # Components
    "CognitiveComponent",
    "ReasoningEngine",
    "InsightGenerator",
    "KnowledgeGraph",
    # Utilities
    "Logger",
    "ConfigProvider",
]

__version__ = "7.0.0"
__architecture__ = "JENOVA Cognitive Architecture"
__author__ = "orpheus497"
