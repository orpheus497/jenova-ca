##Script function and purpose: Package initialization for JENOVA cognitive architecture
"""
JENOVA - Self-aware AI Cognitive Architecture

A cognitive architecture providing:
- Multi-layered memory (unified ChromaDB-based)
- Graph-based cognitive core (dict-based, no networkx)
- RAG-based response generation with deliberate planning
- Assumption tracking and verification
- Fine-tunable embedding model
"""

from jenova.exceptions import JenovaError

__version__ = "4.1.0"
__all__ = ["JenovaError", "__version__"]
