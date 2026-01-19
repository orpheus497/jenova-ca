##Script function and purpose: Memory package initialization - exposes Memory class and types
"""Memory system for JENOVA."""

from jenova.memory.types import MemoryResult, MemoryType
from jenova.memory.memory import Memory

__all__ = ["Memory", "MemoryResult", "MemoryType"]
