##Script function and purpose: LLM package initialization - exposes LLMInterface
"""LLM interface for JENOVA."""

from jenova.llm.interface import LLMInterface
from jenova.llm.types import Completion, GenerationParams, Prompt

__all__ = ["LLMInterface", "Prompt", "Completion", "GenerationParams"]
