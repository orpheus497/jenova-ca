##Script function and purpose: LLM package initialization - exposes LLMInterface
"""LLM interface for JENOVA."""

from jenova.llm.types import Prompt, Completion, GenerationParams
from jenova.llm.interface import LLMInterface

__all__ = ["LLMInterface", "Prompt", "Completion", "GenerationParams"]
