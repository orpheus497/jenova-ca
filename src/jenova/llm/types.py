##Script function and purpose: LLM type definitions - Prompt, Completion, and generation parameters
"""
LLM Types

Type definitions for LLM operations. Provides structured types
for prompts, completions, and generation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


##Class purpose: Generation parameters for LLM inference
@dataclass
class GenerationParams:
    """Parameters for LLM generation."""
    
    max_tokens: int = 1024
    """Maximum tokens to generate."""
    
    temperature: float = 0.7
    """Sampling temperature (0-2)."""
    
    top_p: float = 0.9
    """Top-p (nucleus) sampling."""
    
    top_k: int = 40
    """Top-k sampling."""
    
    repeat_penalty: float = 1.1
    """Repetition penalty."""
    
    stop: list[str] = field(default_factory=lambda: ["</s>", "User:", "\n\nUser:"])
    """Stop sequences."""


##Class purpose: Structured prompt for LLM
@dataclass
class Prompt:
    """A structured prompt for LLM generation."""
    
    system: str
    """System prompt defining AI behavior."""
    
    context: str = ""
    """Retrieved context from memory/knowledge."""
    
    user_message: str = ""
    """Current user message."""
    
    chat_history: list[tuple[str, str]] = field(default_factory=list)
    """List of (role, content) tuples for chat history."""
    
    ##Method purpose: Format as chat template string
    def format_chat(self, template: str = "chatml") -> str:
        """
        Format prompt using chat template.
        
        Args:
            template: Template format (chatml, llama2, etc.)
            
        Returns:
            Formatted prompt string
        """
        ##Condition purpose: Handle ChatML format
        if template == "chatml":
            return self._format_chatml()
        ##Condition purpose: Handle Llama2 format
        elif template == "llama2":
            return self._format_llama2()
        else:
            return self._format_chatml()
    
    ##Method purpose: Format as ChatML template
    def _format_chatml(self) -> str:
        """Format using ChatML template."""
        parts = [f"<|im_start|>system\n{self.system}"]
        
        ##Condition purpose: Add context if present
        if self.context:
            parts.append(f"\n\nContext:\n{self.context}")
        
        parts.append("<|im_end|>")
        
        ##Loop purpose: Add chat history
        for role, content in self.chat_history:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        ##Condition purpose: Add current user message
        if self.user_message:
            parts.append(f"<|im_start|>user\n{self.user_message}<|im_end|>")
        
        parts.append("<|im_start|>assistant\n")
        
        return "\n".join(parts)
    
    ##Method purpose: Format as Llama2 template
    def _format_llama2(self) -> str:
        """Format using Llama2 template."""
        system_part = f"<<SYS>>\n{self.system}"
        
        ##Condition purpose: Add context if present
        if self.context:
            system_part += f"\n\nContext:\n{self.context}"
        
        system_part += "\n<</SYS>>"
        
        parts = [f"[INST] {system_part}"]
        
        ##Loop purpose: Add history (simplified for Llama2)
        for role, content in self.chat_history:
            if role == "user":
                parts.append(f"[INST] {content} [/INST]")
            else:
                parts.append(content)
        
        ##Condition purpose: Add current user message
        if self.user_message:
            parts.append(f"[INST] {self.user_message} [/INST]")
        
        return "\n".join(parts)


##Class purpose: LLM completion result
@dataclass
class Completion:
    """Result from LLM generation."""
    
    content: str
    """Generated text content."""
    
    finish_reason: Literal["stop", "length", "error"]
    """Why generation stopped."""
    
    tokens_generated: int
    """Number of tokens generated."""
    
    tokens_prompt: int
    """Number of tokens in prompt."""
    
    generation_time_ms: float
    """Time taken for generation in milliseconds."""
    
    ##Method purpose: Check if generation completed normally
    @property
    def is_complete(self) -> bool:
        """Check if generation finished normally."""
        return self.finish_reason == "stop"
    
    ##Method purpose: Calculate tokens per second
    @property
    def tokens_per_second(self) -> float:
        """Calculate generation speed."""
        ##Condition purpose: Avoid division by zero
        if self.generation_time_ms <= 0:
            return 0.0
        return (self.tokens_generated / self.generation_time_ms) * 1000
