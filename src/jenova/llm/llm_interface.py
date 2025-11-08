# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
LLM Interface for JENOVA with timeout protection and robust error handling.

This module provides a high-level interface to the language model with:
- Timeout protection to prevent hangs
- Retry logic with exponential backoff
- Detailed error messages
- Resource cleanup
"""

import time
from typing import Optional, List, Dict, Any

from jenova.infrastructure.timeout_manager import with_timeout, timeout, TimeoutError


class LLMInterface:
    """
    High-level interface to the language model.

    Features:
    - Automatic timeout protection
    - Retry logic with exponential backoff
    - System prompt management
    - Resource cleanup
    """

    def __init__(self, config: Dict[str, Any], ui_logger, file_logger, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.llm = llm
        self.system_prompt = self._build_system_prompt()

        # Generation settings
        self.default_temperature = config.get("model", {}).get("temperature", 0.7)
        self.default_max_tokens = config.get("model", {}).get("max_tokens", 512)
        self.default_top_p = config.get("model", {}).get("top_p", 0.95)

        # Retry settings
        self.max_retries = 3
        self.backoff_factor = 2

    def _build_system_prompt(self) -> str:
        """Builds a robust, persistent system prompt to ground the AI."""
        persona = self.config.get("persona", {})
        identity = persona.get("identity", {})
        directives = persona.get("directives", [])

        name = identity.get("name", "Jenova")
        ai_type = identity.get("type", "personalized AI assistant")
        origin = identity.get("origin_story", "You are a helpful assistant.")
        creator = identity.get("creator", "a developer")

        prompt = f"""You are {name}, a {ai_type}. \
Your origin story: {origin} \
Your creator is {creator}. You and the user are separate entities.

You must follow these directives:
{chr(10).join(f'    - {d}' for d in directives)}"""

        return prompt

    def _generate_with_timeout(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: List[str],
        timeout_seconds: int,
    ) -> Optional[str]:
        """
        Generate with timeout protection.

        Args:
            prompt: Full prompt (including system prompt)
            temperature: Temperature setting
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_sequences: List of stop sequences
            timeout_seconds: Timeout in seconds

        Returns:
            Generated text or None if timeout/error

        Raises:
            TimeoutError: If generation times out
        """
        with timeout(
            timeout_seconds, f"LLM generation timed out after {timeout_seconds}s"
        ):
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.1),  # Ensure min temperature
                top_p=top_p,
                stop=stop_sequences,
                echo=False,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["text"].strip()
            else:
                raise ValueError("Invalid or empty response from LLM")

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: int = 120,
    ) -> str:
        """
        Generate a response from the LLM with retry logic and timeout protection.

        Args:
            prompt: User prompt (system prompt added automatically)
            stop: Stop sequences (default: ["\\nUser:", "\\nJenova:", "User:", "Jenova:"])
            temperature: Temperature setting (default from config)
            max_tokens: Max tokens to generate (default from config)
            timeout_seconds: Timeout in seconds (default: 120)

        Returns:
            Generated text (empty string if all retries fail)
        """
        # Build full prompt
        full_prompt = self.system_prompt + "\n\n" + prompt

        # Use defaults if not specified
        temp = temperature if temperature is not None else self.default_temperature
        max_new_tokens = (
            max_tokens if max_tokens is not None else self.default_max_tokens
        )
        stop_sequences = (
            stop if stop is not None else ["\nUser:", "\nJenova:", "User:", "Jenova:"]
        )

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                result = self._generate_with_timeout(
                    prompt=full_prompt,
                    temperature=temp,
                    max_tokens=max_new_tokens,
                    top_p=self.default_top_p,
                    stop_sequences=stop_sequences,
                    timeout_seconds=timeout_seconds,
                )

                if result:
                    return result
                else:
                    raise ValueError("Empty response from LLM")

            except TimeoutError as e:
                if self.file_logger:
                    self.file_logger.log_error(
                        f"LLM generation attempt {attempt + 1} timed out: {e}"
                    )

                if attempt + 1 == self.max_retries:
                    if self.ui_logger:
                        self.ui_logger.error(
                            f"LLM generation timed out after {self.max_retries} attempts"
                        )
                    return ""

                sleep_time = self.backoff_factor**attempt
                if self.ui_logger:
                    self.ui_logger.system_message(
                        f"Generation timed out. Retrying in {sleep_time}s..."
                    )
                time.sleep(sleep_time)

            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(
                        f"LLM generation attempt {attempt + 1} failed: {e}"
                    )

                if attempt + 1 == self.max_retries:
                    if self.file_logger:
                        self.file_logger.log_error(
                            "Max retries reached. LLM generation failed."
                        )
                        import traceback

                        self.file_logger.log_error(
                            f"Traceback: {traceback.format_exc()}"
                        )

                    if self.ui_logger:
                        self.ui_logger.error(
                            "LLM generation failed after multiple attempts"
                        )
                    return ""

                sleep_time = self.backoff_factor**attempt
                if self.ui_logger:
                    self.ui_logger.system_message(
                        f"Generation failed. Retrying in {sleep_time}s..."
                    )
                time.sleep(sleep_time)

        return ""

    def generate_with_context(
        self,
        user_message: str,
        context: str = "",
        memory_context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate response with additional context from memory systems.

        Args:
            user_message: The user's message
            context: Additional context to include
            memory_context: Context from memory systems
            max_tokens: Maximum tokens to generate
            temperature: Temperature setting

        Returns:
            Generated response
        """
        # Build contextualized prompt
        prompt_parts = []

        if memory_context:
            prompt_parts.append(f"Relevant memories:\n{memory_context}\n")

        if context:
            prompt_parts.append(f"Context:\n{context}\n")

        prompt_parts.append(f"User: {user_message}\nJenova:")

        full_prompt = "\n".join(prompt_parts)

        return self.generate(
            prompt=full_prompt, max_tokens=max_tokens, temperature=temperature
        )

    def test_generation(self, test_prompt: str = "Hello, how are you?") -> bool:
        """
        Test if LLM generation works.

        Args:
            test_prompt: Simple test prompt

        Returns:
            True if generation works, False otherwise
        """
        try:
            result = self.generate(
                prompt=test_prompt, max_tokens=50, timeout_seconds=30
            )
            return len(result) > 0

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"LLM test generation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM configuration."""
        return {
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
            "top_p": self.default_top_p,
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
        }

    def update_system_prompt(self, new_directives: Optional[List[str]] = None):
        """
        Update the system prompt with new directives.

        Args:
            new_directives: New list of directives to use
        """
        if new_directives:
            self.config["persona"]["directives"] = new_directives

        self.system_prompt = self._build_system_prompt()

        if self.file_logger:
            self.file_logger.log_info("System prompt updated")

    def close(self):
        """Clean up LLM resources."""
        if self.llm:
            try:
                del self.llm
                self.llm = None

                if self.file_logger:
                    self.file_logger.log_info(
                        "LLM interface closed, resources released"
                    )

                # Force garbage collection
                import gc

                gc.collect()

            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Error closing LLM interface: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
