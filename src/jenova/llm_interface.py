# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module provides the interface to the Language Learning Model.
"""

import time



class LLMInterface:
    def __init__(self, config, ui_logger, file_logger, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.system_prompt = self._build_system_prompt()
        self.llm = llm

    def close(self):
        """Cleans up the LLM resources."""
        if self.llm:
            del self.llm
            self.llm = None
            if self.file_logger:
                self.file_logger.log_info("LLM model resources released.")

    def _build_system_prompt(self) -> str:
        """Builds a robust, persistent system prompt to ground the AI."""
        persona = self.config['persona']
        identity = persona.get('identity', {})
        directives = persona.get('directives', [])

        prompt = f"""You are {identity.get('name', 'Jenova')}, a {identity.get('type', 'personalized AI assistant')}. \
Your origin story: {identity.get('origin_story', 'You are a helpful assistant.')} \
Your creator is {identity.get('creator', 'a developer')}. You and the user are separate entities.

You must follow these directives:
{chr(10).join(f'    - {d}' for d in directives)}"""
        return prompt

    def generate(self, prompt: str, stop: list = None, temperature: float = None, max_tokens: int = None) -> str:
        """Generates a response from the LLM with retry logic."""
        full_prompt = self.system_prompt + "\n\n" + prompt

        temp = temperature if temperature is not None else self.config['model']['temperature']
        max_new_tokens = max_tokens if max_tokens is not None else self.config['model'].get(
            'max_tokens', 512)
        stop_sequences = stop if stop is not None else [
            "\nUser:", "\nJenova:", "User:", "Jenova:"]

        max_retries = 3
        backoff_factor = 2

        for attempt in range(max_retries):
            try:
                response = self.llm.create_completion(
                    prompt=full_prompt,
                    max_tokens=max_new_tokens,
                    temperature=max(temp, 0.1),
                    top_p=self.config['model']['top_p'],
                    stop=stop_sequences,
                    echo=False
                )

                if response and 'choices' in response and len(response['choices']) > 0:
                    generated_text = response['choices'][0]['text'].strip()
                    return generated_text
                else:
                    raise ValueError("Invalid or empty response from LLM")

            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(
                        f"LLM generation attempt {attempt + 1} failed: {e}")
                if attempt + 1 == max_retries:
                    if self.file_logger:
                        self.file_logger.log_error(
                            "Max retries reached. LLM generation failed.")
                    import traceback
                    if self.file_logger:
                        self.file_logger.log_error(
                            f"Traceback: {traceback.format_exc()}")
                    return ""

                sleep_time = backoff_factor ** attempt
                if self.ui_logger:
                    self.ui_logger.system_message(
                        f"LLM generation failed. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        return ""
