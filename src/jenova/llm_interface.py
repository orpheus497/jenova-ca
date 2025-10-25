import os
import getpass
from llama_cpp import Llama

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
            # llama-cpp-python cleanup
            del self.llm
            self.llm = None
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
        """Generates a response from the LLM using llama-cpp-python.
        
        Args:
            prompt: The prompt to generate from
            stop: Stop sequences for early termination
            temperature: Temperature for generation (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
        """
        full_prompt = self.system_prompt + "\n\n" + prompt
        
        # Get parameters from config or use provided values
        temp = temperature if temperature is not None else self.config['model']['temperature']
        max_new_tokens = max_tokens if max_tokens is not None else self.config['model'].get('max_tokens', 512)
        
        # Use default stop sequences if none provided
        if stop is None:
            stop = ["\nUser:", "\nJenova:", "User:", "Jenova:"]
        
        try:
            # Generate using llama-cpp-python
            response = self.llm.create_completion(
                prompt=full_prompt,
                max_tokens=max_new_tokens,
                temperature=max(temp, 0.1),  # Ensure temp is not too low
                top_p=self.config['model']['top_p'],
                stop=stop,
                echo=False
            )
            
            # Extract text from response
            if response and 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['text']
            else:
                generated_text = ""
            
            # Clean up response
            generated_text = generated_text.strip()
            
            return generated_text
            
        except Exception as e:
            self.file_logger.log_error(f"Error during LLM generation: {e}")
            import traceback
            self.file_logger.log_error(f"Traceback: {traceback.format_exc()}")
            return ""
