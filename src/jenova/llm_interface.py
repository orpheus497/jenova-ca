import os
from llama_cpp import Llama

class LLMInterface:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.system_prompt = self._build_system_prompt()
        self.llm = self._load_model()
    
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
{chr(10).join(f'    - {d}' for d in directives)}""".strip()
        return prompt

    def _load_model(self):
        """Load GGUF model using llama-cpp-python."""
        model_config = self.config.get('model', {})
        model_path = model_config.get('model_path', './models/model.gguf')
        threads = model_config.get('threads', 8)
        gpu_layers = model_config.get('gpu_layers', -1)
        use_mlock = model_config.get('mlock', True)
        n_batch = model_config.get('n_batch', 512)
        
        self.ui_logger.info(f"Loading GGUF model from {model_path}...")
        self.file_logger.log_info(f"Model path: {model_path}")
        self.file_logger.log_info(f"Threads: {threads}, GPU layers: {gpu_layers}, mlock: {use_mlock}, batch: {n_batch}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}\n\n" \
                       f"Please download a GGUF model and place it at {model_path}\n" \
                       f"See models/README.md for download instructions."
            self.ui_logger.system_message(f"✗ {error_msg}")
            self.file_logger.log_error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not model_path.endswith('.gguf'):
            warning_msg = f"Warning: Model file does not have .gguf extension: {model_path}"
            self.ui_logger.system_message(f"⚠ {warning_msg}")
            self.file_logger.log_info(warning_msg)
        
        try:
            # Load model with llama-cpp-python
            self.ui_logger.info("Initializing llama-cpp-python...")
            
            llm = Llama(
                model_path=model_path,
                n_threads=threads,
                n_gpu_layers=gpu_layers,
                use_mlock=use_mlock,
                n_batch=n_batch,
                n_ctx=model_config.get('context_size', 2048),
                verbose=False
            )
            
            # Log GPU usage
            if gpu_layers > 0 or gpu_layers == -1:
                self.ui_logger.system_message(f"🎮 GPU Acceleration: Enabled (GPU layers: {gpu_layers})")
                self.file_logger.log_info(f"GPU acceleration enabled with {gpu_layers} layers")
            else:
                self.ui_logger.system_message(f"💻 Running on CPU only")
                self.file_logger.log_info(f"Running on CPU")
            
            # Get model metadata if available
            try:
                metadata = llm.metadata
                if metadata:
                    self.file_logger.log_info(f"Model metadata: {metadata}")
            except:
                pass
            
            self.ui_logger.system_message(f"✓ Model loaded: {os.path.basename(model_path)}")
            self.ui_logger.system_message(f"   Context size: {model_config.get('context_size', 2048)} tokens")
            self.ui_logger.system_message(f"   Max generation: {model_config.get('max_tokens', 512)} tokens")
            self.file_logger.log_info(f"Model loaded successfully from {model_path}")
            
            return llm
            
        except Exception as e:
            error_msg = f"Failed to load model: {e}\n\n" \
                       f"Troubleshooting:\n" \
                       f"- Ensure the model file is a valid GGUF format\n" \
                       f"- For GPU support, install CUDA toolkit and rebuild llama-cpp-python\n" \
                       f"- Try reducing gpu_layers if out of VRAM\n" \
                       f"- Check that you have enough RAM for the model"
            self.ui_logger.system_message(f"✗ {error_msg}")
            self.file_logger.log_error(f"Error loading model: {e}")
            raise

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
            
            # If response is empty, provide fallback
            if not generated_text:
                generated_text = "I understand your query, but I need more context to provide a helpful response."
            
            return generated_text
            
        except Exception as e:
            self.file_logger.log_error(f"Error during LLM generation: {e}")
            import traceback
            self.file_logger.log_error(f"Traceback: {traceback.format_exc()}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again later."