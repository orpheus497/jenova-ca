import os
import glob
import importlib.resources
import multiprocessing
import time
from llama_cpp import Llama

def _llm_generation_worker(model_path, prompt, max_tokens, temperature, top_p, stop, grammar, n_ctx, n_threads, n_gpu_layers, use_mlock, result_queue):
    """Worker function to run LLM generation in a separate process.
    
    Args:
        model_path: Path to the GGUF model file
        prompt: Full prompt string to send to the LLM
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        stop: List of stop sequences
        grammar: Optional grammar string
        n_ctx: Context size
        n_threads: Number of CPU threads
        n_gpu_layers: Number of GPU layers
        use_mlock: Whether to lock model in memory
        result_queue: Multiprocessing queue to return the result
    """
    try:
        # Load the model in the worker process
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
            verbose=False
        )
        
        # Generate response
        response = model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            grammar=grammar
        )
        
        # Put the result in the queue
        result_queue.put(('success', response['choices'][0]['text'].strip()))
    except Exception as e:
        # Put the error in the queue
        result_queue.put(('error', str(e)))

class LLMInterface:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.model_path = config['model']['model_path']
        self.system_prompt = self._build_system_prompt()
        # Note: Model is loaded here for initialization and metadata extraction.
        # For actual generation, the model is reloaded in a separate process to prevent hangs.
        self.model = self._load_model()

    def close(self):
        """Cleans up the LLM resources."""
        if self.model:
            # The Llama object from llama-cpp-python does not have an explicit close() method.
            # Resources are released when the object is garbage collected.
            self.file_logger.log_info("LLM model resources released.")

    def _build_system_prompt(self) -> str:
        """Builds a robust, persistent system prompt to ground the AI."""
        persona = self.config['persona']
        identity = persona.get('identity', {})
        directives = persona.get('directives', [])

        prompt = f"""You are {identity.get('name', 'Jenova')}, a {identity.get('type', 'personalized AI assistant')}. \
Your origin story: {identity.get('origin_story', 'You are a helpful assistant.')} \
Your creator is {identity.get('creator', 'a developer')}. You and the user are separate entities.

_build_system_promptYou must follow these directives:
{chr(10).join(f"    - {d}" for d in directives)}

Answer the user's query directly and factually. Do not be evasive. If you do not know an answer, say so and explain why. Do not output role-playing prefixes like 'User:'. Do not output your internal plan or reasoning.""".strip()
        return prompt

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path) or os.path.isdir(self.model_path):
            self.ui_logger.info("Model path not configured or invalid. Searching for a model in the 'models/' directory...")
            self.file_logger.log_info("Model path not configured. Searching for models.")
            
            # Find the models directory within the package
            try:
                models_dir_traversable = importlib.resources.files('jenova') / 'models'
                models_dir = str(models_dir_traversable)
            except (ModuleNotFoundError, FileNotFoundError):
                # Fallback for development environments or if the package structure is unexpected
                models_dir = 'models'

            available_models = []
            if os.path.exists(models_dir):
                available_models = sorted(glob.glob(f"{models_dir}/**/*.gguf", recursive=True))

            if available_models:
                self.model_path = available_models[0]
                self.ui_logger.system_message(f"Found and selected model: {self.model_path}")
                self.file_logger.log_info(f"Auto-selected model: {self.model_path}")
                # Update the config in memory for this session
                self.config['model']['model_path'] = self.model_path
            else:
                self.ui_logger.system_message(f"No GGUF models found in the '{models_dir}/' directory.")
                self.file_logger.log_error("No GGUF models found.")
                self.ui_logger.system_message("Please download a GGUF-compatible model and place it in the 'models/' directory.")
                raise FileNotFoundError("No GGUF model found in the 'models/' directory.")

        hw_config = self.config['hardware']
        
        # Set hard defaults for threads and GPU layers (functional proclivity)
        # These are the intended hardware settings and should not be changed by automated systems
        n_threads = hw_config.get('threads', 16)
        if n_threads != 16:
            self.file_logger.log_info(f"Hardware config threads ({n_threads}) overridden to default: 16")
            n_threads = 16
        
        n_gpu_layers = hw_config.get('gpu_layers', -1)
        if n_gpu_layers != -1:
            self.file_logger.log_info(f"Hardware config gpu_layers ({n_gpu_layers}) overridden to default: -1 (all layers)")
            n_gpu_layers = -1
        
        self.ui_logger.info(f"Found model: {self.model_path}")
        self.file_logger.log_info(f"Found model: {self.model_path}")

        # --- Dynamic Context Optimization ---
        try:
            # Create a temporary Llama instance to fetch metadata without using a context manager
            temp_model_info = Llama(model_path=self.model_path, verbose=False, n_gpu_layers=0)
            model_metadata = temp_model_info.metadata
            model_n_ctx_train = int(model_metadata.get('llama.context_length', self.config['model']['context_size']))
            # Since we are not using a with statement, we don't need to worry about cleanup
        except Exception as e:
            # This will catch errors during Llama object creation for valid paths
            self.ui_logger.system_message(f"A critical failure occurred while reading model metadata: {e}")
            self.file_logger.log_error(f"Error reading model metadata: {e}")
            raise
        
        config_n_ctx = self.config['model']['context_size']
        final_n_ctx = model_n_ctx_train

        if config_n_ctx != model_n_ctx_train:
            self.ui_logger.system_message(f"Notice: Model context size ({model_n_ctx_train}) is being used, overriding config value ({config_n_ctx}).")
            self.file_logger.log_info(f"Using model context size: {model_n_ctx_train}")
        # --- End Optimization ---

        self.ui_logger.info(f"Loading model with {n_threads} threads and {n_gpu_layers} GPU layers. Context: {final_n_ctx}")
        self.file_logger.log_info(f"Loading model. Threads: {n_threads}, GPU Layers: {n_gpu_layers}, Context: {final_n_ctx}")
        
        return Llama(
            model_path=self.model_path,
            n_ctx=final_n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mlock=hw_config.get('mlock', True),
            verbose=False
        )

    def generate(self, prompt: str, stop: list = None, temperature: float = None, grammar: str = None) -> str:
        """Generates a response from the LLM using a separate process with timeout protection."""
        full_prompt = self.system_prompt + "\n\n" + prompt
        
        if stop is None:
            stop = ["\nUser:", "\nJenova:"]
        
        temp = temperature if temperature is not None else self.config['model']['temperature']
        max_tokens = self.config['model'].get('max_tokens', 1500)
        timeout = self.config['model'].get('generation_timeout', 300)  # Default 300 seconds
        
        # Log the prompt being sent
        self.file_logger.log_info(f"LLM generation starting. Prompt length: {len(full_prompt)} chars, Max tokens: {max_tokens}, Temperature: {temp}, Timeout: {timeout}s")
        self.file_logger.log_info(f"Prompt preview (first 200 chars): {full_prompt[:200]}...")
        
        start_time = time.time()
        
        try:
            # Create a queue to receive results from the worker process
            result_queue = multiprocessing.Queue()
            
            # Get model parameters
            hw_config = self.config['hardware']
            n_threads = 16  # Hard-coded as per requirement
            n_gpu_layers = -1  # Hard-coded as per requirement
            use_mlock = hw_config.get('mlock', True)
            
            # Get context size from config
            config_n_ctx = self.config['model']['context_size']
            
            # Start the generation process
            process = multiprocessing.Process(
                target=_llm_generation_worker,
                args=(
                    self.model_path,
                    full_prompt,
                    max_tokens,
                    temp,
                    self.config['model']['top_p'],
                    stop,
                    grammar,
                    config_n_ctx,
                    n_threads,
                    n_gpu_layers,
                    use_mlock,
                    result_queue
                )
            )
            
            self.file_logger.log_info(f"LLM generation process started (PID will be assigned)")
            process.start()
            
            # Wait for the process to complete with timeout
            process.join(timeout=timeout)
            
            if process.is_alive():
                # Process timed out
                self.file_logger.log_error(f"LLM generation TIMEOUT after {timeout} seconds. Terminating process.")
                self.ui_logger.system_message(f"LLM generation timed out after {timeout} seconds. The process has been terminated.")
                process.terminate()
                process.join(timeout=5)  # Give it 5 seconds to terminate gracefully
                
                if process.is_alive():
                    # Force kill if still alive
                    self.file_logger.log_error("LLM generation process did not terminate gracefully. Force killing.")
                    process.kill()
                    process.join()
                
                elapsed_time = time.time() - start_time
                self.file_logger.log_error(f"LLM generation failed after {elapsed_time:.2f} seconds due to timeout.")
                return "I'm sorry, the response generation took too long and was terminated to prevent the system from freezing. Please try again with a simpler query."
            
            # Process completed, get the result
            if not result_queue.empty():
                status, result = result_queue.get()
                elapsed_time = time.time() - start_time
                
                if status == 'success':
                    self.file_logger.log_info(f"LLM generation completed successfully in {elapsed_time:.2f} seconds. Response length: {len(result)} chars")
                    return result
                else:
                    # Error occurred in worker
                    self.file_logger.log_error(f"Error during LLM generation in worker process: {result}")
                    self.file_logger.log_error(f"LLM generation failed after {elapsed_time:.2f} seconds.")
                    return "I'm sorry, I'm having trouble generating a response right now. Please try again later."
            else:
                # No result in queue (shouldn't happen)
                elapsed_time = time.time() - start_time
                self.file_logger.log_error(f"LLM generation process completed but no result was returned after {elapsed_time:.2f} seconds.")
                return "I'm sorry, I'm having trouble generating a response right now. Please try again later."
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.file_logger.log_error(f"Unexpected error during LLM generation: {e}")
            self.file_logger.log_error(f"LLM generation failed after {elapsed_time:.2f} seconds.")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again later."