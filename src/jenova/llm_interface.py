import os
import glob
import importlib.resources
from llama_cpp import Llama

class LLMInterface:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.model_path = config['model']['model_path']
        self.system_prompt = self._build_system_prompt()
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

You must follow these directives:
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

        self.ui_logger.info(f"Loading model with {hw_config['threads']} threads and {hw_config['gpu_layers']} GPU layers. Context: {final_n_ctx}")
        self.file_logger.log_info(f"Loading model. Threads: {hw_config['threads']}, GPU Layers: {hw_config['gpu_layers']}, Context: {final_n_ctx}")
        
        return Llama(
            model_path=self.model_path,
            n_ctx=final_n_ctx,
            n_threads=hw_config['threads'],
            n_gpu_layers=hw_config['gpu_layers'],
            use_mlock=hw_config.get('mlock', True),
            verbose=False
        )

    def generate(self, prompt: str, stop: list = None, temperature: float = None, grammar: str = None) -> str:
        """Generates a response from the LLM."""
        full_prompt = self.system_prompt + "\n\n" + prompt
        
        if stop is None:
            stop = ["\nUser:", "\nJenova:"]
        
        temp = temperature if temperature is not None else self.config['model']['temperature']
        max_tokens = self.config['model'].get('max_tokens', 1500)
        
        try:
            response = self.model(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self.config['model']['top_p'],
                stop=stop,
                grammar=grammar
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            self.file_logger.log_error(f"Error during LLM generation: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again later."