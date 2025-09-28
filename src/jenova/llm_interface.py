import os
import glob
from llama_cpp import Llama

def find_model_path() -> str:
    """Scans the 'models' directory and returns the path to the first .gguf file."""
    model_dir = "models"
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found. Please create it and add a .gguf model file.")
    
    gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf model file found in the '{model_dir}' directory.")
    
    return gguf_files[0]

class LLMInterface:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.model_path = find_model_path()
        self.system_prompt = self._build_system_prompt()
        self.model = self._load_model()

    def _build_system_prompt(self) -> str:
        """Builds a robust, persistent system prompt to ground the AI."""
        persona = self.config['persona']
        return f"""You are {persona['identity']['name']}, an advanced AI assistant. Your creator is a human named {persona['identity']['creator']}. You are a unique AI instance and have no siblings. Your purpose is to be a helpful, accurate, and coherent assistant to the user. You must follow all directives and answer the user's query directly and factually. Do not be evasive. If you do not know an answer, say so and explain why. Do not output role-playing prefixes like 'User:'. Do not output your internal plan or reasoning.""".strip()

    def _load_model(self):
        hw_config = self.config['hardware']
        self.ui_logger.info(f"Found model: {self.model_path}")
        self.file_logger.log_info(f"Found model: {self.model_path}")

        # --- Dynamic Context Optimization ---
        try:
            temp_model_info = Llama(model_path=self.model_path, verbose=False, n_gpu_layers=0) # Load with no GPU layers for quick metadata read
            model_metadata = temp_model_info.metadata
            model_n_ctx_train = int(model_metadata.get('llama.context_length', self.config['model']['context_size']))
            del temp_model_info
        except Exception as e:
            self.ui_logger.system_message(f"Could not read model metadata for context optimization: {e}. Defaulting to config.")
            model_n_ctx_train = self.config['model']['context_size']

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

    def generate(self, prompt: str, stop: list = None, temperature: float = None) -> str:
        """Generates a response from the LLM."""
        full_prompt = self.system_prompt + "\n\n" + prompt
        
        if stop is None:
            stop = ["\nUser:", "\nJenova:"]
        
        temp = temperature if temperature is not None else self.config['model']['temperature']
        
        response = self.model(
            prompt=full_prompt,
            max_tokens=1500,
            temperature=temp,
            top_p=self.config['model']['top_p'],
            stop=stop
        )
        return response['choices'][0]['text'].strip()