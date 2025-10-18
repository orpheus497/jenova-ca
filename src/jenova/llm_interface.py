import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMInterface:
    def __init__(self, config, ui_logger, file_logger):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.system_prompt = self._build_system_prompt()
        self.model, self.tokenizer = self._load_model()
    
    def close(self):
        """Cleans up the LLM resources."""
        if self.model:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        """Load TinyLlama model from HuggingFace or local cache."""
        model_dir = "/usr/local/share/jenova-ai/models"
        model_name = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
        
        self.ui_logger.info(f"Loading TinyLlama model...")
        self.file_logger.log_info(f"Loading model: {model_name}")
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            self.ui_logger.system_message(f"🎮 GPU Acceleration: Enabled (CUDA)")
            self.file_logger.log_info(f"GPU acceleration enabled: CUDA")
        else:
            device = "cpu"
            self.ui_logger.system_message(f"💻 Running on CPU")
            self.file_logger.log_info(f"Running on CPU")
        
        try:
            # Load tokenizer
            self.ui_logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=False,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            self.ui_logger.info("Loading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=model_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=False,
                trust_remote_code=True
            )
            
            model = model.to(device)
            model.eval()
            
            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.ui_logger.system_message(f"✓ Model loaded: TinyLlama-1.1B")
            self.ui_logger.system_message(f"   Total parameters: {total_params:,}")
            self.ui_logger.system_message(f"   Device: {device.upper()}")
            self.ui_logger.system_message(f"   Model max context: {tokenizer.model_max_length} tokens")
            self.file_logger.log_info(f"Model loaded successfully on {device}")
            self.file_logger.log_info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
            
            return model, tokenizer
            
        except Exception as e:
            self.ui_logger.system_message(f"✗ Failed to load model: {e}")
            self.file_logger.log_error(f"Error loading model: {e}")
            raise

    def generate(self, prompt: str, stop: list = None, temperature: float = None, grammar: str = None, thinking_process=None) -> str:
        """Generates a response from the LLM.
        
        Args:
            prompt: The prompt to generate from
            stop: Stop sequences for early termination
            temperature: Temperature for generation
            grammar: Grammar specification (not used with transformers, kept for compatibility)
            thinking_process: Optional context manager for thinking status
        """
        full_prompt = self.system_prompt + "\n\n" + prompt
        
        temp = temperature if temperature is not None else self.config['model']['temperature']
        max_tokens = self.config['model'].get('max_tokens', 512)
        
        # Use default stop sequences if none provided
        if stop is None:
            stop = ["\nUser:", "\nJenova:", "User:", "Jenova:"]
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['model']['context_size']
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Prepare stopping criteria if stop sequences provided
            stop_token_ids = []
            if stop:
                for stop_seq in stop:
                    tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                    if tokens:
                        stop_token_ids.extend(tokens)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temp, 0.1),  # Ensure temp is not too low
                    top_p=self.config['model']['top_p'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            if generated_text.startswith(full_prompt):
                response = generated_text[len(full_prompt):].strip()
            else:
                response = generated_text.strip()
            
            # Apply stop sequences manually (remove everything after stop sequence)
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0].strip()
            
            # Clean up response
            response = response.strip()
            
            # If response is empty, provide fallback
            if not response:
                response = "I understand your query, but I need more context to provide a helpful response."
            
            return response
            
        except Exception as e:
            self.file_logger.log_error(f"Error during LLM generation: {e}")
            import traceback
            self.file_logger.log_error(f"Traceback: {traceback.format_exc()}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again later."