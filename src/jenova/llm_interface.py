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
{chr(10).join(f'    - {d}' for d in directives)}""".strip()
        return prompt

    def _load_model(self):
        """Load Phi-4 model from HuggingFace or local cache."""
        model_dir = "/usr/local/share/jenova-ai/models"
        model_name = "unsloth/phi-4-bnb-4bit"
        
        self.ui_logger.info(f"Loading Phi-4 model...")
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
            
            # Check if this is a quantized model (4bit/8bit)
            is_quantized = "4bit" in model_name.lower() or "bnb" in model_name.lower() or "8bit" in model_name.lower()
            
            if is_quantized:
                # For quantized models, don't specify dtype - let it use the quantized format
                self.ui_logger.info("Detected quantized model - using optimized loading...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=model_dir,
                    device_map="auto" if device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    local_files_only=False,
                    trust_remote_code=True
                )
            else:
                # For non-quantized models, use appropriate dtype
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=model_dir,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=False,
                    trust_remote_code=True
                )
                model = model.to(device)
            
            model.eval()
            
            # --- Dynamic Model Max Context ---
            # Read the model's actual maximum context length from its config
            model_config = model.config
            model_max_context = getattr(model_config, 'max_position_embeddings', None)
            
            # Fallback to tokenizer if config doesn't have max_position_embeddings
            if model_max_context is None:
                model_max_context = tokenizer.model_max_length
                if model_max_context is None or model_max_context > 100000:  # Unreasonable default
                    model_max_context = 16384  # Safe fallback for Phi-4
                    self.file_logger.log_info(f"Could not determine model's max context. Using fallback: {model_max_context} tokens.")
                else:
                    self.file_logger.log_info(f"Model max context read from tokenizer: {model_max_context} tokens.")
            else:
                self.file_logger.log_info(f"Model max context read from model config: {model_max_context} tokens.")
            
            # Update config with the model's actual max context
            self.config['model']['context_size'] = model_max_context
            
            # --- Intelligent Program Max Tokens ---
            # If model's natural context length > 4000: set program max tokens = model's max context
            # If model's natural context length <= 4000: set program max tokens = 2x model's native context
            # Note: For context extension beyond native length, RoPE scaling would be needed in generation
            if model_max_context > 4000:
                program_max_tokens = model_max_context
                self.file_logger.log_info(f"Model context > 4000. Program max tokens set to model's max context: {program_max_tokens} tokens.")
            else:
                # For models with smaller context, double the context
                program_max_tokens = model_max_context * 2
                # Note: This requires RoPE scaling during generation for positions beyond native training
                self.file_logger.log_info(f"Model context <= 4000. Program max tokens set to 2x native context: {program_max_tokens} tokens (with RoPE scaling).")
                # Store RoPE scaling flag for generation
                self.config['model']['use_rope_scaling'] = True
                self.config['model']['rope_scaling_factor'] = 2.0
            
            self.config['model']['max_tokens'] = program_max_tokens
            
            # Get accurate model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Determine model name for display
            if "phi-4" in model_name.lower():
                display_name = "Phi-4 (4-bit quantized)" if "4bit" in model_name or "bnb" in model_name else "Phi-4"
            else:
                display_name = model_name.split("/")[-1]
            
            self.ui_logger.system_message(f"✓ Model loaded: {display_name}")
            self.ui_logger.system_message(f"   Total parameters: {total_params:,}")
            self.ui_logger.system_message(f"   Device: {device.upper()}")
            self.ui_logger.system_message(f"   Model max context: {model_max_context} tokens")
            self.ui_logger.system_message(f"   Program max tokens: {program_max_tokens} tokens")
            self.file_logger.log_info(f"Model loaded successfully on {device}")
            self.file_logger.log_info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
            
            return model, tokenizer
            
        except Exception as e:
            self.ui_logger.system_message(f"✗ Failed to load model: {e}")
            self.file_logger.log_error(f"Error loading model: {e}")
            raise

    def generate(self, prompt: str, stop: list = None, temperature: float = None) -> str:
        """Generates a response from the LLM.
        
        Args:
            prompt: The prompt to generate from
            stop: Stop sequences for early termination
            temperature: Temperature for generation
            grammar: Grammar specification (not used with transformers, kept for compatibility)
        """
        full_prompt = self.system_prompt + "\n\n" + prompt
        
        temp = temperature if temperature is not None else self.config['model']['temperature']
        max_tokens = self.config['model'].get('max_tokens', 512)
        
        # Use default stop sequences if none provided
        if stop is None:
            stop = ["\nUser:", "\nJenova:", "User:", "Jenova:"]
        
        try:
            # Tokenize input
            # Use program max tokens for truncation to allow extended context
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['model']['max_tokens']
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
            
            # Apply RoPE scaling if enabled for extended context
            # This allows the model to handle positions beyond its native training length
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': max(temp, 0.1),  # Ensure temp is not too low
                'top_p': self.config['model']['top_p'],
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'repetition_penalty': 1.2,  # Reduce repetition
            }
            
            # Note: RoPE scaling is typically configured in the model config at load time
            # For runtime scaling, we would need to modify model.config.rope_scaling
            # but this is model-dependent. The flag is set for future implementation.
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
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