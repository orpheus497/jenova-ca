##Script function and purpose: LLM interface wrapping llama-cpp-python for GGUF inference
"""
LLM Interface

Wrapper around llama-cpp-python for GGUF model inference.
Provides clean typed interface for text generation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from jenova.config.models import ModelConfig, HardwareConfig
from jenova.exceptions import LLMLoadError, LLMGenerationError
from jenova.llm.types import Prompt, Completion, GenerationParams
from jenova.utils.errors import sanitize_path_for_error

if TYPE_CHECKING:
    from llama_cpp import Llama

##Step purpose: Define constant for llama.cpp "all layers" GPU offload value
LLAMA_CPP_ALL_LAYERS: int = -1  # llama.cpp convention: -1 means offload all layers to GPU


##Class purpose: LLM wrapper for GGUF model inference
class LLMInterface:
    """
    Interface for LLM operations using llama-cpp-python.
    
    Wraps a GGUF model for text generation with typed inputs/outputs.
    """
    
    ##Method purpose: Initialize with model and hardware configs
    def __init__(
        self,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
    ) -> None:
        """
        Initialize LLM interface.
        
        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
        """
        ##Step purpose: Store configuration
        self._model_config = model_config
        self._hardware_config = hardware_config
        self._llm: Llama | None = None
    
    ##Method purpose: Load the LLM model
    def load(self, model_path: Path | None = None) -> None:
        """
        Load the LLM model.
        
        Args:
            model_path: Path to GGUF file (overrides config)
            
        Raises:
            LLMLoadError: If model loading fails
        """
        ##Step purpose: Determine model path
        path = model_path or self._model_config.model_path
        
        ##Condition purpose: Handle 'auto' model path
        if path == "auto":
            path = self._find_model()
        
        ##Condition purpose: Validate path exists
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            ##Step purpose: Sanitize path in error message
            safe_path = sanitize_path_for_error(path)
            raise LLMLoadError(safe_path, "File not found")
        
        ##Error purpose: Catch model loading errors
        try:
            from llama_cpp import Llama
            
            ##Step purpose: Determine GPU layers
            gpu_layers = self._resolve_gpu_layers()
            
            ##Step purpose: Check GPU availability if GPU layers requested
            if gpu_layers != 0:
                gpu_available = self._check_gpu_availability()
                if not gpu_available:
                    logger.warning(
                        "gpu_requested_but_unavailable",
                        gpu_layers=gpu_layers,
                        message="GPU layers requested but CUDA/Metal/Vulkan not available. Falling back to CPU.",
                    )
                    gpu_layers = 0
            
            ##Action purpose: Load the model
            self._llm = Llama(
                model_path=str(path),
                n_ctx=self._model_config.context_length,
                n_threads=self._hardware_config.effective_threads,
                n_gpu_layers=gpu_layers,
                verbose=False,
            )
            
            ##Action purpose: Log GPU usage
            if gpu_layers != 0:
                logger.info(
                    "model_loaded_with_gpu",
                    gpu_layers=gpu_layers,
                    model_path=str(path),
                )
            else:
                logger.info("model_loaded_cpu_only", model_path=str(path))
        except Exception as e:
            ##Step purpose: Sanitize path in error message
            safe_path = sanitize_path_for_error(path)
            raise LLMLoadError(safe_path, str(e)) from e
    
    ##Method purpose: Find model file automatically
    def _find_model(self) -> Path:
        """Find a GGUF model file in common locations."""
        ##Step purpose: Define search paths
        search_paths = [
            Path("models"),
            Path(".jenova-ai/models"),
            Path.home() / ".cache" / "jenova" / "models",
        ]
        
        ##Loop purpose: Search each path for GGUF files
        for search_path in search_paths:
            ##Condition purpose: Skip if path doesn't exist
            if not search_path.exists():
                continue
            
            ##Step purpose: Find GGUF files
            gguf_files = list(search_path.glob("*.gguf"))
            ##Condition purpose: Return first found
            if gguf_files:
                return gguf_files[0]
        
        raise LLMLoadError("auto", "No GGUF files found in search paths")
    
    ##Method purpose: Resolve GPU layer configuration
    def _resolve_gpu_layers(self) -> int:
        """Resolve GPU layers setting to integer."""
        gpu = self._hardware_config.gpu_layers
        
        ##Condition purpose: Handle string values
        if gpu == "none":
            return 0
        elif gpu in ("all", "auto"):
            ##Step purpose: Use all GPU layers for 'all' or 'auto' settings
            return LLAMA_CPP_ALL_LAYERS
        else:
            return gpu
    
    ##Method purpose: Check if GPU is available for llama-cpp-python
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for llama-cpp-python.
        
        Checks for CUDA, Metal, or Vulkan support by attempting to
        detect available GPU backends. This is a best-effort check;
        llama-cpp-python will handle actual GPU availability when loading.
        
        Returns:
            True if GPU backend appears available, False otherwise.
        """
        ##Error purpose: Handle import errors gracefully
        try:
            import os
            import platform
            
            ##Step purpose: Check for CUDA environment variables
            ##Condition purpose: CUDA_VISIBLE_DEVICES indicates CUDA setup
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                # CUDA environment is set, likely available
                return True
            
            ##Step purpose: Check for NVIDIA GPU via nvidia-smi (POSIX: FreeBSD, Linux)
            ##Condition purpose: Try to detect NVIDIA GPU on POSIX systems
            if platform.system() in ("Linux", "FreeBSD"):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi"],
                        capture_output=True,
                        timeout=2.0,
                        check=False,
                    )
                    if result.returncode == 0:
                        # nvidia-smi succeeded, GPU likely available
                        return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            
            ##Step purpose: Check for Metal (macOS) support
            ##Condition purpose: macOS with Apple Silicon likely has Metal
            if platform.system() == "Darwin":
                # Check for Apple Silicon (M1/M2/M3)
                try:
                    import subprocess
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=1.0,
                        check=False,
                    )
                    if result.returncode == 0 and "Apple" in result.stdout:
                        # Apple Silicon detected, Metal likely available
                        return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    # Assume Metal may be available on macOS
                    return True
            
            ##Step purpose: Default to False if no detection method works
            # Conservative approach - let llama-cpp-python handle actual GPU check
            # If GPU isn't available, llama-cpp-python will fall back to CPU
            return False
            
        except Exception as e:
            ##Step purpose: Log error but don't fail
            logger.debug("gpu_availability_check_failed", error=str(e))
            # Conservative: assume no GPU if check fails
            return False
    
    ##Method purpose: Check if model is loaded
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._llm is not None
    
    ##Method purpose: Generate completion from prompt
    def generate(
        self,
        prompt: Prompt,
        params: GenerationParams | None = None,
    ) -> Completion:
        """
        Generate completion from prompt.
        
        Args:
            prompt: Structured prompt
            params: Generation parameters (uses defaults if None)
            
        Returns:
            Completion result
            
        Raises:
            LLMGenerationError: If generation fails
        """
        ##Condition purpose: Ensure model is loaded
        if self._llm is None:
            raise LLMGenerationError("Model not loaded")
        
        ##Step purpose: Use default params if not provided
        if params is None:
            params = GenerationParams(
                max_tokens=self._model_config.max_tokens,
                temperature=self._model_config.temperature,
                top_p=self._model_config.top_p,
            )
        
        ##Step purpose: Format prompt
        prompt_text = prompt.format_chat()
        
        ##Error purpose: Catch generation errors
        try:
            ##Action purpose: Generate completion
            start_time = time.perf_counter()
            
            result = self._llm(
                prompt_text,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repeat_penalty=params.repeat_penalty,
                stop=params.stop,
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            ##Step purpose: Extract result data
            choice = result["choices"][0]
            content = choice["text"]
            finish_reason = choice.get("finish_reason", "stop")
            
            return Completion(
                content=content,
                finish_reason=finish_reason,
                tokens_generated=result["usage"]["completion_tokens"],
                tokens_prompt=result["usage"]["prompt_tokens"],
                generation_time_ms=elapsed_ms,
            )
        except Exception as e:
            raise LLMGenerationError(str(e), prompt_text[:200]) from e
    
    ##Method purpose: Generate simple text completion
    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        params: GenerationParams | None = None,
    ) -> str:
        """
        Simple text generation helper.
        
        Args:
            text: User message
            system_prompt: System prompt
            params: Generation parameters
            
        Returns:
            Generated text
        """
        prompt = Prompt(
            system=system_prompt,
            user_message=text,
        )
        completion = self.generate(prompt, params)
        return completion.content
    
    ##Method purpose: Factory method for production use
    @classmethod
    def create(
        cls,
        model_config: ModelConfig,
        hardware_config: HardwareConfig,
        auto_load: bool = True,
    ) -> "LLMInterface":
        """
        Factory method to create and optionally load LLM.
        
        Args:
            model_config: Model configuration
            hardware_config: Hardware configuration
            auto_load: Whether to load model immediately
            
        Returns:
            LLMInterface instance
        """
        interface = cls(model_config, hardware_config)
        
        ##Condition purpose: Load model if requested
        if auto_load:
            interface.load()
        
        return interface
