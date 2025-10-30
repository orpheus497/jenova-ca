# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Model lifecycle management for JENOVA.

This module handles model loading, unloading, and lifecycle management
with proper resource cleanup and error handling.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from llama_cpp import Llama

from jenova.llm.cuda_manager import CUDAManager
from jenova.infrastructure.timeout_manager import timeout, TimeoutError


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ModelManager:
    """
    Manages LLM model lifecycle with proper resource management.

    Features:
    - Safe model loading with validation
    - Proper resource cleanup
    - CUDA-aware configuration
    - Clear error messages
    """

    def __init__(self, config: Dict[str, Any], file_logger=None, ui_logger=None):
        self.config = config
        self.file_logger = file_logger
        self.ui_logger = ui_logger
        self.cuda_manager = CUDAManager(file_logger)
        self.llm: Optional[Llama] = None
        self._model_path: Optional[Path] = None

    def find_model_path(self) -> Path:
        """
        Find the GGUF model file.

        Returns:
            Path to the model file

        Raises:
            ModelLoadError: If no model found
        """
        # Check configured model path
        if 'model' in self.config and 'model_path' in self.config['model']:
            model_path = Path(self.config['model']['model_path'])
            if model_path.exists() and model_path.suffix == '.gguf':
                return model_path

        # Search common locations
        search_paths = [
            Path('/usr/local/share/models'),
            Path.home() / '.cache' / 'jenova' / 'models',
            Path('./models'),
            Path('../models'),
        ]

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            # Find any .gguf file
            gguf_files = list(search_dir.glob('*.gguf'))
            if gguf_files:
                # Prefer files with common patterns
                for pattern in ['q4_k_m', 'q4_0', 'q5_k_m', 'q8_0']:
                    for file in gguf_files:
                        if pattern.lower() in file.name.lower():
                            return file

                # Return first found
                return gguf_files[0]

        raise ModelLoadError(
            "No GGUF model file found. Please:\n"
            "1. Download a GGUF model\n"
            "2. Place it in /usr/local/share/models/ or ./models/\n"
            "3. Or set model_path in main_config.yaml"
        )

    def get_optimal_thread_count(self, configured_threads: int) -> int:
        """
        Calculate optimal thread count for CPU inference.

        Args:
            configured_threads: User-configured thread count (-1 for auto)

        Returns:
            Optimal thread count
        """
        if configured_threads > 0:
            return configured_threads

        # Auto-detect: use physical cores (not hyperthreads)
        try:
            import os
            import psutil

            # Get physical core count
            physical_cores = psutil.cpu_count(logical=False)

            if physical_cores is None:
                # Fallback to logical cores / 2
                logical_cores = os.cpu_count() or 4
                return max(logical_cores // 2, 1)

            return max(physical_cores, 1)

        except ImportError:
            # psutil not available, use simple heuristic
            import os
            logical_cores = os.cpu_count() or 4
            return max(logical_cores // 2, 1)

    def load_model(self) -> Llama:
        """
        Load the LLM model with proper configuration.

        Returns:
            Loaded Llama model

        Raises:
            ModelLoadError: If loading fails
        """
        if self.llm is not None:
            if self.file_logger:
                self.file_logger.log_warning("Model already loaded, returning existing instance")
            return self.llm

        try:
            # Find model file
            if self.ui_logger:
                self.ui_logger.info("Searching for model file...")

            model_path = self.find_model_path()
            self._model_path = model_path

            if self.ui_logger:
                self.ui_logger.info(f"Found model: {model_path.name}")

            # Get configuration
            model_config = self.config.get('model', {})
            gpu_layers = model_config.get('gpu_layers', 0)
            context_size = model_config.get('context_size', 4096)
            n_batch = model_config.get('n_batch', 256)
            threads = model_config.get('threads', -1)
            mlock = model_config.get('mlock', False)

            # Validate GPU configuration
            if gpu_layers > 0:
                valid, message = self.cuda_manager.validate_gpu_config(gpu_layers)
                if not valid:
                    if self.ui_logger:
                        self.ui_logger.warning(message)
                        self.ui_logger.info("Falling back to CPU-only mode")
                    gpu_layers = 0
                else:
                    if self.file_logger:
                        self.file_logger.log_info(message)

            # Calculate optimal threads
            optimal_threads = self.get_optimal_thread_count(threads)

            if self.ui_logger:
                self.ui_logger.info(f"Loading model with {gpu_layers} GPU layers, {optimal_threads} threads...")

            # Log CUDA info if using GPU
            if gpu_layers > 0 and self.file_logger:
                cuda_info = self.cuda_manager.get_info_summary()
                self.file_logger.log_info(f"GPU Configuration:\n{cuda_info}")

            # Load model with timeout protection (3 minutes max)
            try:
                with timeout(180, "Model loading timed out after 3 minutes"):
                    self.llm = Llama(
                        model_path=str(model_path),
                        n_threads=optimal_threads,
                        n_gpu_layers=gpu_layers,
                        n_ctx=context_size,
                        n_batch=n_batch,
                        use_mlock=mlock,
                        verbose=False
                    )
            except TimeoutError as e:
                raise ModelLoadError(
                    f"{e}\n\n"
                    "The model took too long to load. Try:\n"
                    "1. Reduce context_size in config (e.g., 2048)\n"
                    "2. Set gpu_layers: 0 (CPU-only)\n"
                    "3. Use a smaller model"
                )

            if self.ui_logger:
                self.ui_logger.success(f"Model loaded successfully ({model_path.name})")

            if self.file_logger:
                self.file_logger.log_info(
                    f"Model loaded: {model_path.name} "
                    f"(ctx={context_size}, batch={n_batch}, "
                    f"gpu_layers={gpu_layers}, threads={optimal_threads})"
                )

            return self.llm

        except TimeoutError:
            raise
        except ModelLoadError:
            raise
        except Exception as e:
            error_msg = str(e).lower()

            # Provide helpful error messages
            if 'out of memory' in error_msg or 'oom' in error_msg:
                raise ModelLoadError(
                    f"Model loading failed: Out of memory\n\n"
                    f"Your system ran out of memory loading the model. Try:\n"
                    f"1. Set gpu_layers: 0 (CPU-only mode)\n"
                    f"2. Reduce context_size to 2048\n"
                    f"3. Close other memory-intensive applications\n"
                    f"4. Use a smaller model\n\n"
                    f"Original error: {e}"
                )
            elif 'cuda' in error_msg:
                raise ModelLoadError(
                    f"Model loading failed: CUDA error\n\n"
                    f"There was a problem with GPU acceleration. Try:\n"
                    f"1. Set gpu_layers: 0 (CPU-only mode)\n"
                    f"2. Check GPU is not in use: nvidia-smi\n"
                    f"3. Update GPU drivers\n\n"
                    f"Original error: {e}"
                )
            elif 'file' in error_msg or 'path' in error_msg:
                raise ModelLoadError(
                    f"Model loading failed: File error\n\n"
                    f"Could not access the model file. Check:\n"
                    f"1. Model file exists and is readable\n"
                    f"2. Path is correct in main_config.yaml\n"
                    f"3. File is not corrupted\n\n"
                    f"Original error: {e}"
                )
            else:
                raise ModelLoadError(
                    f"Model loading failed: {e}\n\n"
                    f"Try setting gpu_layers: 0 in main_config.yaml for CPU-only mode"
                )

    def unload_model(self):
        """Unload the model and free resources."""
        if self.llm is not None:
            try:
                del self.llm
                self.llm = None

                if self.file_logger:
                    self.file_logger.log_info("Model unloaded successfully")

                # Force garbage collection to free memory
                import gc
                gc.collect()

            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Error unloading model: {e}")

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_path": str(self._model_path) if self._model_path else None,
            "model_name": self._model_path.name if self._model_path else None,
            "context_size": self.config.get('model', {}).get('context_size', 4096),
            "gpu_layers": self.config.get('model', {}).get('gpu_layers', 0),
            "cuda_available": self.cuda_manager.is_cuda_available()
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.unload_model()
