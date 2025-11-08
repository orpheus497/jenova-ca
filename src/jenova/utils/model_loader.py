# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Simplified model loader with clear, single-strategy approach.

Key principles:
1. One clear strategy: CPU-only OR GPU with specific layers
2. No CUDA_VISIBLE_DEVICES manipulation
3. No monkey-patching
4. Clear error messages
5. Fast failure (no complex retries)
"""

import gc
import multiprocessing
import os
from pathlib import Path

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


def find_model_path(config):
    """
    Finds the GGUF model file path.
    Searches in the configured path, then system-wide, then local directories.

    Args:
        config: Configuration dictionary

    Returns:
        Path to model file or None if not found
    """
    model_path = config["model"]["model_path"]
    if os.path.exists(model_path):
        return model_path

    # Search in system-wide directory
    system_model_dir = "/usr/local/share/models"
    if os.path.isdir(system_model_dir):
        for file in os.listdir(system_model_dir):
            if file.endswith(".gguf"):
                return os.path.join(system_model_dir, file)

    # Search in local directory
    local_model_dir = "models"
    if os.path.isdir(local_model_dir):
        for file in os.listdir(local_model_dir):
            if file.endswith(".gguf"):
                return os.path.join(local_model_dir, file)

    return None


def get_optimal_thread_count(configured_threads):
    """
    Determines optimal thread count based on configuration.

    Args:
        configured_threads: Value from config (-1 = auto, 0 = all, N = specific)

    Returns:
        Optimal thread count for the system
    """
    if configured_threads == -1:
        # Auto-detect: use physical cores (excluding hyperthreads)
        try:
            physical_cores = multiprocessing.cpu_count() // 2
            return max(1, physical_cores)
        except Exception:
            return 4  # Safe fallback
    elif configured_threads == 0:
        # Use all available threads
        return multiprocessing.cpu_count()
    else:
        # Use configured value
        return max(1, configured_threads)


def load_llm_model(config, file_logger=None, ui_logger=None):
    """
    Simplified model loader with single, clear strategy.

    Args:
        config: Configuration dictionary
        file_logger: Optional file logger for detailed logging
        ui_logger: Optional UI logger for user-facing messages

    Returns:
        Loaded Llama model instance

    Raises:
        RuntimeError: If model loading fails with detailed error message
    """
    # Find model path
    model_path = find_model_path(config)
    if not model_path:
        raise RuntimeError(
            "No GGUF model found!\n\n"
            "SOLUTIONS:\n"
            "1. Place a GGUF model in /usr/local/share/models/\n"
            "2. Place a GGUF model in ./models/\n"
            "3. Update model_path in main_config.yaml"
        )

    if file_logger:
        file_logger.log_info(f"Found model: {model_path}")

    # Get settings from config
    gpu_layers_config = config["model"]["gpu_layers"]

    # Handle 'auto' gpu_layers - detect optimal value based on available VRAM
    if isinstance(gpu_layers_config, str) and gpu_layers_config.lower() == "auto":
        from jenova.utils.hardware_detector import recommend_gpu_layers

        gpu_layers = recommend_gpu_layers()
        if ui_logger:
            ui_logger.info(f"Auto-detected GPU layers: {gpu_layers}")
        if file_logger:
            file_logger.log_info(
                f"GPU layers auto-detection: {gpu_layers} layers recommended"
            )
    else:
        gpu_layers = int(gpu_layers_config)

    threads = get_optimal_thread_count(config["model"]["threads"])
    context_size = config["model"]["context_size"]
    n_batch = config["model"]["n_batch"]
    mlock = config["model"].get("mlock", False)

    if file_logger:
        file_logger.log_info(
            f"Loading with: gpu_layers={gpu_layers}, threads={threads}, "
            f"context={context_size}, batch={n_batch}"
        )

    # Clear any existing GPU memory before loading
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # Not critical if torch isn't available

    # Single loading attempt - no complex fallbacks
    try:
        llm = Llama(
            model_path=model_path,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
            n_ctx=context_size,
            n_batch=n_batch,
            use_mlock=mlock,
            verbose=False,
        )

        if file_logger:
            file_logger.log_info("Model loaded successfully")

        return llm

    except Exception as e:
        error_msg = str(e)

        # Provide helpful error message based on error type
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            raise RuntimeError(
                f"Model loading failed: {error_msg}\n\n"
                "SOLUTIONS:\n"
                "1. Set gpu_layers: 0 in main_config.yaml (CPU-only mode)\n"
                "2. Reduce context_size to 2048\n"
                "3. Reduce n_batch to 128\n"
                "4. Set prefer_device: 'cpu'\n"
                "5. Check GPU memory: nvidia-smi\n"
                "6. Close other GPU applications"
            )
        else:
            raise RuntimeError(
                f"Model loading failed: {error_msg}\n\n"
                "Check:\n"
                "1. Model file is valid GGUF format\n"
                "2. Model file is not corrupted\n"
                "3. Sufficient system RAM available\n"
                "4. Check logs for detailed error information"
            )


def load_embedding_model(config, file_logger=None):
    """
    Load the embedding model for semantic operations.

    Args:
        config: Configuration dictionary
        file_logger: Optional file logger

    Returns:
        SentenceTransformer model instance

    Raises:
        RuntimeError: If embedding model loading fails
    """
    model_name = config["model"].get("embedding_model", "all-MiniLM-L6-v2")

    if file_logger:
        file_logger.log_info(f"Loading embedding model: {model_name}")

    try:
        # Load with CPU by default for stability
        embedding_model = SentenceTransformer(model_name, device="cpu")

        if file_logger:
            file_logger.log_info("Embedding model loaded successfully")

        return embedding_model

    except Exception as e:
        raise RuntimeError(
            f"Embedding model loading failed: {e}\n\n"
            "SOLUTIONS:\n"
            "1. Check internet connection (first-time download)\n"
            "2. Verify model name in config\n"
            "3. Check ~/.cache/torch/sentence_transformers/ directory"
        )


# For backwards compatibility, keep the main functions
def load_models(config, ui_logger=None, file_logger=None):
    """
    Load both LLM and embedding models.

    Args:
        config: Configuration dictionary
        ui_logger: Optional UI logger
        file_logger: Optional file logger

    Returns:
        tuple: (llm, embedding_model)
    """
    if ui_logger:
        ui_logger.info("Loading AI models...")

    llm = load_llm_model(config, file_logger, ui_logger)
    embedding_model = load_embedding_model(config, file_logger)

    if ui_logger:
        ui_logger.info("✓ Models loaded successfully")

    return llm, embedding_model
