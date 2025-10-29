# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for loading the models for the JENOVA Cognitive Architecture.
"""

import gc
import multiprocessing
import os
import sys
import warnings
from contextlib import contextmanager

import torch
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from jenova.ui.logger import UILogger
from jenova.utils.hardware_detector import HardwareDetector, print_system_info


# Monkey-patch llama-cpp-python to fix the AttributeError in __del__
try:
    from llama_cpp import _internals

    # Save the original close method
    _original_llama_model_close = _internals.LlamaModel.close

    def _patched_llama_model_close(self):
        """Patched close method that handles missing sampler attribute."""
        try:
            # Check if sampler exists before trying to access it
            if hasattr(self, 'sampler') and self.sampler is not None:
                return _original_llama_model_close(self)
            # If sampler doesn't exist, do nothing (model was never fully initialized)
        except AttributeError:
            # Silently handle the error if it occurs anyway
            pass

    # Apply the patch
    _internals.LlamaModel.close = _patched_llama_model_close
except (ImportError, AttributeError):
    # If we can't patch, continue anyway
    pass


@contextmanager
def suppress_llama_cleanup_errors():
    """
    Context manager to force garbage collection after failed model loads.
    This helps clean up any partially-constructed Llama objects.
    """
    try:
        yield
    finally:
        # Force garbage collection to clean up any failed Llama objects
        gc.collect()


def find_model_path(config):
    """
    Finds the GGUF model file path.
    Searches in the configured path, then system-wide, then local directories.
    """
    model_path = config['model']['model_path']
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


def get_gpu_memory_info():
    """
    Get GPU memory information using nvidia-smi.

    Returns:
        tuple: (total_mb, free_mb, used_mb, gpu_name) or (None, None, None, None) if unavailable
    """
    try:
        import subprocess
        # Query total, free, and used memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used,name',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 4:
                total_mb = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                used_mb = int(parts[2].strip())
                gpu_name = parts[3].strip()
                return total_mb, free_mb, used_mb, gpu_name
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None, None, None, None


def clear_cuda_cache():
    """
    Clear CUDA cache to free up GPU memory.
    This must be called before loading the LLM to ensure maximum VRAM availability.
    """
    import gc
    import time

    # Force Python garbage collection
    gc.collect()

    # Try to clear PyTorch CUDA cache if available (shouldn't be, but just in case)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass

    # Give CUDA driver time to release memory
    time.sleep(0.2)
    gc.collect()


def calculate_optimal_gpu_layers(total_vram_mb, free_vram_mb, model_path, context_size=8192):
    """
    Calculate optimal GPU layers based on available VRAM.

    For a typical 7B GGUF model (Q4_K_M quantization):
    - Each layer uses ~120-150 MB VRAM
    - KV cache varies with context size (larger context = more VRAM)
    - CUDA compute buffers scale with context and batch size

    Args:
        total_vram_mb: Total GPU VRAM in MB
        free_vram_mb: Free GPU VRAM in MB
        model_path: Path to model file (for size estimation)
        context_size: Target context size in tokens

    Returns:
        int: Recommended number of GPU layers
    """
    # CUDA overhead (minimal, ~100 MB)
    cuda_overhead = 100

    # CUDA compute buffer overhead (scales with context size)
    # For n_batch=512: roughly 170MB @ 1024ctx, 585MB @ 8192ctx
    # Approximate: base_compute + (context / 1024) * scale
    compute_buffer = 170 + (context_size / 1024 - 1) * (585 - 170) / 7
    # At 1024: 170 MB
    # At 8192: 170 + 7 * 59.3 = ~585 MB
    # At 4096: 170 + 3 * 59.3 = ~348 MB

    # Estimate KV cache size based on context
    # For 7B model (4096 embedding, 32 layers, 8 KV heads):
    # KV cache = n_ctx * n_layers * n_embd_head_k * n_head_kv * 2 (K+V) * 2 bytes (f16)
    # = 8192 * 32 * 128 * 8 * 2 * 2 / 1024^2 = ~512 MB for full model
    # But we're only offloading some layers, so scale proportionally
    kv_cache_per_layer = (context_size * 128 * 8 * 2 * 2) / (1024 * 1024)  # MB per layer

    # Calculate available VRAM for model layers
    available_mb = free_vram_mb - cuda_overhead - compute_buffer

    if available_mb <= 0:
        return 0  # Not enough VRAM, use CPU

    # Estimate bytes per layer for model weights
    # For Q4_K_M 7B models: roughly 120-135 MB per layer
    avg_mb_per_layer = 125

    # Calculate layers accounting for both weights and KV cache
    # Solve: layers * (avg_mb_per_layer + kv_cache_per_layer) = available_mb
    total_mb_per_layer = avg_mb_per_layer + kv_cache_per_layer
    max_layers = int(available_mb / total_mb_per_layer)

    # Cap at model's actual layer count (typically 32 for 7B models)
    max_layers = min(max_layers, 32)

    return max(0, max_layers)


def load_llm_model(config, file_logger=None):
    """
    Loads the Llama model using llama-cpp-python with dynamic resource allocation.
    Automatically detects optimal CPU threads and GPU layers with intelligent fallback.

    Now supports:
    - NVIDIA GPUs (CUDA)
    - Intel GPUs (Iris, UHD, Arc) via OpenCL/Vulkan
    - AMD GPUs and APUs via OpenCL/Vulkan
    - Apple Silicon via Metal
    - Multi-platform: Linux, macOS, Windows, Android
    - Intelligent memory management with swap optimization

    Args:
        config: Configuration dictionary
        file_logger: Optional FileLogger instance for detailed error logging
    """
    # CRITICAL: Keep PyTorch isolated from CUDA
    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:32'

    logger = UILogger()
    model_path = find_model_path(config)

    if not model_path:
        if logger:
            logger.error("No GGUF model found.")
            logger.error(
                "Please download a model and place it in /usr/local/share/models or ./models.")
        return None

    if logger:
        logger.system_message(f"Found model: {model_path}")
        logger.system_message("Detecting hardware...")

    # NEW: Comprehensive hardware detection
    detector = HardwareDetector()
    resources = detector.detect_all()

    # Show hardware info if verbose mode or if user wants details
    show_hardware_details = config.get('hardware', {}).get('show_details', False)
    if show_hardware_details and logger:
        logger.system_message("\n" + "="*60)
        logger.system_message("Hardware Detection Results:")
        logger.system_message("="*60)
        logger.system_message(f"Platform: {resources.platform.value}")
        logger.system_message(f"CPU: {resources.cpu_name} ({resources.cpu_cores_physical} cores)")
        logger.system_message(f"RAM: {resources.ram_available_mb}/{resources.ram_total_mb} MB")
        logger.system_message(f"Swap: {resources.swap_free_mb}/{resources.swap_total_mb} MB")

        if resources.compute_devices:
            logger.system_message(f"\nCompute devices ({len(resources.compute_devices)} found):")
            for i, device in enumerate(resources.compute_devices, 1):
                backends = []
                if device.supports_cuda:
                    backends.append("CUDA")
                if device.supports_opencl:
                    backends.append("OpenCL")
                if device.supports_vulkan:
                    backends.append("Vulkan")
                if device.supports_metal:
                    backends.append("Metal")

                mem_info = f" ({device.memory_free_mb}/{device.memory_mb} MB)" if device.memory_mb else ""
                logger.system_message(f"  {i}. {device.name}{mem_info}")
                logger.system_message(f"     Type: {device.device_type.value}, Backends: {', '.join(backends)}")
        else:
            logger.system_message("\nCompute devices: CPU-only")
        logger.system_message("="*60 + "\n")

    if logger:
        logger.system_message("Loading model with dynamic resource allocation...")

    # Get configuration values
    gpu_layers = config['model']['gpu_layers']
    configured_threads = config['model']['threads']
    n_batch = config['model']['n_batch']
    context_size = config['model']['context_size']
    mlock = config['model'].get('mlock', True)

    # NEW: Get hardware preferences
    hw_config = config.get('hardware', {})
    prefer_device = hw_config.get('prefer_device', 'auto')  # 'auto', 'cuda', 'opencl', 'vulkan', 'metal', 'cpu'
    device_index = hw_config.get('device_index', 0)  # Which GPU to use (0, 1, 2...)

    # Get optimal configuration from hardware detector
    model_size_mb = 4000  # Estimate for 7B Q4_K_M model
    optimal_config = detector.get_optimal_configuration(resources, model_size_mb, context_size)

    # Determine which device to use based on preferences
    selected_device = None
    selected_backend = optimal_config['backend']

    if prefer_device == 'auto':
        # Use the highest priority device
        if resources.compute_devices and len(resources.compute_devices) > device_index:
            selected_device = resources.compute_devices[device_index]
        # Use optimal config recommendations
        if gpu_layers == -1:
            gpu_layers = optimal_config['gpu_layers']
    elif prefer_device == 'cpu':
        gpu_layers = 0
        selected_backend = 'cpu'
    elif prefer_device in ['cuda', 'opencl', 'vulkan', 'metal']:
        # User specified a backend
        selected_backend = prefer_device
        # Find matching device
        for device in resources.compute_devices:
            if (prefer_device == 'cuda' and device.supports_cuda) or \
               (prefer_device == 'opencl' and device.supports_opencl) or \
               (prefer_device == 'vulkan' and device.supports_vulkan) or \
               (prefer_device == 'metal' and device.supports_metal):
                selected_device = device
                break

    # Apply optimal configuration recommendations
    if optimal_config['memory_strategy'] == 'swap_optimized':
        mlock = False
        if logger:
            logger.system_message("Memory strategy: Optimized for swap usage")
    elif optimal_config['memory_strategy'] == 'minimal':
        mlock = False
        context_size = min(context_size, optimal_config['recommended_context'])
        n_batch = optimal_config['n_batch']
        if logger:
            logger.system_message("Memory strategy: Minimal (tight memory conditions)")

    # Dynamic thread detection
    threads = get_optimal_thread_count(configured_threads) if configured_threads != optimal_config['threads'] else optimal_config['threads']
    if configured_threads == -1:
        if logger:
            logger.system_message(f"Auto-detected {threads} threads (physical cores)")
    elif configured_threads == 0:
        if logger:
            logger.system_message(f"Using all {threads} available threads")
    else:
        if logger:
            logger.system_message(f"Using configured {threads} threads")

    # Report selected device
    if selected_device and logger:
        mem_str = f" ({selected_device.memory_free_mb}/{selected_device.memory_mb} MB)" if selected_device.memory_mb else ""
        logger.system_message(f"Selected compute device: {selected_device.name}{mem_str}")
        logger.system_message(f"Backend: {selected_backend.upper()}")

        if selected_device.is_integrated:
            logger.system_message("Using integrated GPU (shares system RAM)")

    # CRITICAL: Clear CUDA cache before GPU operations
    clear_cuda_cache()

    # Handle CUDA device visibility
    if selected_backend == 'cuda' and selected_device:
        # Restore GPU visibility for llama-cpp-python
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_index)
        clear_cuda_cache()

    # Dynamic GPU layer allocation with fallback strategies
    loading_strategies = []

    # Report memory locking strategy
    if logger:
        logger.system_message(f"Model locking: {'enabled (RAM persistent)' if mlock else 'disabled (uses swap as needed)'}")

    if gpu_layers == -1 or gpu_layers > 0:
        # GPU offload requested or auto-detected
        if selected_device and gpu_layers > 0:
            device_type_name = selected_device.device_type.value.replace('_', ' ').title()

            if logger:
                logger.system_message(f"{device_type_name} offload: {gpu_layers} layers")

            # Start with requested/optimal layers
            loading_strategies.append((f"{device_type_name} ({gpu_layers} layers)", gpu_layers, mlock))

            # Create fallback chain by reducing layers
            fallback_steps = [gpu_layers - 4, gpu_layers - 8, gpu_layers // 2, gpu_layers // 4]
            for layers in fallback_steps:
                if layers > 0 and layers not in [s[1] for s in loading_strategies]:
                    loading_strategies.append((f"partial GPU ({layers} layers)", layers, mlock))

        elif gpu_layers > 0:
            # User requested GPU layers but no suitable device found
            if logger:
                logger.system_message(f"Warning: GPU layers requested but no suitable device found")
                logger.system_message(f"Falling back to CPU-only mode")

        # Always add CPU-only as final fallback
        loading_strategies.append(("CPU-only", 0, mlock))
    else:
        # CPU-only mode
        loading_strategies.append(("CPU-only", 0, mlock))

    # Try loading strategies in order
    # Use context manager to suppress llama-cpp-python cleanup errors during all attempts
    with suppress_llama_cleanup_errors():
        for idx, (strategy_name, layers, use_mlock) in enumerate(loading_strategies):
            try:
                if layers == -1:
                    if logger:
                        logger.system_message(f"Attempting {strategy_name}...")
                else:
                    if logger:
                        logger.system_message(f"Attempting {strategy_name}...")

                # Use configured context size
                # The optimal_layers calculation already accounts for KV cache requirements
                # So we can use the full configured context size
                if layers > 0:  # GPU offload enabled
                    # Use configured context - layers were already calculated to fit
                    effective_context = context_size

                    # Only reduce context on retry attempts if we're failing
                    if idx > 3:
                        effective_context = max(2048, context_size // 2)
                    if idx > 5:
                        effective_context = max(1024, context_size // 4)
                else:  # CPU-only
                    # CPU can handle full context as it uses RAM
                    effective_context = context_size

                llm = Llama(
                    model_path=model_path,
                    n_threads=threads,
                    n_gpu_layers=layers,
                    n_batch=n_batch,
                    n_ctx=effective_context,
                    use_mlock=use_mlock,
                    verbose=False,  # Disable verbose output for clean startup
                )

                # Success message
                context_info = f" (context: {effective_context} tokens)" if effective_context != context_size else ""
                if layers == -1:
                    if logger:
                        logger.system_message(
                            f"✓ Model loaded with {strategy_name} (dynamic VRAM management){context_info}")
                elif layers > 0:
                    if logger:
                        logger.system_message(
                            f"✓ Model loaded with {strategy_name}{context_info}")
                else:
                    if logger:
                        logger.system_message(
                            f"✓ Model loaded with {strategy_name}{context_info}")

                return llm

            except Exception as e:
                # Force cleanup of failed CUDA contexts to free VRAM
                if logger:
                    logger.system_message(f"  Cleaning up CUDA memory...")
                clear_cuda_cache()  # Use our comprehensive cache clearing function

                error_msg = str(e)
                if strategy_name == loading_strategies[-1][0]:
                    # Last strategy failed - give up
                    if logger:
                        logger.error(f"Error loading model: {error_msg}")
                        logger.error(
                            "All loading strategies failed. Please check:")
                        logger.error("1. Model file is not corrupted")
                        logger.error("2. Sufficient RAM/VRAM available")
                        logger.error("3. llama-cpp-python is properly installed")
                    if file_logger:
                        file_logger.log_error(f"All model loading strategies failed. Last error: {error_msg}")
                    return None
                else:
                    # Try next strategy
                    if logger:
                        # Show concise error for debugging
                        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                            logger.system_message(
                                f"  {strategy_name} failed (insufficient VRAM), trying next strategy...")
                        elif "cuda" in error_msg.lower():
                            logger.system_message(
                                f"  {strategy_name} failed (CUDA error: {error_msg[:50]}...), trying next strategy...")
                        else:
                            logger.system_message(
                                f"  {strategy_name} failed ({error_msg[:40]}...), trying next strategy...")
                    if file_logger:
                        file_logger.log_error(f"Strategy '{strategy_name}' failed: {error_msg}")


def load_embedding_model(model_name: str, device: str = 'cpu'):
    """
    Loads the SentenceTransformer model on CPU only to preserve GPU VRAM for main LLM.

    Args:
        model_name: Name of the model to load
        device: Device to use (default: 'cpu' - GPU is reserved for main LLM)

    Returns:
        SentenceTransformer model instance or None on failure
    """
    try:
        # CRITICAL: Always use CPU for embeddings to preserve all GPU VRAM for the main LLM
        # Never check torch.cuda.is_available() as it initializes PyTorch CUDA context
        # and allocates GPU memory (~561 MB) that should be available for llama-cpp-python

        # Force CPU device - do not auto-detect or use CUDA
        device = 'cpu'

        # Load the model with explicit CPU device
        model = SentenceTransformer(model_name, device=device)

        return model

    except Exception as e:
        import traceback
        print(f"Error loading embedding model '{model_name}': {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None
