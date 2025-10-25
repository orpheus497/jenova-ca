import os
import multiprocessing
from llama_cpp import Llama
from jenova.ui.logger import UILogger
from sentence_transformers import SentenceTransformer
import torch

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
        except:
            return 4  # Safe fallback
    elif configured_threads == 0:
        # Use all available threads
        return multiprocessing.cpu_count()
    else:
        # Use configured value
        return max(1, configured_threads)

def load_llm_model(config):
    """
    Loads the Llama model using llama-cpp-python with dynamic resource allocation.
    Automatically detects optimal CPU threads and GPU layers with intelligent fallback.
    """
    logger = UILogger()
    model_path = find_model_path(config)

    if not model_path:
        logger.error("No GGUF model found.")
        logger.error("Please download a model and place it in /usr/local/share/models or ./models.")
        return None

    logger.system_message(f"Found model: {model_path}")
    logger.system_message("Loading model with dynamic resource allocation...")

    gpu_layers = config['model']['gpu_layers']
    configured_threads = config['model']['threads']
    n_batch = config['model']['n_batch']
    context_size = config['model']['context_size']
    mlock = config['model'].get('mlock', True)
    
    # Dynamic thread detection
    threads = get_optimal_thread_count(configured_threads)
    if configured_threads == -1:
        logger.system_message(f"Auto-detected {threads} threads (physical cores)")
    elif configured_threads == 0:
        logger.system_message(f"Using all {threads} available threads")
    else:
        logger.system_message(f"Using configured {threads} threads")
    
    # Dynamic GPU layer allocation with fallback strategies
    loading_strategies = []
    
    if gpu_layers == -1 and torch.cuda.is_available():
        # Dynamic: try full GPU offload with intelligent fallback
        logger.system_message("GPU detected: Attempting dynamic VRAM allocation")
        loading_strategies.append(("full GPU (all layers)", -1, mlock))
        loading_strategies.append(("partial GPU (20 layers)", 20, False))
        loading_strategies.append(("partial GPU (15 layers)", 15, False))
        loading_strategies.append(("partial GPU (10 layers)", 10, False))
        loading_strategies.append(("CPU-only", 0, False))
    elif gpu_layers > 0 and torch.cuda.is_available():
        # Manual GPU layer specification
        loading_strategies.append((f"partial GPU ({gpu_layers} layers)", gpu_layers, False))
        # Add fallback options
        if gpu_layers > 15:
            loading_strategies.append(("partial GPU (15 layers)", 15, False))
        if gpu_layers > 10:
            loading_strategies.append(("partial GPU (10 layers)", 10, False))
        loading_strategies.append(("CPU-only", 0, False))
    else:
        # CPU-only mode
        loading_strategies.append(("CPU-only", 0, mlock))
    
    # Try loading strategies in order
    for strategy_name, layers, use_mlock in loading_strategies:
        try:
            if layers == -1:
                logger.system_message(f"Attempting {strategy_name}...")
            else:
                logger.system_message(f"Attempting {strategy_name}...")
                
            llm = Llama(
                model_path=model_path,
                n_threads=threads,
                n_gpu_layers=layers,
                n_batch=n_batch,
                n_ctx=context_size,
                use_mlock=use_mlock,
                verbose=False,
            )
            
            # Success message
            if layers == -1:
                logger.system_message(f"✓ Model loaded with {strategy_name} (dynamic VRAM management)")
            elif layers > 0:
                logger.system_message(f"✓ Model loaded with {strategy_name}")
            else:
                logger.system_message(f"✓ Model loaded with {strategy_name}")
            
            return llm
            
        except Exception as e:
            if strategy_name == loading_strategies[-1][0]:
                # Last strategy failed - give up
                import traceback
                logger.error(f"Error loading model: {e}")
                logger.error("All loading strategies failed. Please check:")
                logger.error("1. Model file is not corrupted")
                logger.error("2. Sufficient RAM/VRAM available")
                logger.error("3. llama-cpp-python is properly installed")
                return None
            else:
                # Try next strategy
                logger.system_message(f"  {strategy_name} failed, trying next strategy...")

def load_embedding_model(model_name: str, device: str = None):
    """
    Loads the SentenceTransformer model with automatic device detection and GPU support.
    
    Args:
        model_name: Name of the model to load
        device: Device to use ('cpu', 'cuda', or None for auto-detection)
        
    Returns:
        SentenceTransformer model instance or None on failure
    """
    try:
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the model with the specified device
        model = SentenceTransformer(model_name, device=device)
        
        return model
        
    except Exception as e:
        import traceback
        print(f"Error loading embedding model '{model_name}': {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None