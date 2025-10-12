"""
Cognitive Process Accelerator (CPA)
Intelligent software-focused optimization layer for speed and responsiveness.
"""

import os
import threading
import time
from typing import Optional


class CognitiveProcessAccelerator:
    """
    Cognitive Process Accelerator (CPA) - Intelligent software optimization.
    
    Features:
    1. Proactive Caching: Pre-loads model metadata and initial layers into RAM cache
    2. JIT Compilation: Compiles hot functions using numba for performance
    """
    
    def __init__(self, model_path: str, ui_logger):
        """
        Initialize the CPA.
        
        Args:
            model_path: Path to the GGUF model file
            ui_logger: Logger for user feedback
        """
        self.model_path = model_path
        self.ui_logger = ui_logger
        self.cache_thread: Optional[threading.Thread] = None
        self.cache_ready = False
        self.model_metadata = {}
        
    def start(self):
        """Start the CPA in a low-priority background thread."""
        if not os.path.exists(self.model_path):
            self.ui_logger.system_message("CPA: Model path not found, skipping pre-warming.")
            return
            
        # Start proactive caching in background thread
        self.cache_thread = threading.Thread(
            target=self._proactive_cache_worker,
            daemon=True,
            name="CPA-Cache-Worker"
        )
        self.cache_thread.start()
        
        # JIT compilation happens on-demand during first use
        self._prepare_jit_compilation()
    
    def _proactive_cache_worker(self):
        """
        Background worker that pre-warms the model cache.
        Runs in a low-priority thread to avoid interfering with main operations.
        """
        try:
            self.ui_logger.system_message("CPA: Pre-warming model cache in background...")
            
            # Read model file metadata (first few KB)
            # This brings the file into OS page cache
            with open(self.model_path, 'rb') as f:
                # Read first 1MB to get model metadata
                header_data = f.read(1024 * 1024)
                self.model_metadata['header_size'] = len(header_data)
                
            self.cache_ready = True
            self.ui_logger.system_message("CPA: Cache pre-warming complete. First prompt will be faster.")
            
        except Exception as e:
            self.ui_logger.system_message(f"CPA: Cache pre-warming failed: {e}")
    
    def _prepare_jit_compilation(self):
        """
        Prepare JIT compilation environment.
        
        Note: Actual JIT compilation happens on first function call.
        This method ensures numba is available and configured.
        """
        try:
            import numba
            # Configure numba for optimal performance
            numba.config.NUMBA_DEFAULT_NUM_THREADS = 1  # Use single thread for JIT functions
            numba.config.NUMBA_NUM_THREADS = 1
            self.ui_logger.system_message("CPA: JIT compilation ready (numba available).")
        except ImportError:
            self.ui_logger.system_message("CPA: JIT compilation unavailable (numba not installed).")
    
    def is_cache_ready(self) -> bool:
        """Check if the proactive cache is ready."""
        return self.cache_ready
    
    def wait_for_cache(self, timeout: float = 5.0) -> bool:
        """
        Wait for cache to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if cache is ready, False if timeout
        """
        start_time = time.time()
        while not self.cache_ready and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        return self.cache_ready


# JIT-compiled helper functions for hot paths
def _setup_jit_helpers():
    """
    Define JIT-compiled helper functions for performance-critical operations.
    These functions are compiled to native code on first use.
    """
    try:
        from numba import jit
        import numpy as np
        
        @jit(nopython=True, cache=True)
        def fast_token_count(text_length: int, avg_chars_per_token: float = 4.0) -> int:
            """Fast approximation of token count from text length."""
            return int(text_length / avg_chars_per_token)
        
        @jit(nopython=True, cache=True)
        def fast_normalize_scores(scores: np.ndarray) -> np.ndarray:
            """Fast normalization of similarity scores."""
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score > 0:
                return (scores - min_score) / (max_score - min_score)
            return scores
        
        return {
            'fast_token_count': fast_token_count,
            'fast_normalize_scores': fast_normalize_scores
        }
    except ImportError:
        # Fallback implementations without JIT
        def fast_token_count(text_length: int, avg_chars_per_token: float = 4.0) -> int:
            return int(text_length / avg_chars_per_token)
        
        def fast_normalize_scores(scores):
            min_score = min(scores)
            max_score = max(scores)
            if max_score - min_score > 0:
                return [(s - min_score) / (max_score - min_score) for s in scores]
            return scores
        
        return {
            'fast_token_count': fast_token_count,
            'fast_normalize_scores': fast_normalize_scores
        }


# Initialize JIT helpers on module load
jit_helpers = _setup_jit_helpers()
