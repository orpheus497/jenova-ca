import os
import traceback
import psutil
import atexit
from jenova.utils.telemetry_fix import apply_telemetry_patch
apply_telemetry_patch()

from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.cognitive_engine.rag_system import RAGSystem

from jenova.ui.terminal import TerminalUI
from jenova.llm_interface import LLMInterface
from jenova.cognitive_engine.memory_search import MemorySearch
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.semantic import SemanticMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.insights.manager import InsightManager
from jenova.config import load_configuration
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger
from jenova.utils.model_loader import load_embedding_model
from jenova.utils.optimization_engine import OptimizationEngine
from jenova import tools

from jenova.assumptions.manager import AssumptionManager

from jenova.cortex.cortex import Cortex

import getpass

# Global variables to store original process settings for graceful shutdown
_original_affinity = None
_original_priority = None

def _restore_process_settings():
    """Graceful shutdown: Reset CPU affinity and process priority to original values."""
    global _original_affinity, _original_priority
    try:
        process = psutil.Process(os.getpid())
        if _original_affinity is not None:
            process.cpu_affinity(_original_affinity)
        if _original_priority is not None:
            process.nice(_original_priority)
    except Exception:
        pass  # Silent failure on shutdown


def main():
    """Main entry point for the perfected Jenova Cognitive Architecture."""
    global _original_affinity, _original_priority
    
    # HYPER-THREADED SYNERGISTIC ENGINE: Set CPU affinity and process priority
    try:
        process = psutil.Process(os.getpid())
        
        # Store original settings for graceful shutdown
        try:
            _original_affinity = process.cpu_affinity()
        except Exception:
            _original_affinity = None
        
        try:
            _original_priority = process.nice()
        except Exception:
            _original_priority = None
        
        # Reserve Core 0 for main application thread (dedicated communication channel)
        # This creates a protected "express lane" for UI, data pipelines, and non-AI functions
        cpu_count = psutil.cpu_count(logical=True)
        if cpu_count and cpu_count > 1:
            process.cpu_affinity([0])  # Reserve Core 0 for main thread initially
        
        # Elevate process priority to prevent deadlock
        process.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, 'HIGH_PRIORITY_CLASS') else -10)
        
        # Register graceful shutdown handler
        atexit.register(_restore_process_settings)
    except Exception:
        # Continue even if affinity/priority settings fail (requires elevated permissions on some systems)
        pass
    
    username = getpass.getuser()
    user_data_root = os.path.join(os.path.expanduser("~"), ".jenova-ai", "users", username)
    os.makedirs(user_data_root, exist_ok=True)
    
    ui_logger = UILogger()
    file_logger = FileLogger(user_data_root=user_data_root)

    try:
        config = load_configuration(ui_logger, file_logger)
        config['user_data_root'] = user_data_root
        insights_root = os.path.join(user_data_root, "insights")
        cortex_root = os.path.join(user_data_root, "cortex")
        
        # Run optimization engine if enabled
        optimization_enabled = config.get('optimization', {}).get('enabled', True)
        if optimization_enabled:
            ui_logger.info(">> Running Performance Optimization Engine...")
            optimizer = OptimizationEngine(user_data_root, ui_logger)
            optimizer.run()
            config = optimizer.apply_settings(config)
            
            # Display optimization report with enhanced feedback
            hardware = optimizer.settings.get('hardware', {})
            optimal = optimizer.settings.get('optimal_settings', {})
            
            cpu_vendor = hardware.get('cpu', {}).get('vendor', 'Unknown')
            cpu_cores = hardware.get('cpu', {}).get('physical_cores', 'Unknown')
            gpu_vendor = hardware.get('gpu', {}).get('vendor', 'None')
            gpu_vram = hardware.get('gpu', {}).get('vram_mb', 0)
            hardware_profile = hardware.get('hardware_profile', 'Unknown')
            strategy = optimal.get('strategy', 'Unknown')
            
            ui_logger.system_message(f"Detected Hardware Profile: {hardware_profile}")
            ui_logger.system_message(f"Hardware: {cpu_vendor} CPU ({cpu_cores} cores), {gpu_vendor} GPU ({gpu_vram} MB VRAM)")
            ui_logger.system_message(f"Applying {strategy} strategy for speed.")
            ui_logger.system_message(f"Optimal Settings: {optimal.get('n_threads', 'N/A')} threads, {optimal.get('n_gpu_layers', 'N/A')} GPU layers")
        
        # HYPER-THREADED SYNERGISTIC ENGINE: Reset CPU affinity before loading LLM
        # This allows LLM worker threads to use all available cores (not just Core 0)
        # while maintaining high process priority for the main thread
        try:
            process = psutil.Process(os.getpid())
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count and cpu_count > 1:
                # Reset affinity to all cores for LLM worker threads
                process.cpu_affinity(list(range(cpu_count)))
        except Exception:
            pass  # Continue even if affinity reset fails
        
        ui_logger.info(">> Initializing Intelligence Matrix...")
        llm_interface = LLMInterface(config, ui_logger, file_logger)
        
        # Load the embedding model
        embedding_model = load_embedding_model(config['model']['embedding_model'])
        if not embedding_model:
            ui_logger.system_message("Fatal: Could not load the embedding model. Please check the logs for details.")
            return

        # User-specific memory paths
        episodic_mem_path = os.path.join(user_data_root, 'memory', 'episodic')
        procedural_mem_path = os.path.join(user_data_root, 'memory', 'procedural')
        semantic_mem_path = os.path.join(user_data_root, 'memory', 'semantic')

        semantic_memory = SemanticMemory(config, ui_logger, file_logger, semantic_mem_path, llm_interface, embedding_model)
        episodic_memory = EpisodicMemory(config, ui_logger, file_logger, episodic_mem_path, llm_interface)
        procedural_memory = ProceduralMemory(config, ui_logger, file_logger, procedural_mem_path, llm_interface)

        cortex = Cortex(config, ui_logger, file_logger, llm_interface, cortex_root)
        memory_search = MemorySearch(semantic_memory, episodic_memory, procedural_memory, config, file_logger)
        insight_manager = InsightManager(config, ui_logger, file_logger, insights_root, llm_interface, cortex, memory_search)
        memory_search.insight_manager = insight_manager # Set insight_manager after initialization
        assumption_manager = AssumptionManager(config, ui_logger, file_logger, user_data_root, cortex, llm_interface)
        
        rag_system = RAGSystem(llm_interface, memory_search, insight_manager, config)

        ui_logger.info(">> Cognitive Engine: Online.")
        cognitive_engine = CognitiveEngine(llm_interface, memory_search, insight_manager, assumption_manager, config, ui_logger, file_logger, cortex, rag_system)
        
        ui = TerminalUI(cognitive_engine, ui_logger)
        # Pass the console lock to the cognitive engine and ui_logger
        cognitive_engine.console_lock = ui.console_lock
        ui_logger.console_lock = ui.console_lock
        # Pass the llm_interface to the tools that need it.
        tools.llm_interface = llm_interface
        ui.run()
        
    except Exception as e:
        error_message = f"A critical failure occurred: {e}"
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        file_logger.log_error(traceback.format_exc())
    finally:
        # Ensure LLM resources are released
        if 'llm_interface' in locals() and llm_interface.model:
            llm_interface.close()

        shutdown_message = "Jenova AI shutting down."
        ui_logger.info(shutdown_message)
        file_logger.log_info(shutdown_message)
