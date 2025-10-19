import os
import traceback
import queue
import signal
import sys
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
from jenova import tools

from jenova.assumptions.manager import AssumptionManager

from jenova.cortex.cortex import Cortex

import getpass

def main():
    """Main entry point for The JENOVA Cognitive Architecture."""
    # Set up graceful shutdown handlers
    shutdown_requested = [False]  # Use list to allow modification in nested function
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        if not shutdown_requested[0]:
            shutdown_requested[0] = True
            print("\n🛑 Shutdown signal received. Exiting gracefully...")
            # Raise KeyboardInterrupt to be caught by main loop
            raise KeyboardInterrupt
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    username = getpass.getuser()
    user_data_root = os.path.join(os.path.expanduser("~"), ".jenova-ai", "users", username)
    os.makedirs(user_data_root, exist_ok=True)
    
    # Create message queue BEFORE UILogger to enable thread-safe UI updates
    message_queue = queue.Queue()
    ui_logger = UILogger(message_queue=message_queue)
    file_logger = FileLogger(user_data_root=user_data_root)

    try:
        config = load_configuration(ui_logger, file_logger)
        config['user_data_root'] = user_data_root
        insights_root = os.path.join(user_data_root, "insights")
        cortex_root = os.path.join(user_data_root, "cortex")
        
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
        # Pass the llm_interface to the tools that need it.
        tools.llm_interface = llm_interface
        ui.run()
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        ui_logger.system_message("\n🛑 Keyboard interrupt received. Shutting down gracefully...")
    except Exception as e:
        error_message = f"A critical failure occurred: {e}"
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        file_logger.log_error(traceback.format_exc())
    finally:
        # Ensure LLM resources are released
        try:
            if 'llm_interface' in locals() and llm_interface and hasattr(llm_interface, 'model') and llm_interface.model:
                llm_interface.close()
        except Exception as e:
            # Don't crash during cleanup
            if 'file_logger' in locals():
                file_logger.log_error(f"Error during LLM cleanup: {e}")

        shutdown_message = "JENOVA shutting down."
        if 'ui_logger' in locals():
            ui_logger.info(shutdown_message)
        if 'file_logger' in locals():
            file_logger.log_info(shutdown_message)
