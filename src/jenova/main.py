import os
import traceback
from jenova.utils.telemetry_fix import apply_telemetry_patch
apply_telemetry_patch()

from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.cognitive_engine.rag_system import RAGSystem
from jenova.cognitive_engine.document_processor import DocumentProcessor
from jenova.ui.terminal import TerminalUI
from jenova.llm_interface import LLMInterface
from jenova.cognitive_engine.memory_search import MemorySearch
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.semantic import SemanticMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.insights.manager import InsightManager
from jenova.config import load_configuration
from jenova.tools.file_tools import FileTools
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger

from jenova.assumptions.manager import AssumptionManager

from jenova.cortex.cortex import Cortex

from jenova.tools.system_tools import SystemTools

import getpass

def main():
    """Main entry point for the perfected Jenova Cognitive Architecture."""
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
        docs_path = os.path.join(os.getcwd(), "src", "jenova", "docs")
        
        ui_logger.info(">> Initializing Intelligence Matrix...")
        llm_interface = LLMInterface(config, ui_logger, file_logger)
        file_tools = FileTools(config, ui_logger, file_logger)
        system_tools = SystemTools(ui_logger, file_logger)
        
        # User-specific memory paths
        episodic_mem_path = os.path.join(user_data_root, 'memory', 'episodic')
        procedural_mem_path = os.path.join(user_data_root, 'memory', 'procedural')
        semantic_mem_path = os.path.join(user_data_root, 'memory', 'semantic')

        semantic_memory = SemanticMemory(config, ui_logger, file_logger, semantic_mem_path, llm_interface)
        episodic_memory = EpisodicMemory(config, ui_logger, file_logger, episodic_mem_path, llm_interface)
        procedural_memory = ProceduralMemory(config, ui_logger, file_logger, procedural_mem_path, llm_interface)

        cortex = Cortex(config, ui_logger, file_logger, llm_interface, cortex_root)
        insight_manager = InsightManager(config, ui_logger, file_logger, insights_root, llm_interface, cortex, None) # Memory search is set later
        assumption_manager = AssumptionManager(config, ui_logger, file_logger, user_data_root, cortex, llm_interface)
        memory_search = MemorySearch(semantic_memory, episodic_memory, procedural_memory, insight_manager)
        insight_manager.memory_search = memory_search
        
        rag_system = RAGSystem(llm_interface, memory_search, insight_manager)
        document_processor = DocumentProcessor(docs_path, semantic_memory, insight_manager, assumption_manager, llm_interface)

        ui_logger.info(">> Cognitive Engine: Online.")
        cognitive_engine = CognitiveEngine(llm_interface, memory_search, file_tools, insight_manager, assumption_manager, config, ui_logger, file_logger, cortex, system_tools, rag_system, document_processor)
        
        ui = TerminalUI(cognitive_engine, ui_logger)
        ui.run()
        
    except Exception as e:
        error_message = f"A critical failure occurred: {e}"
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        file_logger.log_error(traceback.format_exc())
    finally:
        shutdown_message = "Jenova AI shutting down."
        ui_logger.info(shutdown_message)
        file_logger.log_info(shutdown_message)