##Script function and purpose: Main entry point for The JENOVA Cognitive Architecture
##This script initializes all components of the cognitive architecture and starts the Bubble Tea TUI
##This is the SOLE entry point - BubbleTea is the only supported UI

import os
import traceback
from typing import Optional

##Block purpose: Import Pydantic compatibility fix FIRST, before any ChromaDB imports
##This must happen before chromadb is imported anywhere in the application
from jenova.utils.pydantic_compat import *  # noqa: F401, F403

from jenova.utils.telemetry_fix import apply_telemetry_patch

##Function purpose: Apply telemetry patch to disable ChromaDB telemetry
apply_telemetry_patch()

from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.cognitive_engine.rag_system import RAGSystem

from jenova.ui.bubbletea import BubbleTeaUI
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
from jenova.cognitive_engine.integration_layer import IntegrationLayer
from jenova.utils.cache import CacheManager

import getpass


##Function purpose: Initialize all JENOVA components and return configured instances
##This factory function sets up the entire cognitive architecture
def initialize_jenova(ui_logger: UILogger, file_logger: FileLogger, user_data_root: str):
    """
    Initialize all JENOVA cognitive architecture components.
    
    Args:
        ui_logger: UILogger instance for user-facing messages
        file_logger: FileLogger instance for persistent logging
        user_data_root: Path to user-specific data directory
        
    Returns:
        tuple: (cognitive_engine, llm_interface) or (None, None) on failure
    """
    ##Block purpose: Load configuration and set up paths
    config: dict = load_configuration(ui_logger, file_logger)
    config['user_data_root'] = user_data_root
    insights_root: str = os.path.join(user_data_root, "insights")
    cortex_root: str = os.path.join(user_data_root, "cortex")
    
    ##Block purpose: Initialize LLM interface
    ui_logger.info(">> Initializing Intelligence Matrix...")
    llm_interface: LLMInterface = LLMInterface(config, ui_logger, file_logger)
    
    ##Block purpose: Load embedding model for vector operations
    embedding_model = load_embedding_model(config['model']['embedding_model'])
    if not embedding_model:
        ui_logger.system_message("Fatal: Could not load the embedding model. Please check the logs for details.")
        ##Block purpose: Clean up LLM interface before returning on failure
        if llm_interface and llm_interface.model:
            llm_interface.close()
        return None, None

    ##Block purpose: Initialize memory systems with user-specific paths
    episodic_mem_path: str = os.path.join(user_data_root, 'memory', 'episodic')
    procedural_mem_path: str = os.path.join(user_data_root, 'memory', 'procedural')
    semantic_mem_path: str = os.path.join(user_data_root, 'memory', 'semantic')

    semantic_memory: SemanticMemory = SemanticMemory(config, ui_logger, file_logger, semantic_mem_path, llm_interface, embedding_model)
    episodic_memory: EpisodicMemory = EpisodicMemory(config, ui_logger, file_logger, episodic_mem_path, llm_interface)
    procedural_memory: ProceduralMemory = ProceduralMemory(config, ui_logger, file_logger, procedural_mem_path, llm_interface)

    ##Block purpose: Initialize cache manager for performance optimization
    cache_manager: Optional[CacheManager] = None
    performance_config = config.get('performance', {}).get('caching', {})
    if performance_config.get('enabled', True):
        cache_manager = CacheManager(config)
        file_logger.log_info("Cache manager initialized for performance optimization")
    
    ##Block purpose: Initialize Cortex and cognitive components
    cortex: Cortex = Cortex(config, ui_logger, file_logger, llm_interface, cortex_root)
    
    ##Block purpose: Initialize integration layer for Cortex-Memory coordination
    integration_layer: Optional[IntegrationLayer] = None
    integration_config = config.get('cortex', {}).get('integration', {})
    if integration_config.get('enabled', True):
        integration_layer = IntegrationLayer(cortex, None, config, file_logger, cache_manager)
    
    ##Block purpose: Initialize memory search with all memory systems
    memory_search: MemorySearch = MemorySearch(
        semantic_memory, episodic_memory, procedural_memory, 
        config, file_logger, cortex, integration_layer, 
        llm_interface, embedding_model, cache_manager
    )
    
    ##Block purpose: Initialize insight manager with memory search reference
    insight_manager: InsightManager = InsightManager(
        config, ui_logger, file_logger, insights_root, 
        llm_interface, cortex, memory_search, integration_layer
    )
    memory_search.insight_manager = insight_manager
    
    ##Block purpose: Update integration layer with memory_search reference
    if integration_layer:
        integration_layer.memory_search = memory_search
    
    ##Block purpose: Initialize assumption manager for hypothesis tracking
    assumption_manager: AssumptionManager = AssumptionManager(
        config, ui_logger, file_logger, user_data_root, 
        cortex, llm_interface, integration_layer
    )
    
    ##Block purpose: Initialize RAG system and cognitive engine
    rag_system: RAGSystem = RAGSystem(llm_interface, memory_search, insight_manager, config)

    ui_logger.info(">> Cognitive Engine: Online.")
    cognitive_engine: CognitiveEngine = CognitiveEngine(
        llm_interface, memory_search, insight_manager, assumption_manager, 
        config, ui_logger, file_logger, cortex, rag_system
    )
    
    ##Block purpose: Set integration layer reference in cognitive engine for feedback loops
    if integration_layer:
        cognitive_engine.integration_layer = integration_layer
    
    return cognitive_engine, llm_interface


##Function purpose: Main entry point that initializes all components and starts the Bubble Tea UI
def main() -> None:
    """Main entry point for The JENOVA Cognitive Architecture with Bubble Tea UI."""
    ##Block purpose: Initialize user-specific data directory
    username: str = getpass.getuser()
    user_data_root: str = os.path.join(os.path.expanduser("~"), ".jenova-ai", "users", username)
    os.makedirs(user_data_root, exist_ok=True)
    
    ##Block purpose: Initialize logging systems
    ui_logger: UILogger = UILogger()
    file_logger: FileLogger = FileLogger(user_data_root=user_data_root)
    
    llm_interface = None

    try:
        ##Block purpose: Initialize all JENOVA components using factory function
        cognitive_engine, llm_interface = initialize_jenova(ui_logger, file_logger, user_data_root)
        
        if cognitive_engine is None:
            return
        
        ##Block purpose: Initialize and start Bubble Tea TUI
        ui: BubbleTeaUI = BubbleTeaUI(cognitive_engine, ui_logger)
        tools.llm_interface = llm_interface
        ui.run()
        
    except Exception as e:
        ##Block purpose: Handle critical errors gracefully
        error_message: str = f"A critical failure occurred: {e}"
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        file_logger.log_error(traceback.format_exc())
    finally:
        ##Block purpose: Ensure LLM resources are released on shutdown
        if llm_interface and llm_interface.model:
            llm_interface.close()

        shutdown_message: str = "JENOVA shutting down."
        ui_logger.info(shutdown_message)
        file_logger.log_info(shutdown_message)


##Block purpose: Allow running as module or script
if __name__ == "__main__":
    main()
