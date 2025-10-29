# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is the main entry point for the JENOVA Cognitive Architecture.
"""

import os

# CRITICAL GPU Memory Management - Step 1: Blind PyTorch to CUDA
# This MUST be set BEFORE importing PyTorch (via any module)
# This prevents PyTorch from initializing CUDA context and consuming ~561 MB GPU memory
# Strategy:
#   - PyTorch (embeddings): CPU only, cannot see GPU
#   - llama-cpp-python (main LLM): Full GPU access via direct CUDA bindings
# All NVIDIA VRAM must be available for the main LLM via llama-cpp-python
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA from PyTorch during initialization
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:32'

# Fix tokenizers parallelism warning when spawning subprocesses (e.g., web_search)
# This must be set before importing any libraries that use tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import getpass
import queue
import traceback

from jenova.assumptions.manager import AssumptionManager
from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.cognitive_engine.memory_search import MemorySearch
from jenova.cognitive_engine.rag_system import RAGSystem
from jenova.config import load_configuration
from jenova.cortex.cortex import Cortex
from jenova.insights.manager import InsightManager
from jenova.llm_interface import LLMInterface
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.memory.semantic import SemanticMemory
from jenova.tools import ToolHandler
from jenova.ui.logger import UILogger
from jenova.ui.terminal import TerminalUI
from jenova.utils.file_logger import FileLogger
from jenova.utils.model_loader import load_embedding_model, load_llm_model
from jenova.utils.telemetry_fix import apply_telemetry_patch

apply_telemetry_patch()


def main():
    """Main entry point for The JENOVA Cognitive Architecture."""
    username = getpass.getuser()
    user_data_root = os.path.join(os.path.expanduser(
        "~"), ".jenova-ai", "users", username)
    os.makedirs(user_data_root, exist_ok=True)

    # Setup logging first
    message_queue = queue.Queue()
    ui_logger = UILogger(message_queue=message_queue)
    file_logger = FileLogger(user_data_root=user_data_root)

    llm_interface = None
    try:
        # --- Configuration ---
        try:
            config = load_configuration(ui_logger, file_logger)
            config['user_data_root'] = user_data_root
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

        insights_root = os.path.join(user_data_root, "insights")
        cortex_root = os.path.join(user_data_root, "cortex")

        ui_logger.info(">> Initializing Intelligence Matrix...")

        # --- Language Models ---
        # CRITICAL: Load LLM FIRST to claim GPU memory before PyTorch initializes CUDA
        try:
            llm = load_llm_model(config, file_logger)
            if not llm:
                raise RuntimeError(
                    "LLM model could not be loaded. Check model path and integrity.")
            llm_interface = LLMInterface(config, ui_logger, file_logger, llm)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLMInterface: {e}")

        # Load embedding model AFTER main LLM
        # Embedding model runs on CPU to preserve all GPU VRAM for main LLM
        try:
            ui_logger.info("Loading embedding model...")
            # CRITICAL: Embedding model MUST use CPU only
            # PyTorch was initialized with CUDA_VISIBLE_DEVICES='' so it cannot access GPU
            # This preserves all 4096 MB of GPU VRAM for the main LLM
            embedding_model = load_embedding_model(
                config['model']['embedding_model'], device='cpu')
            if not embedding_model:
                raise RuntimeError("Embedding model could not be loaded.")
            ui_logger.system_message(
                "✓ Embedding model loaded on CPU (GPU VRAM: 100% available for main LLM)")
            file_logger.log_info("Embedding model loaded on device: CPU - GPU VRAM fully reserved for LLM")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")

        # --- Memory Systems ---
        try:
            episodic_mem_path = os.path.join(
                user_data_root, 'memory', 'episodic')
            procedural_mem_path = os.path.join(
                user_data_root, 'memory', 'procedural')
            semantic_mem_path = os.path.join(
                user_data_root, 'memory', 'semantic')

            semantic_memory = SemanticMemory(
                config, ui_logger, file_logger, semantic_mem_path, llm_interface, embedding_model)
            episodic_memory = EpisodicMemory(
                config, ui_logger, file_logger, episodic_mem_path, llm_interface, embedding_model)
            procedural_memory = ProceduralMemory(
                config, ui_logger, file_logger, procedural_mem_path, llm_interface, embedding_model)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize memory systems: {e}")

        # --- Cognitive Core ---
        try:
            cortex = Cortex(config, ui_logger, file_logger,
                            llm_interface, cortex_root)
            memory_search = MemorySearch(
                semantic_memory, episodic_memory, procedural_memory, config, file_logger)
            insight_manager = InsightManager(
                config, ui_logger, file_logger, insights_root, llm_interface, cortex, memory_search)
            memory_search.insight_manager = insight_manager  # Circular dependency resolution
            assumption_manager = AssumptionManager(
                config, ui_logger, file_logger, user_data_root, cortex, llm_interface)
            tool_handler = ToolHandler(config, ui_logger, file_logger)
            rag_system = RAGSystem(
                llm_interface, memory_search, insight_manager, config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize cognitive core components: {e}")

        # --- Final Engine and UI Setup ---
        ui_logger.info(">> Cognitive Engine: Online.")
        cognitive_engine = CognitiveEngine(llm_interface, memory_search, insight_manager,
                                           assumption_manager, config, ui_logger, file_logger, cortex, rag_system, tool_handler)

        ui = TerminalUI(cognitive_engine, ui_logger)
        ui.run()

    except (RuntimeError, Exception) as e:
        error_message = f"A critical failure occurred during startup: {e}"
        ui_logger.system_message(error_message)
        file_logger.log_error(error_message)
        file_logger.log_error(traceback.format_exc())
    finally:
        # Ensure LLM resources are released
        if llm_interface and llm_interface.llm:
            llm_interface.close()

        shutdown_message = "JENOVA shutting down."
        if ui_logger:
            ui_logger.info(shutdown_message)
        if file_logger:
            file_logger.log_info(shutdown_message)


if __name__ == "__main__":
    main()
