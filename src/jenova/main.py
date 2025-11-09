# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is the main entry point for the JENOVA Cognitive Architecture."""

import os
from pathlib import Path
import yaml

# CRITICAL GPU Memory Management - Step 1: Conditional PyTorch CUDA Access
# This MUST be set BEFORE importing PyTorch (via any module)
# Load config to check hardware.pytorch_gpu_enabled setting

# Quick config load to determine PyTorch GPU access
_config_path = Path(__file__).parent / "config" / "main_config.yaml"
try:
    with open(_config_path, "r", encoding="utf-8") as f:
        _quick_config = yaml.safe_load(f)
        _pytorch_gpu = _quick_config.get("hardware", {}).get(
            "pytorch_gpu_enabled", False
        )
except (FileNotFoundError, yaml.YAMLError, KeyError, ValueError) as e:
    # If config can't be loaded, default to safe mode (PyTorch on CPU)
    # This handles: missing config file, invalid YAML, missing keys, invalid values
    import sys
    print(f"Warning: Could not load config ({type(e).__name__}), using CPU-only mode for PyTorch", file=sys.stderr)
    _pytorch_gpu = False

if not _pytorch_gpu:
    # Default behavior: Hide CUDA from PyTorch to preserve VRAM for main LLM
    # Strategy:
    #   - PyTorch (embeddings): CPU only, cannot see GPU
    #   - llama-cpp-python (main LLM): Full GPU access via direct CUDA bindings
    # All NVIDIA VRAM is available for the main LLM via llama-cpp-python
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA from PyTorch
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:32"
else:
    # Allow PyTorch to see GPU for accelerated embeddings (6GB+ VRAM recommended)
    # Trade-off: Shares VRAM with main LLM, but provides 5-10x faster embeddings
    # PyTorch will allocate ~500MB-1GB VRAM, reducing available space for LLM layers
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

# Fix tokenizers parallelism warning when spawning subprocesses (e.g., web_search)
# This must be set before importing any libraries that use tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import getpass
import queue
import traceback

from jenova.assumptions.manager import AssumptionManager
from jenova.cognitive_engine.engine import CognitiveEngine
from jenova.cognitive_engine.memory_search import MemorySearch
from jenova.cognitive_engine.rag_system import RAGSystem
from jenova.config import load_configuration
from jenova.cortex.cortex import Cortex
from jenova.infrastructure import (
    ErrorHandler,
    ErrorSeverity,
    HealthMonitor,
    MetricsCollector,
    FileManager,
    DataValidator,
)
from jenova.insights.manager import InsightManager
from jenova.llm import (
    ModelManager,
    ModelLoadError,
    EmbeddingManager,
    EmbeddingLoadError,
    LLMInterface,
)
from jenova.memory.episodic import EpisodicMemory
from jenova.memory.procedural import ProceduralMemory
from jenova.memory.semantic import SemanticMemory
from jenova.tools import ToolHandler
from jenova.ui.logger import UILogger
from jenova.ui.terminal import TerminalUI
from jenova.utils.file_logger import FileLogger
from jenova.utils.telemetry_fix import apply_telemetry_patch

# Phase 8: Distributed computing imports
from jenova.network import (
    JenovaDiscoveryService,
    PeerManager,
    JenovaRPCClient,
    SecurityManager,
)
from jenova.network.rpc_service import JenovaRPCServicer, JenovaRPCServer
from jenova.network.metrics import NetworkMetricsCollector
from jenova.llm.distributed_llm_interface import DistributedLLMInterface
from jenova.memory.distributed_memory_search import DistributedMemorySearch

# Phase 10: User profiling and personalization
from jenova.user.profile import UserProfileManager

# Phase 12: Contextual learning
from jenova.learning.contextual_engine import ContextualLearningEngine

# Phase 19: Backup and restore capabilities
from jenova.memory.backup_manager import BackupManager

# Phases 13-17: Enhanced CLI capabilities
from jenova.analysis import (
    ContextOptimizer,
    CodeMetrics,
    SecurityScanner,
    IntentClassifier,
    CommandDisambiguator,
)
from jenova.code_tools import (
    FileEditor,
    CodeParser,
    RefactoringEngine,
    SyntaxHighlighter,
    CodebaseMapper,
    InteractiveTerminal,
)
from jenova.git_tools import (
    GitInterface,
    CommitAssistant,
    DiffAnalyzer,
    HooksManager,
    BranchManager,
)
from jenova.orchestration import (
    TaskPlanner,
    SubagentManager,
    ExecutionEngine,
    CheckpointManager,
    BackgroundTaskManager,
)
from jenova.automation import (
    CustomCommandManager,
    HooksSystem,
    TemplateEngine,
    WorkflowLibrary,
)

apply_telemetry_patch()


def main():
    """Main entry point for The JENOVA Cognitive Architecture."""
    username = getpass.getuser()
    user_data_root = Path.home() / ".jenova-ai" / "users" / username
    user_data_root.mkdir(parents=True, exist_ok=True)

    # Setup logging first
    message_queue = queue.Queue()
    ui_logger = UILogger(message_queue=message_queue)
    file_logger = FileLogger(user_data_root=user_data_root)

    llm_interface = None
    model_manager = None
    embedding_manager = None
    error_handler = None
    health_monitor = None
    metrics = None

    # Phase 8: Network layer components
    security_manager = None
    peer_manager = None
    rpc_client = None
    rpc_server = None
    discovery_service = None
    network_metrics = None
    distributed_llm = None
    distributed_memory = None

    # Phase 10: User profiling
    user_profile_manager = None

    # Phase 12: Learning engine
    learning_engine = None

    # Phases 13-17: Enhanced CLI components
    context_optimizer = None
    code_metrics = None
    security_scanner = None
    intent_classifier = None
    command_disambiguator = None
    file_editor = None
    code_parser = None
    refactoring_engine = None
    syntax_highlighter = None
    codebase_mapper = None
    interactive_terminal = None
    git_interface = None
    commit_assistant = None
    diff_analyzer = None
    hooks_manager = None
    branch_manager = None
    task_planner = None
    subagent_manager = None
    execution_engine = None
    checkpoint_manager = None
    background_task_manager = None
    custom_command_manager = None
    hooks_system = None
    template_engine = None
    workflow_library = None

    try:
        # --- Configuration ---
        ui_logger.progress_message("Loading configuration", 10)
        try:
            config = load_configuration(ui_logger, file_logger)
            config["user_data_root"] = user_data_root
            ui_logger.success("Configuration loaded")
        except Exception as e:
            ui_logger.error(f"Configuration error: {e}")
            ui_logger.error("Fix your configuration and try again.")
            return 1

        # --- Initialize Infrastructure (Phase 2) ---
        ui_logger.progress_message("Initializing infrastructure", 20)
        error_handler = ErrorHandler(ui_logger, file_logger)
        health_monitor = HealthMonitor(ui_logger, file_logger)
        metrics = MetricsCollector(ui_logger, file_logger)
        file_manager = FileManager(ui_logger, file_logger)
        ui_logger.success("Infrastructure initialized")

        # --- Initialize User Profiling (Phase 10) ---
        user_profile_manager = UserProfileManager(config, file_logger)
        user_profile = user_profile_manager.get_profile(username)
        file_logger.log_info(f"User profile loaded for: {username}")

        # --- Initialize Learning Engine (Phase 12) ---
        learning_engine = ContextualLearningEngine(config, file_logger, user_data_root)
        file_logger.log_info("Contextual learning engine initialized")

        # Check system health at startup (Phase 6: Enhanced display)
        ui_logger.progress_message("Checking system health", 30)
        with metrics.measure("startup_health_check"):
            health = health_monitor.get_health_snapshot()
            health_data = {
                "status": health.status.value,
                "cpu_percent": health.cpu_percent,
                "memory_percent": health.memory_percent,
                "memory_available_gb": health.memory_available_gb,
                "gpu_available": health.gpu_memory_total_mb is not None,
            }
            if health.gpu_memory_total_mb is not None:
                gpu_memory_percent = (
                    health.gpu_memory_used_mb / health.gpu_memory_total_mb
                ) * 100
                health_data["gpu"] = {
                    "memory_percent": gpu_memory_percent,
                    "memory_used_mb": health.gpu_memory_used_mb,
                }

            # Display health status using Phase 6 methods
            ui_logger.health_status(health_data)

            if health.status.value != "healthy":
                for warning in health.warnings:
                    ui_logger.warning(f"  {warning}")

            file_logger.log_info(
                f"Startup health: CPU {health.cpu_percent:.1f}%, "
                f"Memory {health.memory_percent:.1f}% "
                f"({health.memory_available_gb:.1f}GB free)"
            )

        insights_root = user_data_root / "insights"
        cortex_root = user_data_root / "cortex"

        ui_logger.info(">> Initializing Intelligence Matrix...")

        # --- Language Models (Phase 3: New LLM Layer) ---
        # CRITICAL: Load LLM FIRST to claim GPU memory before PyTorch initializes CUDA
        ui_logger.progress_message("Loading AI model", 40)
        try:
            with metrics.measure(
                "model_load_llm", {"gpu_layers": config["model"]["gpu_layers"]}
            ):
                # Use new ModelManager (Phase 3)
                model_manager = ModelManager(config, file_logger, ui_logger)
                llm = model_manager.load_model()

                if not llm:
                    raise RuntimeError(
                        "LLM model could not be loaded. Check model path and integrity."
                    )

                # Use new LLMInterface (Phase 3)
                llm_interface = LLMInterface(config, ui_logger, file_logger, llm)

            # Log model load stats (Phase 6: Enhanced startup info)
            stats = metrics.get_stats("model_load_llm")
            model_info = model_manager.get_model_info()
            model_details = (
                f"{model_info.get('model_name', 'unknown')} | "
                f"ctx={model_info.get('context_size', 0)} | "
                f"gpu_layers={model_info.get('gpu_layers', 0)}"
            )
            ui_logger.startup_info("LLM Model", stats.avg_time, model_details)
            file_logger.log_info(
                f"LLM loaded in {stats.avg_time:.2f}s - "
                f"{model_info.get('model_name', 'unknown')} "
                f"(ctx={model_info.get('context_size', 0)}, "
                f"gpu_layers={model_info.get('gpu_layers', 0)})"
            )
        except (ModelLoadError, Exception) as e:
            error_handler.handle_error(e, "Model Loading", ErrorSeverity.CRITICAL)
            recommendation = error_handler.get_cuda_recommendation()
            if recommendation:
                ui_logger.system_message(recommendation)
            return 1

        # Load embedding model AFTER main LLM
        # Embedding model runs on CPU to preserve all GPU VRAM for main LLM
        ui_logger.progress_message("Loading embedding model", 55)
        try:
            with metrics.measure("model_load_embedding"):
                # CRITICAL: Embedding model MUST use CPU only
                # PyTorch was initialized with CUDA_VISIBLE_DEVICES='' so it cannot access GPU
                # This preserves all 4096 MB of GPU VRAM for the main LLM

                # Use new EmbeddingManager (Phase 3)
                embedding_manager = EmbeddingManager(config, file_logger, ui_logger)
                embedding_manager.load_model()
                embedding_model = embedding_manager.embedding_model

                if not embedding_model:
                    raise RuntimeError("Embedding model could not be loaded.")

            stats = metrics.get_stats("model_load_embedding")
            embed_info = embedding_manager.get_model_info()
            embed_details = (
                f"{embed_info.get('model_name', 'unknown')} | "
                f"device={embed_info.get('device', 'CPU').upper()} | "
                f"dim={embed_info.get('dimension', 0)}"
            )
            ui_logger.startup_info("Embedding Model", stats.avg_time, embed_details)
            file_logger.log_info(
                f"Embedding model loaded: {embed_info.get('model_name', 'unknown')} "
                f"on device: {embed_info.get('device', 'CPU')} in {stats.avg_time:.2f}s "
                f"(dim={embed_info.get('dimension', 0)}) - GPU VRAM fully reserved for LLM"
            )
        except (EmbeddingLoadError, Exception) as e:
            traceback.print_exc()
            error_handler.handle_error(
                e, "Embedding Model Loading", ErrorSeverity.CRITICAL
            )
            return 1

        # --- Memory Systems ---
        ui_logger.progress_message("Initializing memory systems", 70)
        try:
            with metrics.measure("memory_systems_init"):
                episodic_mem_path = user_data_root / "memory" / "episodic"
                procedural_mem_path = user_data_root / "memory" / "procedural"
                semantic_mem_path = user_data_root / "memory" / "semantic"

                try:
                    semantic_memory = SemanticMemory(
                        config,
                        ui_logger,
                        file_logger,
                        semantic_mem_path,
                        llm_interface,
                        embedding_model,
                    )
                except Exception as e:
                    traceback.print_exc()
                    raise
                episodic_memory = EpisodicMemory(
                    config,
                    ui_logger,
                    file_logger,
                    episodic_mem_path,
                    llm_interface,
                    embedding_model,
                )
                procedural_memory = ProceduralMemory(
                    config,
                    ui_logger,
                    file_logger,
                    procedural_mem_path,
                    llm_interface,
                    embedding_model,
                )

            stats = metrics.get_stats("memory_systems_init")
            ui_logger.startup_info(
                "Memory Systems", stats.avg_time, "Semantic, Episodic, Procedural"
            )
        except Exception as e:
            error_handler.handle_error(
                e, "Memory System Initialization", ErrorSeverity.CRITICAL
            )
            return 1

        # --- Cognitive Core ---
        ui_logger.progress_message("Initializing cognitive core", 85)
        try:
            with metrics.measure("cognitive_core_init"):
                cortex = Cortex(
                    config, ui_logger, file_logger, llm_interface, cortex_root
                )
                memory_search = MemorySearch(
                    semantic_memory,
                    episodic_memory,
                    procedural_memory,
                    config,
                    file_logger,
                )
                insight_manager = InsightManager(
                    config,
                    ui_logger,
                    file_logger,
                    insights_root,
                    llm_interface,
                    cortex,
                    memory_search,
                )
                memory_search.insight_manager = (
                    insight_manager  # Circular dependency resolution
                )
                assumption_manager = AssumptionManager(
                    config,
                    ui_logger,
                    file_logger,
                    user_data_root,
                    cortex,
                    llm_interface,
                )
                tool_handler = ToolHandler(config, ui_logger, file_logger)
                rag_system = RAGSystem(
                    llm_interface, memory_search, insight_manager, config
                )

            stats = metrics.get_stats("cognitive_core_init")
            core_details = "Cortex, RAG (Phase 5), Insights, Tools"
            ui_logger.startup_info("Cognitive Core", stats.avg_time, core_details)
        except Exception as e:
            error_handler.handle_error(
                e, "Cognitive Core Initialization", ErrorSeverity.CRITICAL
            )
            return 1

        # --- Phase 19: Backup Manager ---
        ui_logger.progress_message("Initializing backup manager", 86)
        try:
            with metrics.measure("backup_manager_init"):
                backup_manager = BackupManager(
                    user_data_root=user_data_root,
                    backup_dir=None,  # Uses default: {user_data_root}/backups
                    file_logger=file_logger,
                )
            stats = metrics.get_stats("backup_manager_init")
            ui_logger.startup_info("Backup Manager", stats.avg_time, "Export/Import/Backup")
        except Exception as e:
            error_handler.handle_error(
                e, "Backup Manager Initialization", ErrorSeverity.MEDIUM
            )
            ui_logger.warning(
                "Backup features unavailable - continuing without backup capabilities"
            )
            file_logger.log_warning(f"Backup manager initialization failed: {e}")
            backup_manager = None

        # --- Phases 13-17: Enhanced CLI Capabilities ---
        ui_logger.progress_message("Initializing enhanced CLI capabilities", 87)
        try:
            with metrics.measure("cli_enhancements_init"):
                # Analysis Module
                context_optimizer = ContextOptimizer(config, file_logger)
                code_metrics = CodeMetrics(config, file_logger)
                security_scanner = SecurityScanner(config, file_logger)
                intent_classifier = IntentClassifier(config, file_logger)
                command_disambiguator = CommandDisambiguator(config, file_logger)

                # Code Tools Module
                file_editor = FileEditor(config, file_logger)
                code_parser = CodeParser(config, file_logger)
                refactoring_engine = RefactoringEngine(config, file_logger)
                syntax_highlighter = SyntaxHighlighter(config, file_logger)
                codebase_mapper = CodebaseMapper(config, file_logger)
                interactive_terminal = InteractiveTerminal(config, file_logger)

                # Git Tools Module
                git_interface = GitInterface(config, file_logger)
                commit_assistant = CommitAssistant(config, file_logger, llm_interface)
                diff_analyzer = DiffAnalyzer(config, file_logger, llm_interface)
                hooks_manager = HooksManager(config, file_logger)
                branch_manager = BranchManager(config, file_logger)

                # Orchestration Module
                task_planner = TaskPlanner(config, file_logger, llm_interface)
                subagent_manager = SubagentManager(config, file_logger)
                execution_engine = ExecutionEngine(config, file_logger, task_planner)
                checkpoint_manager = CheckpointManager(
                    config, file_logger, user_data_root
                )
                background_task_manager = BackgroundTaskManager(config, file_logger)

                # Automation Module
                custom_commands_dir = user_data_root / "custom_commands"
                custom_commands_dir.mkdir(parents=True, exist_ok=True)
                template_engine = TemplateEngine(config, file_logger)
                custom_command_manager = CustomCommandManager(
                    config, file_logger, template_engine, custom_commands_dir
                )
                hooks_system = HooksSystem(config, file_logger)
                workflow_library = WorkflowLibrary(config, file_logger)

            stats = metrics.get_stats("cli_enhancements_init")
            cli_details = (
                "Analysis (5 tools), Code Tools (6 tools), "
                "Git (5 tools), Orchestration (5 systems), Automation (4 systems)"
            )
            ui_logger.startup_info("CLI Enhancements", stats.avg_time, cli_details)
            ui_logger.success("Phase 13-17 capabilities activated")
            file_logger.log_info(
                "Enhanced CLI capabilities initialized: "
                "Analysis, Code Tools, Git Integration, Task Orchestration, Automation"
            )
        except Exception as e:
            # CLI enhancements are non-critical - core system can operate without them
            error_handler.handle_error(
                e, "CLI Enhancements Initialization", ErrorSeverity.MEDIUM
            )
            ui_logger.warning(
                "Enhanced CLI features unavailable - continuing with core features"
            )
            file_logger.log_warning(f"CLI enhancements initialization failed: {e}")
            # Set all CLI components to None on failure
            context_optimizer = None
            code_metrics = None
            security_scanner = None
            intent_classifier = None
            command_disambiguator = None
            file_editor = None
            code_parser = None
            refactoring_engine = None
            syntax_highlighter = None
            codebase_mapper = None
            interactive_terminal = None
            git_interface = None
            commit_assistant = None
            diff_analyzer = None
            hooks_manager = None
            branch_manager = None
            task_planner = None
            subagent_manager = None
            execution_engine = None
            checkpoint_manager = None
            background_task_manager = None
            custom_command_manager = None
            hooks_system = None
            template_engine = None
            workflow_library = None

        # --- Phase 8: Network Layer (Distributed Computing) ---
        network_enabled = config.get("network", {}).get("enabled", False)

        if network_enabled:
            # SECURITY: Validate that distributed mode requires SSL/TLS and JWT authentication
            security_config = config.get("network", {}).get("security", {})
            ssl_enabled = security_config.get("enabled", False)
            jwt_enabled = security_config.get("require_auth", False)

            if not ssl_enabled or not jwt_enabled:
                ui_logger.error("╔═══════════════════════════════════════════════════════════════╗")
                ui_logger.error("║ SECURITY ERROR: Insecure Distributed Mode Configuration      ║")
                ui_logger.error("╠═══════════════════════════════════════════════════════════════╣")
                ui_logger.error("║ Distributed mode requires both SSL/TLS and JWT authentication║")
                ui_logger.error("║ to protect network communications.                            ║")
                ui_logger.error("║                                                               ║")
                ui_logger.error("║ Current settings:                                             ║")
                ui_logger.error(f"║   network.security.enabled = {ssl_enabled:<31} ║")
                ui_logger.error(f"║   network.security.require_auth = {jwt_enabled:<28} ║")
                ui_logger.error("║                                                               ║")
                ui_logger.error("║ Fix: Edit src/jenova/config/main_config.yaml:                 ║")
                ui_logger.error("║   network:                                                    ║")
                ui_logger.error("║     security:                                                 ║")
                ui_logger.error("║       enabled: true      # Enable SSL/TLS encryption          ║")
                ui_logger.error("║       require_auth: true # Enable JWT authentication          ║")
                ui_logger.error("╚═══════════════════════════════════════════════════════════════╝")
                file_logger.log_critical(
                    f"Distributed mode security validation failed: "
                    f"ssl_enabled={ssl_enabled}, jwt_enabled={jwt_enabled}"
                )
                sys.exit(1)

            ui_logger.progress_message("Initializing distributed networking", 90)
            try:
                with metrics.measure("network_layer_init"):
                    # Initialize security manager first
                    security_config = config.get("network", {}).get("security", {})
                    cert_dir = security_config.get("cert_dir", "~/.jenova-ai/certs")

                    security_manager = SecurityManager(
                        config=config, file_logger=file_logger, cert_dir=cert_dir
                    )

                    # Generate/verify certificates
                    instance_name = config.get("instance_name", f"jenova-{username}")
                    security_manager.ensure_certificates(instance_name)

                    # Initialize network metrics
                    network_metrics = NetworkMetricsCollector(
                        file_logger=file_logger, history_size=1000
                    )

                    # Initialize peer manager
                    peer_manager = PeerManager(config=config, file_logger=file_logger)

                    # Initialize RPC client
                    rpc_client = JenovaRPCClient(
                        config=config,
                        file_logger=file_logger,
                        peer_manager=peer_manager,
                        security_manager=security_manager,
                    )

                    # Create authentication token for outgoing requests
                    instance_id = config.get("instance_id", username)
                    auth_token = security_manager.create_auth_token(
                        instance_id=instance_id,
                        instance_name=instance_name,
                        validity_seconds=3600,
                    )
                    rpc_client.set_auth_token(auth_token)

                    # Initialize RPC servicer (exposes local resources)
                    rpc_servicer = JenovaRPCServicer(
                        config=config,
                        file_logger=file_logger,
                        llm_interface=llm_interface,
                        embedding_manager=embedding_manager,
                        memory_search=memory_search,
                        health_monitor=health_monitor,
                    )

                    # Initialize RPC server
                    discovery_config = config.get("network", {}).get("discovery", {})
                    rpc_port = discovery_config.get("port", 50051)

                    rpc_server = JenovaRPCServer(
                        config=config,
                        file_logger=file_logger,
                        servicer=rpc_servicer,
                        security_manager=security_manager,
                        port=rpc_port,
                    )

                    # Start RPC server
                    rpc_server.start()

                    # Initialize discovery service
                    discovery_service = JenovaDiscoveryService(
                        config=config,
                        file_logger=file_logger,
                        peer_manager=peer_manager,
                        rpc_port=rpc_port,
                    )

                    # Start discovery (mDNS advertising + browsing)
                    discovery_service.start_advertising()
                    discovery_service.start_browsing()

                    # Initialize distributed LLM interface
                    distributed_llm = DistributedLLMInterface(
                        config=config,
                        file_logger=file_logger,
                        ui_logger=ui_logger,
                        local_llm_interface=llm_interface,
                        rpc_client=rpc_client,
                        peer_manager=peer_manager,
                        network_metrics=network_metrics,
                    )

                    # Initialize distributed memory search
                    distributed_memory = DistributedMemorySearch(
                        config=config,
                        file_logger=file_logger,
                        local_memory_search=memory_search,
                        rpc_client=rpc_client,
                        peer_manager=peer_manager,
                        network_metrics=network_metrics,
                    )

                stats = metrics.get_stats("network_layer_init")
                network_details = (
                    f"Discovery ({discovery_config.get('service_name', 'jenova-ai')}), "
                    f"RPC (port {rpc_port}), "
                    f"Security ({'enabled' if security_config.get('enabled', True) else 'disabled'})"
                )
                ui_logger.startup_info("Network Layer", stats.avg_time, network_details)
                ui_logger.success("Distributed computing enabled")
                file_logger.log_info(
                    f"Network layer initialized: {instance_name} on port {rpc_port}"
                )

            except Exception as e:
                # Network layer failures are non-critical - system can operate without it
                error_handler.handle_error(
                    e, "Network Layer Initialization", ErrorSeverity.HIGH
                )
                ui_logger.warning(
                    "Distributed features unavailable - continuing in local-only mode"
                )
                file_logger.log_warning(f"Network layer initialization failed: {e}")
                # Set all network components to None
                security_manager = None
                peer_manager = None
                rpc_client = None
                rpc_server = None
                discovery_service = None
                network_metrics = None
                distributed_llm = None
                distributed_memory = None
        else:
            file_logger.log_info(
                "Network layer disabled in configuration (network.enabled=false)"
            )

        # --- Final Engine and UI Setup ---
        ui_logger.progress_message("Finalizing cognitive engine", 95)

        # Log total startup time (Phase 6: Enhanced startup summary)
        startup_stats = {
            "llm_load": (
                metrics.get_stats("model_load_llm").avg_time
                if metrics.get_stats("model_load_llm")
                else 0
            ),
            "embedding_load": (
                metrics.get_stats("model_load_embedding").avg_time
                if metrics.get_stats("model_load_embedding")
                else 0
            ),
            "memory_init": (
                metrics.get_stats("memory_systems_init").avg_time
                if metrics.get_stats("memory_systems_init")
                else 0
            ),
            "cognitive_init": (
                metrics.get_stats("cognitive_core_init").avg_time
                if metrics.get_stats("cognitive_core_init")
                else 0
            ),
        }
        total_startup = sum(startup_stats.values())

        cognitive_engine = CognitiveEngine(
            llm_interface,
            memory_search,
            insight_manager,
            assumption_manager,
            config,
            ui_logger,
            file_logger,
            cortex,
            rag_system,
            tool_handler,
        )

        # Store infrastructure in cognitive engine for use during operation (Phase 5)
        if hasattr(cognitive_engine, "set_infrastructure"):
            cognitive_engine.set_infrastructure(
                health_monitor=health_monitor,
                metrics=metrics,
                error_handler=error_handler,
            )

        # Phase 8: Pass network layer to cognitive engine
        if hasattr(cognitive_engine, "set_network_layer"):
            cognitive_engine.set_network_layer(
                distributed_llm=distributed_llm,
                distributed_memory=distributed_memory,
                peer_manager=peer_manager,
                network_metrics=network_metrics,
            )

        # Phase 10: Pass user profile to cognitive engine
        if hasattr(cognitive_engine, "set_user_profile"):
            cognitive_engine.set_user_profile(
                user_profile_manager=user_profile_manager, user_profile=user_profile
            )

        # Phase 12: Pass learning engine to cognitive engine
        if hasattr(cognitive_engine, "set_learning_engine"):
            cognitive_engine.set_learning_engine(learning_engine)

        # Phases 13-17: Pass CLI enhancements to cognitive engine
        if hasattr(cognitive_engine, "set_cli_enhancements"):
            cognitive_engine.set_cli_enhancements(
                # Analysis
                context_optimizer=context_optimizer,
                code_metrics=code_metrics,
                security_scanner=security_scanner,
                intent_classifier=intent_classifier,
                command_disambiguator=command_disambiguator,
                # Code Tools
                file_editor=file_editor,
                code_parser=code_parser,
                refactoring_engine=refactoring_engine,
                syntax_highlighter=syntax_highlighter,
                codebase_mapper=codebase_mapper,
                interactive_terminal=interactive_terminal,
                # Git Tools
                git_interface=git_interface,
                commit_assistant=commit_assistant,
                diff_analyzer=diff_analyzer,
                hooks_manager=hooks_manager,
                branch_manager=branch_manager,
                # Orchestration
                task_planner=task_planner,
                subagent_manager=subagent_manager,
                execution_engine=execution_engine,
                checkpoint_manager=checkpoint_manager,
                background_task_manager=background_task_manager,
                # Automation
                custom_command_manager=custom_command_manager,
                hooks_system=hooks_system,
                template_engine=template_engine,
                workflow_library=workflow_library,
            )

        # Display startup summary (Phase 6)
        ui_logger.progress_message("Ready", 100)
        ui_logger.info(">> Cognitive Engine: Online.")
        ui_logger.success(f"Startup complete in {total_startup:.2f}s")
        file_logger.log_info(f"Startup complete in {total_startup:.2f}s")
        file_logger.log_info(f"Startup breakdown: {startup_stats}")

        # Phase 6: Display RAG cache stats at startup
        if hasattr(rag_system, "get_cache_stats"):
            cache_stats = rag_system.get_cache_stats()
            file_logger.log_info(f"RAG cache initialized: {cache_stats}")

        # Phase 6: Pass health_monitor and metrics to TerminalUI
        # Phases 13-17: Also pass CLI enhancement modules
        # Phase 19: Pass backup_manager
        ui = TerminalUI(
            cognitive_engine,
            ui_logger,
            health_monitor=health_monitor,
            metrics=metrics,
            # Phase 19: Backup capabilities
            backup_manager=backup_manager,
            # Phase 13-17 CLI enhancements
            context_optimizer=context_optimizer,
            code_metrics=code_metrics,
            security_scanner=security_scanner,
            intent_classifier=intent_classifier,
            command_disambiguator=command_disambiguator,
            file_editor=file_editor,
            code_parser=code_parser,
            refactoring_engine=refactoring_engine,
            syntax_highlighter=syntax_highlighter,
            codebase_mapper=codebase_mapper,
            interactive_terminal=interactive_terminal,
            git_interface=git_interface,
            commit_assistant=commit_assistant,
            diff_analyzer=diff_analyzer,
            hooks_manager=hooks_manager,
            branch_manager=branch_manager,
            task_planner=task_planner,
            subagent_manager=subagent_manager,
            execution_engine=execution_engine,
            checkpoint_manager=checkpoint_manager,
            background_task_manager=background_task_manager,
            custom_command_manager=custom_command_manager,
            hooks_system=hooks_system,
            template_engine=template_engine,
            workflow_library=workflow_library,
        )
        ui.run()

        # Log final metrics summary on clean shutdown
        if metrics:
            file_logger.log_info("=== Session Metrics Summary ===")
            metrics.log_summary(top_n=10)

    except KeyboardInterrupt:
        ui_logger.info("Shutdown requested by user")
        return 0
    except (RuntimeError, Exception) as e:
        error_message = f"Critical failure during startup: {e}"
        ui_logger.error(error_message)
        if file_logger:
            file_logger.log_error(error_message)
            file_logger.log_error(f"Traceback: {traceback.format_exc()}")
        return 1
    finally:
        # Ensure all resources are released (Phase 3: New cleanup)
        try:
            if llm_interface:
                llm_interface.close()
            if model_manager:
                model_manager.unload_model()
            if embedding_manager:
                embedding_manager.unload_model()

            # Phase 8: Network layer cleanup
            if discovery_service:
                file_logger.log_info("Stopping discovery service...")
                discovery_service.stop_advertising()
                discovery_service.stop_browsing()

            if rpc_server:
                file_logger.log_info("Stopping RPC server...")
                rpc_server.stop(grace_period=5)

            if rpc_client:
                file_logger.log_info("Closing RPC client connections...")
                rpc_client.close_all_connections()

            if security_manager:
                file_logger.log_info("Cleaning up security resources...")
                security_manager.close()

            if network_metrics:
                file_logger.log_info("Logging network metrics summary...")
                network_metrics.log_summary(top_n=10)

        except Exception as e:
            if file_logger:
                file_logger.log_error(f"Error during resource cleanup: {e}")

        shutdown_message = "JENOVA shutdown complete."
        if ui_logger:
            ui_logger.info(shutdown_message)
        if file_logger:
            file_logger.log_info(shutdown_message)


if __name__ == "__main__":
    import sys

    result = main()
    sys.exit(result if result is not None else 0)
