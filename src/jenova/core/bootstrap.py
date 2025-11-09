# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.
#
# Phase 21: Full Project Remediation & Modernization
# Application Bootstrap - Phased Initialization Logic

"""
Application bootstrap module for The JENOVA Cognitive Architecture.

This module handles the phased initialization of the application, replacing the
monolithic 793-line main() function with structured, testable initialization logic.

The bootstrap process follows these phases:
    1. Configuration Loading (10%)
    2. Infrastructure Initialization (20%)
    3. Health Checks (30%)
    4. Model Loading (40%)
    5. Embedding Loading (50%)
    6. Memory System Initialization (60%)
    7. Cognitive Engine Initialization (70%)
    8. Network Layer Initialization (80%)
    9. CLI Tools Initialization (90%)
    10. Finalization (100%)

Example:
    >>> from jenova.core import ApplicationBootstrapper
    >>> bootstrapper = ApplicationBootstrapper(username="alice")
    >>> container = bootstrapper.bootstrap()
    >>> app = container.resolve('application')
"""

import getpass
import os
import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from jenova.config import load_configuration
from jenova.config.constants import (
    PROGRESS_LOADING_CONFIG,
    PROGRESS_INIT_INFRASTRUCTURE,
    PROGRESS_CHECKING_HEALTH,
    PROGRESS_LOADING_MODEL,
    PROGRESS_INIT_MEMORY,
    PROGRESS_INIT_COGNITIVE,
    PROGRESS_INIT_NETWORK,
    PROGRESS_INIT_CLI_TOOLS,
    PROGRESS_COMPLETE,
)
from jenova.core.container import DependencyContainer
from jenova.infrastructure import (
    ErrorHandler,
    HealthMonitor,
    MetricsCollector,
    FileManager,
    DataValidator,
)
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger


class ApplicationBootstrapper:
    """
    Handles phased application initialization.

    This class orchestrates the complex initialization process, ensuring
    components are created and initialized in the correct order with proper
    error handling and progress reporting.

    Attributes:
        username: Current system user
        user_data_root: User-specific data directory
        container: Dependency injection container
        config: Loaded configuration
        message_queue: Queue for UI messages
        ui_logger: UI logger instance
        file_logger: File logger instance

    Example:
        >>> bootstrapper = ApplicationBootstrapper()
        >>> try:
        >>>     container = bootstrapper.bootstrap()
        >>>     app = container.resolve('application')
        >>>     app.run()
        >>> except Exception as e:
        >>>     print(f"Bootstrap failed: {e}")
    """

    def __init__(self, username: Optional[str] = None):
        """
        Initialize application bootstrapper.

        Args:
            username: Optional username (defaults to current system user)
        """
        self.username = username or getpass.getuser()
        self.user_data_root = Path.home() / ".jenova-ai" / "users" / self.username
        self.user_data_root.mkdir(parents=True, exist_ok=True)

        self.container = DependencyContainer()
        self.config: Dict[str, Any] = {}
        self.message_queue = queue.Queue()
        self.ui_logger: Optional[UILogger] = None
        self.file_logger: Optional[FileLogger] = None
        self.start_time: float = 0.0

    def bootstrap(self) -> DependencyContainer:
        """
        Execute full bootstrap process.

        Returns:
            Fully initialized dependency container

        Raises:
            RuntimeError: If any bootstrap phase fails
        """
        # Track startup time
        self.start_time = time.time()

        try:
            self._phase_1_setup_logging()
            self._phase_2_load_configuration()
            self._phase_3_init_infrastructure()
            self._phase_4_check_health()
            self._phase_5_load_models()
            self._phase_6_init_memory()
            self._phase_7_init_cognitive_engine()
            self._phase_8_init_network()
            self._phase_9_init_cli_tools()
            self._phase_10_finalize()

            return self.container

        except Exception as e:
            if self.ui_logger:
                self.ui_logger.error(f"Bootstrap failed: {e}")
            if self.file_logger:
                self.file_logger.log_error(f"Bootstrap failed: {e}")
            raise RuntimeError(f"Application bootstrap failed: {e}") from e

    def _phase_1_setup_logging(self) -> None:
        """Phase 1: Setup logging infrastructure."""
        self.ui_logger = UILogger(message_queue=self.message_queue)
        self.file_logger = FileLogger(user_data_root=self.user_data_root)

        # Register loggers in container
        self.container.register_instance('ui_logger', self.ui_logger)
        self.container.register_instance('file_logger', self.file_logger)
        self.container.register_instance('message_queue', self.message_queue)
        self.container.register_instance('username', self.username)
        self.container.register_instance('user_data_root', self.user_data_root)

    def _phase_2_load_configuration(self) -> None:
        """Phase 2: Load and validate configuration."""
        self.ui_logger.progress_message("Loading configuration", PROGRESS_LOADING_CONFIG)

        try:
            self.config = load_configuration(
                ui_logger=self.ui_logger,
                file_logger=self.file_logger
            )
            self.container.register_instance('config', self.config)

        except Exception as e:
            raise RuntimeError(f"Configuration loading failed: {e}") from e

    def _phase_3_init_infrastructure(self) -> None:
        """Phase 3: Initialize infrastructure components."""
        self.ui_logger.progress_message(
            "Initializing infrastructure",
            PROGRESS_INIT_INFRASTRUCTURE
        )

        # Register infrastructure components
        self.container.register_singleton(
            'error_handler',
            ErrorHandler,
            depends_on=['file_logger']
        )

        self.container.register_singleton(
            'health_monitor',
            HealthMonitor,
            depends_on=['config']
        )

        self.container.register_singleton(
            'metrics',
            MetricsCollector
        )

        self.container.register_singleton(
            'file_manager',
            FileManager
        )

        self.container.register_singleton(
            'data_validator',
            DataValidator
        )

    def _phase_4_check_health(self) -> None:
        """Phase 4: Perform system health checks."""
        self.ui_logger.progress_message(
            "Checking system health",
            PROGRESS_CHECKING_HEALTH
        )

        try:
            # Get health monitor and metrics from container
            health_monitor = self.container.resolve('health_monitor')
            metrics = self.container.resolve('metrics')

            # Perform health check
            with metrics.measure("bootstrap_health_check"):
                health = health_monitor.get_health_snapshot()

                # Log health status
                self.file_logger.log_info(
                    f"System health: CPU {health.cpu_percent:.1f}%, "
                    f"Memory {health.memory_percent:.1f}% "
                    f"({health.memory_available_gb:.1f}GB free)"
                )

                # Warn on degraded health
                if health.status.value != "healthy":
                    for warning in health.warnings:
                        self.ui_logger.warning(f"Health warning: {warning}")
                        self.file_logger.log_warning(warning)

        except Exception as e:
            # Health check failure is non-fatal but should be logged
            self.file_logger.log_warning(f"Health check failed: {e}")
            self.ui_logger.warning(f"Could not complete health check: {e}")

    def _phase_5_load_models(self) -> None:
        """Phase 5: Load LLM and embedding models."""
        self.ui_logger.progress_message(
            "Loading models",
            PROGRESS_LOADING_MODEL
        )

        try:
            from jenova.llm import ModelManager, EmbeddingManager, LLMInterface

            metrics = self.container.resolve('metrics')

            # Load LLM model
            with metrics.measure("bootstrap_model_load"):
                model_manager = ModelManager(
                    self.config,
                    self.file_logger,
                    self.ui_logger
                )
                llm = model_manager.load_model()

                if not llm:
                    raise RuntimeError("LLM model could not be loaded")

                # Create LLM interface
                llm_interface = LLMInterface(
                    self.config,
                    self.ui_logger,
                    self.file_logger,
                    llm
                )

                # Register in container
                self.container.register_instance('model_manager', model_manager)
                self.container.register_instance('llm', llm)
                self.container.register_instance('llm_interface', llm_interface)

            # Load embedding model
            with metrics.measure("bootstrap_embedding_load"):
                embedding_manager = EmbeddingManager(
                    self.config,
                    self.file_logger,
                    self.ui_logger
                )
                embeddings = embedding_manager.load_model()

                if not embeddings:
                    raise RuntimeError("Embedding model could not be loaded")

                # Register in container
                self.container.register_instance('embedding_manager', embedding_manager)
                self.container.register_instance('embeddings', embeddings)

            self.file_logger.log_info("Models loaded successfully")

        except Exception as e:
            # Model loading failure is critical
            self.file_logger.log_error(f"Model loading failed: {e}")
            raise RuntimeError(f"Failed to load models: {e}") from e

    def _phase_6_init_memory(self) -> None:
        """Phase 6: Initialize memory systems."""
        self.ui_logger.progress_message(
            "Initializing memory systems",
            PROGRESS_INIT_MEMORY
        )

        try:
            from jenova.memory import (
                EpisodicMemory,
                SemanticMemory,
                ProceduralMemory,
                MemoryManager
            )
            from jenova.memory.backup_manager import BackupManager

            embeddings = self.container.resolve('embeddings')

            # Initialize memory systems
            episodic_memory = EpisodicMemory(
                self.config,
                embeddings,
                self.user_data_root,
                self.file_logger
            )

            semantic_memory = SemanticMemory(
                self.config,
                embeddings,
                self.user_data_root,
                self.file_logger
            )

            procedural_memory = ProceduralMemory(
                self.config,
                embeddings,
                self.user_data_root,
                self.file_logger
            )

            # Initialize memory manager
            memory_manager = MemoryManager(
                episodic=episodic_memory,
                semantic=semantic_memory,
                procedural=procedural_memory,
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize backup manager
            backup_manager = BackupManager(
                memory_manager=memory_manager,
                config=self.config,
                user_data_root=self.user_data_root,
                file_logger=self.file_logger
            )

            # Register in container
            self.container.register_instance('episodic_memory', episodic_memory)
            self.container.register_instance('semantic_memory', semantic_memory)
            self.container.register_instance('procedural_memory', procedural_memory)
            self.container.register_instance('memory_manager', memory_manager)
            self.container.register_instance('backup_manager', backup_manager)

            self.file_logger.log_info("Memory systems initialized")

        except Exception as e:
            # Memory initialization failure is critical
            self.file_logger.log_error(f"Memory initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize memory systems: {e}") from e

    def _phase_7_init_cognitive_engine(self) -> None:
        """Phase 7: Initialize cognitive engine."""
        self.ui_logger.progress_message(
            "Initializing cognitive engine",
            PROGRESS_INIT_COGNITIVE
        )

        try:
            from jenova.cognitive_engine import CognitiveEngine, RAGSystem, MemorySearch
            from jenova.cortex import Cortex
            from jenova.insights import InsightManager
            from jenova.assumptions import AssumptionManager

            llm_interface = self.container.resolve('llm_interface')
            memory_manager = self.container.resolve('memory_manager')
            episodic_memory = self.container.resolve('episodic_memory')
            semantic_memory = self.container.resolve('semantic_memory')

            # Create cortex and insights directories
            insights_root = self.user_data_root / "insights"
            cortex_root = self.user_data_root / "cortex"
            insights_root.mkdir(exist_ok=True)
            cortex_root.mkdir(exist_ok=True)

            # Initialize RAG system
            rag_system = RAGSystem(
                llm_interface=llm_interface,
                memory_manager=memory_manager,
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize memory search
            memory_search = MemorySearch(
                episodic=episodic_memory,
                semantic=semantic_memory,
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize cortex (knowledge graph)
            cortex = Cortex(
                config=self.config,
                storage_root=cortex_root,
                file_logger=self.file_logger
            )

            # Initialize insight manager
            insight_manager = InsightManager(
                llm_interface=llm_interface,
                config=self.config,
                storage_root=insights_root,
                file_logger=self.file_logger
            )

            # Initialize assumption manager
            assumption_manager = AssumptionManager(
                llm_interface=llm_interface,
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize cognitive engine
            cognitive_engine = CognitiveEngine(
                llm_interface=llm_interface,
                memory_manager=memory_manager,
                rag_system=rag_system,
                cortex=cortex,
                insight_manager=insight_manager,
                assumption_manager=assumption_manager,
                config=self.config,
                file_logger=self.file_logger
            )

            # Register in container
            self.container.register_instance('rag_system', rag_system)
            self.container.register_instance('memory_search', memory_search)
            self.container.register_instance('cortex', cortex)
            self.container.register_instance('insight_manager', insight_manager)
            self.container.register_instance('assumption_manager', assumption_manager)
            self.container.register_instance('cognitive_engine', cognitive_engine)

            self.file_logger.log_info("Cognitive engine initialized")

        except Exception as e:
            # Cognitive engine initialization failure is critical
            self.file_logger.log_error(f"Cognitive engine initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize cognitive engine: {e}") from e

    def _phase_8_init_network(self) -> None:
        """Phase 8: Initialize network layer (if enabled)."""
        if not self.config.get('network', {}).get('enabled', False):
            self.file_logger.log_info("Network layer disabled, skipping initialization")
            return

        self.ui_logger.progress_message(
            "Initializing network layer",
            PROGRESS_INIT_NETWORK
        )

        try:
            from jenova.network import (
                SecurityManager,
                PeerManager,
                JenovaRPCClient,
                JenovaDiscoveryService
            )
            from jenova.network.rpc_service import JenovaRPCServicer, JenovaRPCServer
            from jenova.network.metrics import NetworkMetricsCollector
            from jenova.llm.distributed_llm_interface import DistributedLLMInterface
            from jenova.memory.distributed_memory_search import DistributedMemorySearch

            llm_interface = self.container.resolve('llm_interface')
            memory_search = self.container.resolve('memory_search')

            # Initialize security manager
            security_manager = SecurityManager(
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize peer manager
            peer_manager = PeerManager(
                config=self.config,
                security_manager=security_manager,
                file_logger=self.file_logger
            )

            # Initialize RPC client
            rpc_client = JenovaRPCClient(
                peer_manager=peer_manager,
                security_manager=security_manager,
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize network metrics
            network_metrics = NetworkMetricsCollector(
                peer_manager=peer_manager,
                file_logger=self.file_logger
            )

            # Initialize distributed LLM interface
            distributed_llm = DistributedLLMInterface(
                local_llm=llm_interface,
                rpc_client=rpc_client,
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize distributed memory search
            distributed_memory = DistributedMemorySearch(
                local_memory_search=memory_search,
                rpc_client=rpc_client,
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize discovery service (if enabled)
            discovery_service = None
            if self.config.get('network', {}).get('discovery_enabled', True):
                discovery_service = JenovaDiscoveryService(
                    peer_manager=peer_manager,
                    config=self.config,
                    file_logger=self.file_logger
                )
                discovery_service.start()

            # Initialize RPC server (if enabled)
            rpc_server = None
            if self.config.get('network', {}).get('server_enabled', True):
                rpc_servicer = JenovaRPCServicer(
                    llm_interface=llm_interface,
                    memory_search=memory_search,
                    file_logger=self.file_logger
                )
                rpc_server = JenovaRPCServer(
                    servicer=rpc_servicer,
                    security_manager=security_manager,
                    config=self.config,
                    file_logger=self.file_logger
                )
                rpc_server.start()

            # Register in container
            self.container.register_instance('security_manager', security_manager)
            self.container.register_instance('peer_manager', peer_manager)
            self.container.register_instance('rpc_client', rpc_client)
            self.container.register_instance('network_metrics', network_metrics)
            self.container.register_instance('distributed_llm', distributed_llm)
            self.container.register_instance('distributed_memory', distributed_memory)
            if discovery_service:
                self.container.register_instance('discovery_service', discovery_service)
            if rpc_server:
                self.container.register_instance('rpc_server', rpc_server)

            self.file_logger.log_info("Network layer initialized")

        except Exception as e:
            # Network initialization failure is non-fatal but should be logged
            self.file_logger.log_error(f"Network initialization failed: {e}")
            self.ui_logger.warning(f"Network features disabled due to initialization failure: {e}")

    def _phase_9_init_cli_tools(self) -> None:
        """Phase 9: Initialize CLI enhancement tools."""
        self.ui_logger.progress_message(
            "Initializing CLI tools",
            PROGRESS_INIT_CLI_TOOLS
        )

        try:
            # Import CLI tools
            from jenova.code_tools import (
                FileEditor,
                CodeParser,
                RefactoringEngine,
                SyntaxHighlighter,
                CodebaseMapper,
                InteractiveTerminal
            )
            from jenova.git_tools import (
                GitInterface,
                CommitAssistant,
                DiffAnalyzer,
                HooksManager,
                BranchManager
            )
            from jenova.analysis import (
                ContextOptimizer,
                CodeMetrics,
                SecurityScanner,
                IntentClassifier,
                CommandDisambiguator
            )
            from jenova.orchestration import (
                TaskPlanner,
                SubagentManager,
                ExecutionEngine,
                CheckpointManager,
                BackgroundTaskManager
            )
            from jenova.automation import (
                CustomCommandManager,
                HooksSystem,
                TemplateEngine,
                WorkflowLibrary
            )

            llm_interface = self.container.resolve('llm_interface')

            # Initialize code tools
            file_editor = FileEditor(config=self.config, file_logger=self.file_logger)
            code_parser = CodeParser(file_logger=self.file_logger)
            refactoring_engine = RefactoringEngine(
                file_editor=file_editor,
                code_parser=code_parser,
                file_logger=self.file_logger
            )
            syntax_highlighter = SyntaxHighlighter(file_logger=self.file_logger)
            codebase_mapper = CodebaseMapper(
                code_parser=code_parser,
                file_logger=self.file_logger
            )
            interactive_terminal = InteractiveTerminal(
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize git tools
            git_interface = GitInterface(config=self.config, file_logger=self.file_logger)
            commit_assistant = CommitAssistant(
                llm_interface=llm_interface,
                git_interface=git_interface,
                file_logger=self.file_logger
            )
            diff_analyzer = DiffAnalyzer(
                llm_interface=llm_interface,
                file_logger=self.file_logger
            )
            hooks_manager = HooksManager(
                git_interface=git_interface,
                file_logger=self.file_logger
            )
            branch_manager = BranchManager(
                git_interface=git_interface,
                file_logger=self.file_logger
            )

            # Initialize analysis tools
            context_optimizer = ContextOptimizer(
                llm_interface=llm_interface,
                config=self.config,
                file_logger=self.file_logger
            )
            code_metrics = CodeMetrics(file_logger=self.file_logger)
            security_scanner = SecurityScanner(file_logger=self.file_logger)
            intent_classifier = IntentClassifier(
                llm_interface=llm_interface,
                file_logger=self.file_logger
            )
            command_disambiguator = CommandDisambiguator(
                llm_interface=llm_interface,
                file_logger=self.file_logger
            )

            # Initialize orchestration tools
            task_planner = TaskPlanner(
                llm_interface=llm_interface,
                config=self.config,
                file_logger=self.file_logger
            )
            subagent_manager = SubagentManager(
                config=self.config,
                file_logger=self.file_logger
            )
            execution_engine = ExecutionEngine(
                config=self.config,
                file_logger=self.file_logger
            )
            checkpoint_manager = CheckpointManager(
                user_data_root=self.user_data_root,
                file_logger=self.file_logger
            )
            background_task_manager = BackgroundTaskManager(
                config=self.config,
                file_logger=self.file_logger
            )

            # Initialize automation tools
            custom_command_manager = CustomCommandManager(
                user_data_root=self.user_data_root,
                file_logger=self.file_logger
            )
            hooks_system = HooksSystem(
                user_data_root=self.user_data_root,
                file_logger=self.file_logger
            )
            template_engine = TemplateEngine(file_logger=self.file_logger)
            workflow_library = WorkflowLibrary(
                user_data_root=self.user_data_root,
                file_logger=self.file_logger
            )

            # Register all CLI tools in container
            self.container.register_instance('file_editor', file_editor)
            self.container.register_instance('code_parser', code_parser)
            self.container.register_instance('refactoring_engine', refactoring_engine)
            self.container.register_instance('syntax_highlighter', syntax_highlighter)
            self.container.register_instance('codebase_mapper', codebase_mapper)
            self.container.register_instance('interactive_terminal', interactive_terminal)
            self.container.register_instance('git_interface', git_interface)
            self.container.register_instance('commit_assistant', commit_assistant)
            self.container.register_instance('diff_analyzer', diff_analyzer)
            self.container.register_instance('hooks_manager', hooks_manager)
            self.container.register_instance('branch_manager', branch_manager)
            self.container.register_instance('context_optimizer', context_optimizer)
            self.container.register_instance('code_metrics', code_metrics)
            self.container.register_instance('security_scanner', security_scanner)
            self.container.register_instance('intent_classifier', intent_classifier)
            self.container.register_instance('command_disambiguator', command_disambiguator)
            self.container.register_instance('task_planner', task_planner)
            self.container.register_instance('subagent_manager', subagent_manager)
            self.container.register_instance('execution_engine', execution_engine)
            self.container.register_instance('checkpoint_manager', checkpoint_manager)
            self.container.register_instance('background_task_manager', background_task_manager)
            self.container.register_instance('custom_command_manager', custom_command_manager)
            self.container.register_instance('hooks_system', hooks_system)
            self.container.register_instance('template_engine', template_engine)
            self.container.register_instance('workflow_library', workflow_library)

            self.file_logger.log_info("CLI tools initialized")

        except Exception as e:
            # CLI tools initialization failure is non-fatal but should be logged
            self.file_logger.log_error(f"CLI tools initialization failed: {e}")
            self.ui_logger.warning(f"Some CLI features may be unavailable: {e}")

    def _phase_10_finalize(self) -> None:
        """Phase 10: Finalize initialization."""
        self.ui_logger.progress_message(
            "Initialization complete",
            PROGRESS_COMPLETE
        )

        # Calculate total startup time
        total_time = time.time() - self.start_time

        # Verify critical components
        critical_components = [
            'config', 'ui_logger', 'file_logger',
            'error_handler', 'health_monitor', 'metrics',
            'model_manager', 'llm', 'llm_interface',
            'embeddings', 'memory_manager',
            'cognitive_engine', 'rag_system'
        ]

        missing_components: List[str] = []
        for component in critical_components:
            try:
                self.container.resolve(component)
            except (KeyError, RuntimeError):
                missing_components.append(component)

        # Warn if critical components are missing
        if missing_components:
            warning = f"Warning: Missing critical components: {', '.join(missing_components)}"
            self.ui_logger.warning(warning)
            self.file_logger.log_warning(warning)

        # Get component inventory
        optional_components = [
            'security_manager', 'peer_manager', 'rpc_client', 'rpc_server',
            'discovery_service', 'distributed_llm', 'distributed_memory',
            'file_editor', 'git_interface', 'task_planner'
        ]

        active_components = []
        for component in optional_components:
            try:
                self.container.resolve(component)
                active_components.append(component)
            except (KeyError, RuntimeError):
                pass

        # Get final health snapshot
        try:
            health_monitor = self.container.resolve('health_monitor')
            health = health_monitor.get_health_snapshot()
            health_status = health.status.value
        except (KeyError, RuntimeError):
            health_status = "unknown"

        # Get metrics summary
        try:
            metrics = self.container.resolve('metrics')
            bootstrap_stats = metrics.get_stats("bootstrap_health_check")
            model_load_stats = metrics.get_stats("bootstrap_model_load")
        except (KeyError, RuntimeError):
            bootstrap_stats = None
            model_load_stats = None

        # Display startup banner
        self.ui_logger.system_message("\n" + "="*60)
        self.ui_logger.system_message("  JENOVA Cognitive Architecture - Initialized Successfully")
        self.ui_logger.system_message("="*60)
        self.ui_logger.system_message(f"  Username: {self.username}")
        self.ui_logger.system_message(f"  Startup Time: {total_time:.2f}s")
        self.ui_logger.system_message(f"  Health Status: {health_status}")
        self.ui_logger.system_message(f"  Critical Components: {len(critical_components) - len(missing_components)}/{len(critical_components)}")
        self.ui_logger.system_message(f"  Optional Components: {len(active_components)}/{len(optional_components)}")

        # Show network status if available
        if 'peer_manager' in active_components:
            self.ui_logger.system_message("  Network: Enabled")
        else:
            self.ui_logger.system_message("  Network: Disabled")

        self.ui_logger.system_message("="*60 + "\n")

        # Log detailed summary
        summary_lines = [
            "="*60,
            "Bootstrap Finalization Summary",
            "="*60,
            f"Total Startup Time: {total_time:.2f}s",
            f"Username: {self.username}",
            f"User Data Root: {self.user_data_root}",
            f"Health Status: {health_status}",
            "",
            f"Critical Components ({len(critical_components) - len(missing_components)}/{len(critical_components)}):",
        ]

        for component in critical_components:
            status = "✓" if component not in missing_components else "✗"
            summary_lines.append(f"  {status} {component}")

        summary_lines.extend([
            "",
            f"Optional Components ({len(active_components)}/{len(optional_components)}):",
        ])

        for component in optional_components:
            status = "✓" if component in active_components else "-"
            summary_lines.append(f"  {status} {component}")

        if bootstrap_stats:
            summary_lines.extend([
                "",
                "Performance Metrics:",
                f"  Health Check: {bootstrap_stats.avg_time:.2f}s",
            ])

        if model_load_stats:
            summary_lines.append(f"  Model Load: {model_load_stats.avg_time:.2f}s")

        summary_lines.append("="*60)

        # Log summary to file
        for line in summary_lines:
            self.file_logger.log_info(line)

        # Final success message
        self.file_logger.log_info("Application bootstrap completed successfully")
