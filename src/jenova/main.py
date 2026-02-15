"""
JENOVA CLI Entry Point

Main entry point with command-line argument parsing.
Supports config override, debug mode, and headless operation.
Wires together all components: CognitiveEngine, KnowledgeStore, LLM, UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

##Fix: Import Python 3.14 compatibility patches FIRST (BH-2026-02-11T02:12:55Z)
##Note: This must come before any imports that use ChromaDB/Pydantic V1
import jenova.compat_py314  # noqa: F401
from jenova import __version__
from jenova.config import JenovaConfig, load_config
from jenova.utils.logging import configure_logging, get_logger

##Step purpose: Common exception tuple for optional subsystem initialization
##Refactor: Removed AttributeError/TypeError so programming bugs propagate (D3-2026-02-14T10:24:30Z)
_SUBSYSTEM_INIT_EXCEPTIONS = (
    RuntimeError, ValueError, KeyError, ImportError
)

if TYPE_CHECKING:
    from jenova.core.engine import CognitiveEngine


##Class purpose: Protocol for LLM used by CognitiveEngine
@runtime_checkable
class EngineLLMProtocol(Protocol):
    """Minimal LLM protocol that CognitiveEngine actually requires.

    Both LLMInterface and DevelopmentLLM must satisfy this protocol.
    """

    @property
    def is_loaded(self) -> bool: ...

    def generate(self, prompt: object, params: object = None) -> object: ...

    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        params: object = None,
    ) -> str: ...


##Class purpose: Minimal mock LLM for development/testing
class DevelopmentLLM:
    """Development mock LLM that returns echo responses."""

    @property
    def is_loaded(self) -> bool:
        return True

    def generate(self, prompt: object, params: object = None) -> object:
        from jenova.llm.types import Completion
        from jenova.llm.types import Prompt as LLMPrompt

        ##Step purpose: Return echo response
        user_msg = prompt.user_message if isinstance(prompt, LLMPrompt) else str(prompt)
        return Completion(
            content=f"[DEV MODE] Received: {user_msg}",
            finish_reason="stop",
            tokens_generated=10,
            tokens_prompt=len(user_msg.split()),
            generation_time_ms=1.0,
        )

    ##Update: WIRING-001 (2026-02-13T11:26:36Z) — Satisfy graph.LLMProtocol for scheduler tasks
    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        params: object = None,
    ) -> str:
        """Simple text generation for dev mode."""
        return f"[DEV MODE] {text[:100]}"


##Class purpose: Adapter for LLMInterface to match ProactiveEngine LLMProtocol
class ProactiveLLMWrapper:
    """Wraps an LLMInterface (or DevelopmentLLM) for ProactiveEngine."""

    def __init__(self, llm_interface: object) -> None:
        self._llm = llm_interface

    def generate(self, prompt: str) -> str:
        return self._llm.generate_text(
            prompt, system_prompt="You are a proactive cognitive assistant."
        )


##Function purpose: Parse command-line arguments
def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: Arguments to parse (None = sys.argv)

    Returns:
        Parsed arguments namespace
    """
    ##Step purpose: Create parser with description
    parser = argparse.ArgumentParser(
        prog="jenova",
        description="JENOVA - Self-Aware AI Cognitive Architecture",
    )

    ##Action purpose: Add version argument
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"JENOVA {__version__}",
    )

    ##Action purpose: Add config argument
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to configuration YAML file",
    )

    ##Action purpose: Add debug argument
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )

    ##Action purpose: Add no-tui argument
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Run in headless mode without TUI",
    )

    ##Action purpose: Add log-file argument
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (logs to stdout if not specified)",
    )

    ##Action purpose: Add json-logs argument
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format",
    )

    ##Action purpose: Add skip-model-load argument for testing
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip loading LLM model (for testing/development)",
    )

    return parser.parse_args(args)


##Function purpose: Create the cognitive engine with all dependencies
def create_engine(config: JenovaConfig, skip_model_load: bool = False) -> CognitiveEngine:
    """
    Create and wire up the CognitiveEngine with all dependencies.

    Args:
        config: JENOVA configuration
        skip_model_load: If True, skip loading the LLM model (uses mock)

    Returns:
        Configured CognitiveEngine instance

    Raises:
        Exception: If component initialization fails
    """
    logger = get_logger(__name__)

    ##Step purpose: Import components
    from jenova.assumptions.manager import AssumptionManager
    from jenova.core.engine import CognitiveEngine, EngineConfig
    from jenova.core.integration import IntegrationHub
    from jenova.core.knowledge import KnowledgeStore
    from jenova.core.response import ResponseConfig, ResponseGenerator
    from jenova.core.scheduler import CognitiveScheduler, SchedulerConfig
    from jenova.core.task_executor import CognitiveTaskExecutor
    from jenova.graph.proactive import ProactiveEngine
    from jenova.insights.manager import InsightManager
    from jenova.llm.interface import LLMInterface
    from jenova.memory.types import MemoryType

    ##Update: WIRING-005 (2026-02-14) - Import Web Search Provider
    from jenova.tools.web_search import DuckDuckGoSearchProvider

    ##Action purpose: Log initialization start
    logger.info("creating_engine", skip_model_load=skip_model_load)

    ##Step purpose: Create KnowledgeStore with memory and graph
    logger.debug("initializing_knowledge_store")
    knowledge_store = KnowledgeStore(
        memory_config=config.memory,
        graph_config=config.graph,
    )

    ##Step purpose: Create LLM interface
    logger.debug("initializing_llm", skip_load=skip_model_load)
    ##Condition purpose: Use mock or real LLM based on flag
    if skip_model_load:
        llm: LLMInterface | DevelopmentLLM = DevelopmentLLM()
    else:
        ##Step purpose: Create and load real LLM
        llm = LLMInterface(
            model_config=config.model,
            hardware_config=config.hardware,
        )
        llm.load()

    ##Step purpose: Create ResponseGenerator
    logger.debug("initializing_response_generator")
    ##Update: WIRING-005 (2026-02-14) - Wire Web Search
    # Note: Use standard 10s timeout, could be moved to config if needed
    web_search_provider = DuckDuckGoSearchProvider(timeout=10)

    response_generator = ResponseGenerator(
        config=config,
        response_config=ResponseConfig(
            include_sources=True,
            max_length=0,  # No limit
            format_style="default",
        ),
        web_search=web_search_provider,
    )

    ##Step purpose: Initialize insight manager
    logger.debug("initializing_insight_manager")
    ##Step purpose: Determine insights storage path (user-specific subdirectories handled by manager)
    insights_root = config.memory.storage_path.parent / "insights"
    insight_manager = InsightManager.create(
        insights_root=insights_root,
        graph=knowledge_store.graph,
        llm=llm,
        memory_search=None,  ##Step purpose: Optional - not available in v4 architecture
    )

    ##Step purpose: Initialize assumption manager
    logger.debug("initializing_assumption_manager")
    ##Step purpose: Determine assumptions storage path (user-specific subdirectories handled by manager)
    assumptions_root = config.memory.storage_path.parent / "assumptions"
    assumption_manager = AssumptionManager.create(
        storage_path=assumptions_root,
        graph=knowledge_store.graph,
        llm=llm,
    )

    ##Update: WIRING-003 (2026-02-14) — Initialize ProactiveEngine
    logger.debug("initializing_proactive_engine")

    ##Note: ProactiveConfig (Pydantic) and ProactiveConfig (dataclass) must stay in sync.
    ##TODO: Centralize canonical schema in a shared location if divergence becomes frequent.
    ##See: jenova.config.models.ProactiveConfig and jenova.graph.proactive.ProactiveConfig
    proactive_config = config.proactive.to_proactive_config()
    proactive_engine = ProactiveEngine(
        config=proactive_config,
        graph=knowledge_store.graph,
        llm=ProactiveLLMWrapper(llm),
    )
    proactive_engine.set_assumption_manager(assumption_manager)

    ##Step purpose: Create CognitiveEngine
    logger.debug("initializing_cognitive_engine")
    ##Refactor: Explicit check replaces assert for -O safety (D3-2026-02-14T10:24:30Z)
    if not isinstance(llm, EngineLLMProtocol):
        raise TypeError(
            f"{type(llm).__name__} does not implement EngineLLMProtocol"
        )
    engine = CognitiveEngine(
        config=config,
        knowledge_store=knowledge_store,
        llm=llm,  # type: ignore[arg-type]  # EngineLLMProtocol verified at runtime
        response_generator=response_generator,
        engine_config=EngineConfig(
            max_context_items=config.memory.max_results,
            temperature=config.model.temperature,
            enable_learning=True,
            max_history_turns=10,
            planning=config.planning.to_planning_config(),
        ),
        insight_manager=insight_manager,
        assumption_manager=assumption_manager,
        proactive_engine=proactive_engine,
    )

    ##Update: WIRING-002 (2026-02-13T13:05:14Z) — Wire IntegrationHub into engine
    logger.debug("initializing_integration_hub")
    try:
        semantic_memory = knowledge_store.get_memory(MemoryType.SEMANTIC)
        integration_hub = IntegrationHub(
            graph=knowledge_store.graph,
            memory=semantic_memory,
            config=config.integration,
        )
        engine.set_integration_hub(integration_hub)
    except _SUBSYSTEM_INIT_EXCEPTIONS as e:
        ##Fix: Narrow catch and include exc_info for better diagnostics (PATCH-007)
        logger.warning(
            "integration_hub_init_failed",
            error=str(e),
            msg="Continuing without integration subsystem",
            exc_info=True,
        )

    ##Update: WIRING-001 (2026-02-13T11:26:36Z) — Wire CognitiveScheduler into engine
    logger.debug("initializing_scheduler")
    try:
        task_executor = CognitiveTaskExecutor(
            insight_manager=insight_manager,
            assumption_manager=assumption_manager,
            knowledge_store=knowledge_store,
            llm=llm,
            get_recent_history=engine.get_recent_history,
        )
        scheduler = CognitiveScheduler(SchedulerConfig(), executor=task_executor)
        engine.set_scheduler(scheduler)
    except _SUBSYSTEM_INIT_EXCEPTIONS as e:
        ##Fix: Handle scheduler initialization failure and log traceback (PATCH-008)
        logger.warning(
            "scheduler_init_failed",
            error=str(e),
            msg="Continuing without scheduler subsystem",
            exc_info=True,
        )

    logger.info("engine_created")
    return engine


##Function purpose: Run JENOVA in headless mode
def run_headless(config: JenovaConfig, engine: CognitiveEngine) -> None:
    """
    Run JENOVA in headless mode (no TUI).

    Args:
        config: JENOVA configuration
        engine: Cognitive engine instance
    """
    logger = get_logger(__name__)
    logger.info("starting_headless", config_debug=config.debug)

    ##Step purpose: Print startup message
    print(f"JENOVA {__version__} - Headless Mode")
    print(f"Persona: {config.persona.name}")
    print("Type 'quit' or 'exit' to stop.\n")

    ##Loop purpose: Main REPL loop
    while True:
        ##Error purpose: Handle keyboard interrupt
        try:
            ##Action purpose: Read user input
            user_input = input("You: ").strip()

            ##Condition purpose: Check for exit commands
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            ##Condition purpose: Skip empty input
            if not user_input:
                continue

            ##Condition purpose: Handle special commands
            if user_input.lower() == "/help":
                print("\nCommands:")
                print("  /help    - Show this help")
                print("  /reset   - Reset conversation")
                print("  /debug   - Toggle debug mode")
                print("  quit     - Exit JENOVA\n")
                continue

            if user_input.lower() == "/reset":
                engine.reset()
                print(">> Conversation reset.\n")
                continue

            if user_input.lower() == "/debug":
                ##Step purpose: Toggle debug logging
                current = logger.isEnabledFor(10)  # DEBUG level
                print(f">> Debug mode: {'off' if current else 'on'}\n")
                continue

            ##Action purpose: Process through cognitive engine
            result = engine.think(user_input)

            ##Condition purpose: Handle error responses
            if result.is_error:
                print(f"JENOVA: [Error] {result.error_message}")
            else:
                print(f"JENOVA: {result.content}")

            print()  ##Step purpose: Add spacing

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


##Function purpose: Run JENOVA with TUI
def run_tui(config: JenovaConfig, engine: CognitiveEngine) -> None:
    """
    Run JENOVA with TUI interface.

    Args:
        config: JENOVA configuration
        engine: Cognitive engine instance
    """
    from jenova.ui.app import JenovaApp

    ##Step purpose: Create app and connect engine
    app = JenovaApp(config)
    app.set_engine(engine)

    ##Action purpose: Run the app
    app.run()


##Function purpose: Main entry point
def main(args: list[str] | None = None) -> int:
    """
    Main entry point for JENOVA.

    Args:
        args: Command-line arguments (None = sys.argv)

    Returns:
        Exit code (0 = success)
    """
    ##Step purpose: Parse arguments
    parsed = parse_args(args)

    ##Step purpose: Configure logging
    log_level = "DEBUG" if parsed.debug else "INFO"
    configure_logging(
        level=log_level,
        log_file=parsed.log_file,
        json_format=parsed.json_logs,
    )

    logger = get_logger(__name__)
    logger.info("jenova_starting", version=__version__)

    ##Error purpose: Handle configuration errors
    try:
        ##Step purpose: Load configuration
        config = load_config(parsed.config)

        ##Condition purpose: Override debug from CLI
        if parsed.debug:
            config = config.model_copy(update={"debug": True})

    except Exception as e:
        logger.error("config_error", error=str(e))
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    ##Error purpose: Handle engine initialization errors
    try:
        ##Step purpose: Create the cognitive engine
        engine = create_engine(
            config=config,
            skip_model_load=parsed.skip_model_load,
        )
    except Exception as e:
        logger.error("engine_init_error", error=str(e), exc_info=True)
        print(f"Failed to initialize engine: {e}", file=sys.stderr)
        return 1

    ##Error purpose: Handle runtime errors
    try:
        ##Condition purpose: Choose run mode
        if parsed.no_tui:
            run_headless(config, engine)
        else:
            run_tui(config, engine)
    except Exception as e:
        logger.error("runtime_error", error=str(e), exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    logger.info("jenova_stopped")
    return 0


##Condition purpose: Run main when executed directly
if __name__ == "__main__":
    sys.exit(main())
