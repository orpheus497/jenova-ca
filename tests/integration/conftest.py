##Script function and purpose: Integration test fixtures - Provides real component instances for integration testing.
##Dependency purpose: Creates actual instances of JENOVA components with temporary storage for testing.
"""Integration test fixtures for JENOVA.

Provides fixtures that create real component instances with:
- Temporary directories for data storage
- Real ChromaDB instances
- Mock LLM for deterministic testing (no GGUF model required)
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from chromadb.api.types import EmbeddingFunction

from jenova.config.models import (
    GraphConfig,
    HardwareConfig,
    JenovaConfig,
    MemoryConfig,
    ModelConfig,
    PersonaConfig,
)
from jenova.core.engine import CognitiveEngine, EngineConfig
from jenova.core.knowledge import KnowledgeStore
from jenova.core.response import ResponseConfig, ResponseGenerator
from jenova.llm.types import Completion, GenerationParams, Prompt
from jenova.memory import Memory


##Class purpose: Mock embedding function to avoid ChromaDB ONNX/numpy incompatibility in CI
class MockEmbeddingFunction(EmbeddingFunction):
    """Mock embedding function returning fixed-dimension vectors."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in input]


_original_memory_init = Memory.__init__


def _patched_memory_init(self, memory_type, storage_path, embedding_function=None):
    """Patch Memory.__init__ to always use MockEmbeddingFunction in integration tests."""
    _original_memory_init(
        self,
        memory_type=memory_type,
        storage_path=storage_path,
        embedding_function=MockEmbeddingFunction(),
    )


##Function purpose: Auto-patch Memory to use mock embeddings for all integration tests
@pytest.fixture(autouse=True)
def _use_mock_embeddings(monkeypatch):
    """Ensure all Memory instances use MockEmbeddingFunction to avoid ONNX/numpy issues."""
    monkeypatch.setattr(Memory, "__init__", _patched_memory_init)


##Class purpose: Mock LLM interface for deterministic testing
class MockLLMInterface:
    """
    Mock LLM for integration testing.

    Provides deterministic responses without requiring a GGUF model.
    Responses are based on input patterns for testing specific behaviors.
    """

    ##Method purpose: Initialize mock with optional response patterns
    def __init__(self, default_response: str = "This is a test response from JENOVA.") -> None:
        """
        Initialize mock LLM.

        Args:
            default_response: Default response when no pattern matches.
        """
        self._default_response = default_response
        self._response_patterns: dict[str, str] = {}
        self._call_count: int = 0
        self._last_prompt: Prompt | None = None

    ##Method purpose: Check if model is loaded (always True for mock)
    @property
    def is_loaded(self) -> bool:
        """Mock is always 'loaded'."""
        return True

    ##Method purpose: Add a response pattern for testing
    def add_response_pattern(self, contains: str, response: str) -> None:
        """
        Add a pattern-based response.

        Args:
            contains: If prompt contains this string, use this response.
            response: Response to return.
        """
        self._response_patterns[contains.lower()] = response

    ##Method purpose: Generate completion from prompt
    def generate(
        self,
        prompt: Prompt,
        params: GenerationParams | None = None,
    ) -> Completion:
        """
        Generate mock completion.

        Args:
            prompt: Structured prompt
            params: Generation parameters (ignored for mock)

        Returns:
            Mock completion result
        """
        ##Step purpose: Track call for assertions
        self._call_count += 1
        self._last_prompt = prompt

        ##Step purpose: Format prompt to check patterns
        prompt_text = prompt.format_chat().lower()

        ##Loop purpose: Check for matching patterns
        for pattern, response in self._response_patterns.items():
            ##Condition purpose: Return pattern response if matched
            if pattern in prompt_text:
                return Completion(
                    content=response,
                    finish_reason="stop",
                    tokens_generated=len(response.split()),
                    tokens_prompt=len(prompt_text.split()),
                    generation_time_ms=10.0,
                )

        ##Step purpose: Return default response
        return Completion(
            content=self._default_response,
            finish_reason="stop",
            tokens_generated=len(self._default_response.split()),
            tokens_prompt=len(prompt_text.split()),
            generation_time_ms=10.0,
        )

    ##Method purpose: Simple text generation helper
    def generate_text(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        params: GenerationParams | None = None,
    ) -> str:
        """Simple text generation helper."""
        prompt = Prompt(system=system_prompt, user_message=text)
        completion = self.generate(prompt, params)
        return completion.content

    ##Method purpose: Reset mock state for clean test
    def reset(self) -> None:
        """Reset call count and last prompt."""
        self._call_count = 0
        self._last_prompt = None


##Function purpose: Provide temporary data directory for integration tests
@pytest.fixture
def integration_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for integration tests.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to temporary data directory.
    """
    ##Step purpose: Create data directory structure
    data_dir = tmp_path / "jenova-test-data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ##Step purpose: Create subdirectories for memory and graph
    (data_dir / "memory").mkdir(exist_ok=True)
    (data_dir / "graph").mkdir(exist_ok=True)

    return data_dir


##Function purpose: Provide test configuration for integration tests
@pytest.fixture
def integration_config(integration_data_dir: Path) -> JenovaConfig:
    """Create a test configuration for integration tests.

    Args:
        integration_data_dir: Temporary data directory.

    Returns:
        JenovaConfig configured for testing.
    """
    ##Step purpose: Create config with test-specific paths
    return JenovaConfig(
        hardware=HardwareConfig(
            threads=1,  # Single thread for test consistency
            gpu_layers="none",  # No GPU for tests
        ),
        model=ModelConfig(
            model_path="auto",  # Won't be used with mock LLM
            context_length=2048,
            temperature=0.1,  # Low temperature for deterministic tests
            max_tokens=256,
        ),
        memory=MemoryConfig(
            storage_path=integration_data_dir / "memory",
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        ),
        graph=GraphConfig(
            storage_path=integration_data_dir / "graph",
            max_depth=2,
        ),
        persona=PersonaConfig(
            name="JENOVA-TEST",
            directives=["Be helpful", "Answer concisely"],
            system_prompt="You are JENOVA-TEST, a test AI assistant.",
        ),
        debug=True,
    )


##Function purpose: Provide mock LLM for integration tests
@pytest.fixture
def mock_llm() -> MockLLMInterface:
    """Create a mock LLM for integration tests.

    Returns:
        MockLLMInterface instance with default responses.
    """
    llm = MockLLMInterface(default_response="I understand your query. This is a test response.")

    ##Step purpose: Add common test patterns
    llm.add_response_pattern("hello", "Hello! How can I help you today?")
    llm.add_response_pattern("name", "I am JENOVA-TEST, your AI assistant.")
    llm.add_response_pattern(
        "favorite color", "Based on our previous conversation, your favorite color is blue."
    )
    llm.add_response_pattern("capital of france", "The capital of France is Paris.")

    return llm


##Function purpose: Provide real KnowledgeStore instance for integration tests
@pytest.fixture
def knowledge_store(integration_config: JenovaConfig) -> Generator[KnowledgeStore, None, None]:
    """Create a real KnowledgeStore for integration tests.

    Args:
        integration_config: Test configuration.

    Yields:
        Configured KnowledgeStore instance.
    """
    ##Action purpose: Create knowledge store with real ChromaDB
    store = KnowledgeStore(
        memory_config=integration_config.memory,
        graph_config=integration_config.graph,
    )

    yield store

    ##Step purpose: Cleanup is handled by tmp_path fixture


##Function purpose: Provide response generator for integration tests
@pytest.fixture
def response_generator(integration_config: JenovaConfig) -> ResponseGenerator:
    """Create a ResponseGenerator for integration tests.

    Args:
        integration_config: Test configuration.

    Returns:
        Configured ResponseGenerator instance.
    """
    return ResponseGenerator(
        config=integration_config,
        response_config=ResponseConfig(
            include_sources=True,
            max_length=0,  # No limit for tests
            format_style="default",
        ),
    )


##Function purpose: Provide real CognitiveEngine instance for integration tests
@pytest.fixture
def cognitive_engine(
    integration_config: JenovaConfig,
    knowledge_store: KnowledgeStore,
    mock_llm: MockLLMInterface,
    response_generator: ResponseGenerator,
) -> Generator[CognitiveEngine, None, None]:
    """Create a real CognitiveEngine for integration tests.

    Args:
        integration_config: Test configuration.
        knowledge_store: Knowledge store fixture.
        mock_llm: Mock LLM fixture.
        response_generator: Response generator fixture.

    Yields:
        Configured CognitiveEngine instance.
    """
    ##Action purpose: Create cognitive engine with real components and mock LLM
    engine = CognitiveEngine(
        config=integration_config,
        knowledge_store=knowledge_store,
        llm=mock_llm,
        response_generator=response_generator,
        engine_config=EngineConfig(
            max_context_items=5,
            temperature=0.1,
            enable_learning=True,
            max_history_turns=5,
        ),
    )

    yield engine

    ##Step purpose: Cleanup - reset engine state
    engine.reset()
