##Script function and purpose: Pytest configuration and shared fixtures
##This module provides common test fixtures and configuration for all tests

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

##Block purpose: Add src directory to Python path so tests can import jenova module
# Get the project root directory (parent of tests directory)
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

##Function purpose: Create a temporary directory for test data
@pytest.fixture
def temp_user_data_dir() -> str:
    """Creates a temporary directory for user data during tests."""
    temp_dir = tempfile.mkdtemp(prefix="jenova_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

##Function purpose: Create a mock configuration dictionary for testing
@pytest.fixture
def mock_config(temp_user_data_dir: str) -> Dict[str, Any]:
    """Creates a mock configuration dictionary for testing."""
    return {
        'hardware': {
            'threads': 4,
            'gpu_layers': 0,
            'mlock': False
        },
        'model': {
            'model_path': '',
            'embedding_model': 'all-MiniLM-L6-v2',
            'context_size': 4096,
            'max_tokens': 1500,
            'temperature': 0.4,
            'top_p': 0.9
        },
        'memory': {
            'preload_memories': False,
            'episodic_db_path': os.path.join(temp_user_data_dir, 'memory_db/episodic'),
            'semantic_db_path': os.path.join(temp_user_data_dir, 'memory_db/semantic'),
            'procedural_db_path': os.path.join(temp_user_data_dir, 'memory_db/procedural')
        },
        'persona': {
            'identity': {
                'name': 'JENOVA',
                'creator': 'orpheus497',
                'creator_alias': 'The Architect',
                'origin_story': 'Test AI',
                'type': 'Test AI'
            },
            'directives': ['Test directive'],
            'initial_facts': []
        },
        'scheduler': {
            'generate_insight_interval': 5,
            'generate_assumption_interval': 7,
            'proactively_verify_assumption_interval': 8,
            'reflect_interval': 10
        },
        'memory_search': {
            'semantic_n_results': 5,
            'episodic_n_results': 3,
            'procedural_n_results': 3,
            'insight_n_results': 5
        },
        'cortex': {
            'relationship_weights': {
                'elaborates_on': 1.5,
                'conflicts_with': 2.0,
                'related_to': 1.0
            },
            'pruning': {
                'enabled': False,
                'prune_interval': 10,
                'max_age_days': 30,
                'min_centrality': 0.1
            }
        },
        'user_data_root': temp_user_data_dir
    }

##Function purpose: Create a mock UI logger for testing
@pytest.fixture
def mock_ui_logger() -> Mock:
    """Creates a mock UI logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.system_message = Mock()
    logger.thinking_process = MagicMock()
    logger.cognitive_process = MagicMock()
    return logger

##Function purpose: Create a mock file logger for testing
@pytest.fixture
def mock_file_logger() -> Mock:
    """Creates a mock file logger for testing."""
    logger = Mock()
    logger.log_info = Mock()
    logger.log_warning = Mock()
    logger.log_error = Mock()
    return logger

##Function purpose: Create a mock LLM interface for testing
@pytest.fixture
def mock_llm_interface() -> Mock:
    """Creates a mock LLM interface for testing."""
    llm = Mock()
    llm.generate = Mock(return_value="Test response")
    llm.model = Mock()
    llm.close = Mock()
    return llm

