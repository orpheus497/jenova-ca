# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
LLM Layer for JENOVA Cognitive Architecture.

This package provides comprehensive LLM functionality including:
- CUDA detection and management
- Model lifecycle management
- Embedding operations
- High-level LLM interface
- Timeout protection
- Resource cleanup

Phase 3 Implementation - Part of JENOVA Remediation
"""

from jenova.llm.cuda_manager import CUDAManager, CUDAInfo
from jenova.llm.model_manager import ModelManager, ModelLoadError
from jenova.llm.embedding_manager import EmbeddingManager, EmbeddingLoadError
from jenova.llm.llm_interface import LLMInterface

__all__ = [
    # CUDA Management
    'CUDAManager',
    'CUDAInfo',

    # Model Management
    'ModelManager',
    'ModelLoadError',

    # Embedding Management
    'EmbeddingManager',
    'EmbeddingLoadError',

    # LLM Interface
    'LLMInterface',
]

__version__ = '4.2.0'
__phase__ = 'Phase 3: LLM Layer'
