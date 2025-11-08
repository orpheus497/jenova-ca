# The JENOVA Cognitive Architecture - Analysis Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Context analysis and code quality module for JENOVA.

Provides context optimization, code metrics, security scanning,
intent classification, and command disambiguation capabilities.
"""

from jenova.analysis.context_optimizer import ContextOptimizer
from jenova.analysis.code_metrics import CodeMetrics
from jenova.analysis.security_scanner import SecurityScanner
from jenova.analysis.intent_classifier import IntentClassifier
from jenova.analysis.command_disambiguator import CommandDisambiguator

__all__ = [
    "ContextOptimizer",
    "CodeMetrics",
    "SecurityScanner",
    "IntentClassifier",
    "CommandDisambiguator",
]

__version__ = "5.2.0"
