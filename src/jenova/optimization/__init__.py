# The JENOVA Cognitive Architecture - Optimization Module
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 25: Self-Optimization Engine.

Autonomous parameter optimization through Bayesian optimization, performance
tracking, and adaptive tuning based on task types and user feedback.

Modules:
    - SelfTuner: Main optimization orchestrator
    - PerformanceDB: SQLite database for metrics storage
    - BayesianOptimizer: Gaussian Process-based parameter search
    - TaskClassifier: Automatic task type detection

Example:
    >>> from jenova.optimization import SelfTuner, PerformanceDB
    >>> from pathlib import Path
    >>>
    >>> db = PerformanceDB(Path("~/.jenova-ai/performance.db"))
    >>> tuner = SelfTuner(performance_db=db, config=config)
    >>>
    >>> # Optimize parameters for task type
    >>> best = tuner.optimize_parameters("code_generation", iterations=30)
    >>> tuner.apply_optimal_parameters("code_generation")
    >>>
    >>> # Record performance
    >>> tuner.record_performance(
    ...     "code_generation",
    ...     {"temperature": 0.3, "max_tokens": 512},
    ...     duration=5.2,
    ...     quality_score=0.89
    ... )
"""

from jenova.optimization.self_tuner import SelfTuner
from jenova.optimization.performance_db import PerformanceDB
from jenova.optimization.bayesian_optimizer import BayesianOptimizer
from jenova.optimization.task_classifier import TaskClassifier

__all__ = [
    "SelfTuner",
    "PerformanceDB",
    "BayesianOptimizer",
    "TaskClassifier",
]
