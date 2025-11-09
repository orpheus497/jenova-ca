# The JENOVA Cognitive Architecture - Self-Tuner
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 25: Self-Tuner for autonomous parameter optimization.

Automatically learns optimal parameters for different task types using
Bayesian optimization, A/B testing, and user feedback integration.
"""

from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import time

from jenova.optimization.performance_db import PerformanceDB
from jenova.optimization.bayesian_optimizer import BayesianOptimizer


class SelfTuner:
    """
    Autonomous parameter optimization engine.

    Features:
        - Tracks performance metrics per task type
        - Runs Bayesian optimization to find optimal parameters
        - A/B tests parameter variations
        - Auto-adjusts based on user feedback
        - Maintains performance history for analysis

    Example:
        >>> tuner = SelfTuner(
        ...     performance_db=db,
        ...     config=config,
        ...     llm_interface=llm
        ... )
        >>> best = tuner.optimize_parameters("code_generation", iterations=50)
        >>> tuner.apply_optimal_parameters("code_generation")
    """

    def __init__(
        self,
        performance_db: PerformanceDB,
        config: Dict[str, Any],
        llm_interface: Optional[Any] = None,
    ):
        """
        Initialize self-tuner.

        Args:
            performance_db: Performance database for metrics storage
            config: System configuration dictionary
            llm_interface: Optional LLM interface for evaluation
        """
        self.performance_db = performance_db
        self.config = config
        self.llm_interface = llm_interface

        # Define parameter search space
        self.parameter_space = {
            "temperature": (0.0, 1.0),
            "top_p": (0.0, 1.0),
            "max_tokens": (128, 2048),
            # context_size is model-dependent, handle separately
        }

        # Current active parameters per task type
        self.active_parameters: Dict[str, Dict[str, Any]] = {}

        # Load default parameters from config
        self._load_default_parameters()

    def _load_default_parameters(self) -> None:
        """Load default parameters from system configuration."""
        defaults = {
            "temperature": self.config.get("model", {}).get("temperature", 0.7),
            "top_p": self.config.get("model", {}).get("top_p", 0.95),
            "max_tokens": self.config.get("model", {}).get("max_tokens", 512),
        }

        # Set defaults for all known task types
        task_types = [
            "general_qa",
            "code_generation",
            "summarization",
            "analysis",
            "creative_writing",
        ]

        for task_type in task_types:
            self.active_parameters[task_type] = defaults.copy()

    def optimize_parameters(
        self, task_type: str, iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization for task type.

        Args:
            task_type: Task type to optimize
            iterations: Number of optimization iterations

        Returns:
            Best parameters found

        Example:
            >>> best = tuner.optimize_parameters("summarization", iterations=30)
            >>> print(best)  # {'temperature': 0.3, 'max_tokens': 256, ...}
        """
        # Define objective function
        def objective(params: Dict[str, Any]) -> float:
            """Evaluate parameter set."""
            # Record test run
            start_time = time.time()

            # Simulate or run actual task (in production, run real tasks)
            quality_score = self._evaluate_parameters(task_type, params)

            duration = time.time() - start_time

            # Record performance
            self.performance_db.record_task_run(
                task_type=task_type,
                parameters=params,
                duration=duration,
                quality_score=quality_score,
            )

            return quality_score

        # Create optimizer
        optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space, objective_function=objective
        )

        # Run optimization
        best_params = optimizer.optimize(n_iterations=iterations)

        # Record optimization
        metrics = optimizer.get_convergence_metrics()
        if metrics:
            self.performance_db.record_optimization(
                task_type=task_type,
                best_params=best_params,
                iterations=iterations,
                convergence_score=metrics.get("std_value", 0.0),
                initial_quality=optimizer.y_observed[0]
                if optimizer.y_observed
                else 0.0,
                final_quality=metrics.get("best_value", 0.0),
            )

        return best_params

    def _evaluate_parameters(
        self, task_type: str, params: Dict[str, Any]
    ) -> float:
        """
        Evaluate parameter set for task type.

        In production, this would run actual tasks. For now, uses heuristics.

        Args:
            task_type: Task type
            params: Parameters to evaluate

        Returns:
            Quality score 0.0-1.0
        """
        # Task-specific optimal ranges (heuristics)
        optimal_ranges = {
            "general_qa": {"temperature": (0.6, 0.8), "max_tokens": (256, 512)},
            "code_generation": {"temperature": (0.2, 0.4), "max_tokens": (512, 1024)},
            "summarization": {"temperature": (0.2, 0.4), "max_tokens": (128, 256)},
            "analysis": {"temperature": (0.3, 0.5), "max_tokens": (512, 1024)},
            "creative_writing": {"temperature": (0.8, 1.0), "max_tokens": (512, 1536)},
        }

        ranges = optimal_ranges.get(task_type, {})

        # Calculate score based on proximity to optimal ranges
        score = 1.0

        for param_name, (optimal_low, optimal_high) in ranges.items():
            if param_name in params:
                value = params[param_name]
                optimal_mid = (optimal_low + optimal_high) / 2.0
                optimal_range = optimal_high - optimal_low

                # Distance from optimal
                distance = abs(value - optimal_mid) / optimal_range
                # Penalty for distance (0 penalty at optimal, up to 0.5 penalty at extremes)
                penalty = min(0.5, distance * 0.5)
                score -= penalty

        # Add small random noise for realism
        import random

        score += random.uniform(-0.05, 0.05)

        return max(0.0, min(1.0, score))

    def record_performance(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        duration: float,
        quality_score: Optional[float] = None,
        user_feedback: Optional[str] = None,
    ) -> None:
        """
        Record performance for parameter set.

        Args:
            task_type: Task type
            parameters: Parameters used
            duration: Execution duration
            quality_score: Quality score (optional)
            user_feedback: User feedback (optional)

        Example:
            >>> tuner.record_performance(
            ...     "code_generation",
            ...     {"temperature": 0.3, "max_tokens": 512},
            ...     duration=5.2,
            ...     quality_score=0.89,
            ...     user_feedback="good"
            ... )
        """
        self.performance_db.record_task_run(
            task_type=task_type,
            parameters=parameters,
            duration=duration,
            quality_score=quality_score,
            user_feedback=user_feedback,
        )

    def get_optimal_parameters(self, task_type: str) -> Dict[str, Any]:
        """
        Get current optimal parameters for task type.

        Args:
            task_type: Task type

        Returns:
            Optimal parameters

        Example:
            >>> params = tuner.get_optimal_parameters("summarization")
        """
        # Check database for best parameters
        best = self.performance_db.get_best_parameters(task_type)

        if best:
            return best

        # Fallback to active parameters
        return self.active_parameters.get(
            task_type, self.active_parameters.get("general_qa", {})
        )

    def apply_optimal_parameters(self, task_type: str) -> None:
        """
        Apply optimal parameters to active configuration.

        Args:
            task_type: Task type

        Example:
            >>> tuner.apply_optimal_parameters("code_generation")
        """
        optimal = self.get_optimal_parameters(task_type)
        self.active_parameters[task_type] = optimal

    def a_b_test(
        self,
        task_type: str,
        param_set_a: Dict[str, Any],
        param_set_b: Dict[str, Any],
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        A/B test two parameter sets.

        Args:
            task_type: Task type
            param_set_a: First parameter set
            param_set_b: Second parameter set
            n_trials: Number of trials per set

        Returns:
            Dict with test results

        Example:
            >>> results = tuner.a_b_test(
            ...     "general_qa",
            ...     {"temperature": 0.7},
            ...     {"temperature": 0.5},
            ...     n_trials=10
            ... )
            >>> print(results["winner"])  # "A" or "B"
        """
        results_a = []
        results_b = []

        # Run trials for A
        for _ in range(n_trials):
            score = self._evaluate_parameters(task_type, param_set_a)
            results_a.append(score)
            self.record_performance(
                task_type, param_set_a, duration=1.0, quality_score=score
            )

        # Run trials for B
        for _ in range(n_trials):
            score = self._evaluate_parameters(task_type, param_set_b)
            results_b.append(score)
            self.record_performance(
                task_type, param_set_b, duration=1.0, quality_score=score
            )

        # Calculate statistics
        import numpy as np

        mean_a = np.mean(results_a)
        mean_b = np.mean(results_b)
        std_a = np.std(results_a)
        std_b = np.std(results_b)

        # Determine winner
        winner = "A" if mean_a > mean_b else "B"
        confidence = abs(mean_a - mean_b) / max(std_a, std_b, 0.01)

        return {
            "param_set_a": param_set_a,
            "param_set_b": param_set_b,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": std_a,
            "std_b": std_b,
            "winner": winner,
            "confidence": confidence,
            "trials": n_trials,
        }

    def auto_adjust(
        self, task_type: str, user_feedback: str, current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Auto-adjust parameters based on user feedback.

        Args:
            task_type: Task type
            user_feedback: User feedback ("good", "bad", "too_creative", etc.)
            current_params: Current parameters

        Returns:
            Adjusted parameters

        Example:
            >>> adjusted = tuner.auto_adjust(
            ...     "code_generation",
            ...     "too_creative",
            ...     {"temperature": 0.8}
            ... )
            >>> # Returns: {"temperature": 0.6}
        """
        adjusted = current_params.copy()

        # Feedback-based adjustments
        feedback_adjustments = {
            "too_creative": {"temperature": -0.1},
            "too_boring": {"temperature": +0.1},
            "too_long": {"max_tokens": -128},
            "too_short": {"max_tokens": +128},
            "good": {},  # No adjustment
            "bad": {"temperature": -0.05},  # Slight decrease for more focus
        }

        adjustments = feedback_adjustments.get(user_feedback, {})

        for param, delta in adjustments.items():
            if param in adjusted:
                new_value = adjusted[param] + delta

                # Clip to bounds
                if param in self.parameter_space:
                    low, high = self.parameter_space[param]
                    adjusted[param] = max(low, min(high, new_value))

        return adjusted

    def get_optimization_history(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Get optimization history for task type.

        Args:
            task_type: Task type

        Returns:
            List of parameter sets with performance metrics

        Example:
            >>> history = tuner.get_optimization_history("summarization")
            >>> for entry in history:
            ...     print(entry["parameters"], entry["avg_quality"])
        """
        return self.performance_db.get_all_parameter_sets(task_type)

    def get_statistics(self, task_type: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for task type.

        Args:
            task_type: Task type

        Returns:
            Statistics dictionary

        Example:
            >>> stats = tuner.get_statistics("code_generation")
            >>> print(stats["total_runs"], stats["best_parameters"])
        """
        return self.performance_db.get_optimization_stats(task_type)
