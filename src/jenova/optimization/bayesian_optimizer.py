# The JENOVA Cognitive Architecture - Bayesian Optimizer
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 25: Bayesian Optimization for hyperparameter search.

Uses Gaussian Process regression to model parameter-performance relationship
and Expected Improvement acquisition function to intelligently select next
parameters to test, converging faster than grid or random search.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
from scipy.stats import norm
from scipy.optimize import minimize


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter search.

    Uses Gaussian Process to model the objective function and Expected
    Improvement for acquisition. Efficiently finds optimal parameters
    with fewer evaluations than grid or random search.

    Example:
        >>> def objective(params):
        ...     # Run task with params, return quality score
        ...     return quality_score
        >>>
        >>> optimizer = BayesianOptimizer(
        ...     parameter_space={
        ...         'temperature': (0.0, 1.0),
        ...         'max_tokens': (128, 2048)
        ...     },
        ...     objective_function=objective
        ... )
        >>> best_params = optimizer.optimize(n_iterations=50)
    """

    def __init__(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: Callable[[Dict[str, Any]], float],
        exploration_weight: float = 0.1,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            parameter_space: Dict mapping param names to (min, max) bounds
            objective_function: Function to evaluate (returns score to maximize)
            exploration_weight: Weight for exploration vs exploitation (0.0-1.0)
        """
        self.parameter_space = parameter_space
        self.objective = objective_function
        self.exploration_weight = exploration_weight

        # Parameter names and dimensions
        self.param_names = list(parameter_space.keys())
        self.n_dims = len(self.param_names)

        # Bounds as numpy array for scipy
        self.bounds = np.array([parameter_space[name] for name in self.param_names])

        # Observations: (X, y) where X is params and y is objective value
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []

        # Best observation so far
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = -np.inf

    def optimize(self, n_iterations: int = 50) -> Dict[str, Any]:
        """
        Run Bayesian optimization for n iterations.

        Args:
            n_iterations: Number of optimization iterations

        Returns:
            Best parameters found

        Example:
            >>> best = optimizer.optimize(n_iterations=30)
            >>> print(best)  # {'temperature': 0.7, 'max_tokens': 512}
        """
        # Initialize with random samples
        n_initial = min(5, n_iterations // 10)  # 10% for random exploration
        for _ in range(n_initial):
            params = self._sample_random_params()
            self._evaluate_and_record(params)

        # Bayesian optimization loop
        for iteration in range(n_initial, n_iterations):
            # Select next point using acquisition function
            next_params = self._select_next_point()

            # Evaluate objective
            self._evaluate_and_record(next_params)

            # Early stopping if converged
            if self._check_convergence():
                break

        return self.best_params

    def _sample_random_params(self) -> np.ndarray:
        """
        Sample random parameters from parameter space.

        Returns:
            Random parameter vector
        """
        params = np.zeros(self.n_dims)
        for i, (low, high) in enumerate(self.bounds):
            params[i] = np.random.uniform(low, high)
        return params

    def _evaluate_and_record(self, params_array: np.ndarray) -> float:
        """
        Evaluate objective function and record observation.

        Args:
            params_array: Parameter vector

        Returns:
            Objective value
        """
        # Convert to dict for objective function
        params_dict = {
            self.param_names[i]: params_array[i] for i in range(self.n_dims)
        }

        # Evaluate objective
        value = self.objective(params_dict)

        # Record observation
        self.X_observed.append(params_array)
        self.y_observed.append(value)

        # Update best
        if value > self.best_value:
            self.best_value = value
            self.best_params = params_dict.copy()

        return value

    def _select_next_point(self) -> np.ndarray:
        """
        Select next point to evaluate using Expected Improvement.

        Returns:
            Next parameter vector to try
        """
        # Define negative acquisition function for minimization
        def neg_acquisition(x):
            return -self._acquisition_function(x)

        # Random restart optimization to avoid local optima
        best_acq = -np.inf
        best_x = None

        n_restarts = 5
        for _ in range(n_restarts):
            x0 = self._sample_random_params()

            result = minimize(
                neg_acquisition, x0, bounds=self.bounds, method="L-BFGS-B"
            )

            if result.success and -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x

        return best_x if best_x is not None else self._sample_random_params()

    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Expected Improvement acquisition function.

        Args:
            x: Parameter vector

        Returns:
            Expected improvement value
        """
        if not self.y_observed:
            return 0.0

        # Predict mean and std using simple GP
        mu, sigma = self._predict_gp(x)

        # Current best
        f_best = max(self.y_observed)

        # Expected Improvement
        if sigma == 0:
            return 0.0

        # Standardized improvement
        z = (mu - f_best - self.exploration_weight) / sigma

        # EI = improvement * probability(improvement)
        ei = (mu - f_best - self.exploration_weight) * norm.cdf(z) + sigma * norm.pdf(
            z
        )

        return ei

    def _predict_gp(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict mean and std using Gaussian Process.

        Simplified GP using RBF kernel and basic hyperparameters.
        In production, use scikit-learn's GaussianProcessRegressor.

        Args:
            x: Parameter vector

        Returns:
            (mean, std) prediction
        """
        if not self.X_observed:
            return 0.0, 1.0

        X = np.array(self.X_observed)
        y = np.array(self.y_observed)

        # RBF kernel with length scale
        length_scale = 0.5
        kernel_matrix = self._rbf_kernel(X, X, length_scale)

        # Add noise
        noise = 1e-6
        K = kernel_matrix + noise * np.eye(len(X))

        # Kernel between x and observed points
        k = self._rbf_kernel(np.array([x]), X, length_scale).flatten()

        # GP mean prediction
        try:
            K_inv = np.linalg.inv(K)
            mu = k @ K_inv @ y
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            mu = np.mean(y)

        # GP std prediction
        k_self = self._rbf_kernel(np.array([x]), np.array([x]), length_scale)[0, 0]
        try:
            sigma_squared = k_self - k @ K_inv @ k
            sigma = np.sqrt(max(0, sigma_squared))
        except:
            sigma = 1.0

        return float(mu), float(sigma)

    def _rbf_kernel(
        self, X1: np.ndarray, X2: np.ndarray, length_scale: float
    ) -> np.ndarray:
        """
        RBF (Radial Basis Function) kernel.

        Args:
            X1: First set of points (n1 x d)
            X2: Second set of points (n2 x d)
            length_scale: Kernel length scale

        Returns:
            Kernel matrix (n1 x n2)
        """
        # Pairwise squared distances
        dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(
            X2**2, axis=1
        ) - 2 * X1 @ X2.T

        # RBF kernel
        return np.exp(-0.5 * dists / (length_scale**2))

    def _check_convergence(self, tolerance: float = 1e-3, patience: int = 5) -> bool:
        """
        Check if optimization has converged.

        Args:
            tolerance: Improvement tolerance
            patience: Number of iterations without improvement

        Returns:
            True if converged
        """
        if len(self.y_observed) < patience + 1:
            return False

        # Check last `patience` iterations
        recent_values = self.y_observed[-patience:]
        improvement = max(recent_values) - min(recent_values)

        return improvement < tolerance

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """
        Get convergence metrics for analysis.

        Returns:
            Dict with convergence stats

        Example:
            >>> metrics = optimizer.get_convergence_metrics()
            >>> print(metrics["best_value"], metrics["iterations"])
        """
        if not self.y_observed:
            return {}

        return {
            "best_value": self.best_value,
            "best_params": self.best_params,
            "iterations": len(self.y_observed),
            "mean_value": np.mean(self.y_observed),
            "std_value": np.std(self.y_observed),
            "improvement": (
                self.best_value - self.y_observed[0] if len(self.y_observed) > 0 else 0
            ),
        }

    def get_observations(self) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get all observations (params, value) pairs.

        Returns:
            List of (parameters_dict, objective_value) tuples

        Example:
            >>> for params, value in optimizer.get_observations():
            ...     print(f"{params} -> {value}")
        """
        observations = []
        for x, y in zip(self.X_observed, self.y_observed):
            params = {self.param_names[i]: x[i] for i in range(self.n_dims)}
            observations.append((params, y))
        return observations
