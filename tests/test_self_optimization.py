# The JENOVA Cognitive Architecture - Self-Optimization Tests
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 25: Tests for Self-Optimization Engine.

Tests performance database, Bayesian optimizer, self-tuner, and task classifier
with comprehensive coverage of optimization workflows.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from jenova.optimization import (
    SelfTuner,
    PerformanceDB,
    BayesianOptimizer,
    TaskClassifier,
)


class TestPerformanceDB:
    """Test suite for PerformanceDB."""

    @pytest.fixture
    def temp_db(self):
        """Fixture providing temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        db = PerformanceDB(db_path)
        yield db
        db.close()
        db_path.unlink()

    def test_initialization(self, temp_db):
        """Test database initialization and schema creation."""
        # Should create tables without error
        assert temp_db.connection is not None

    def test_record_task_run(self, temp_db):
        """Test recording task execution."""
        run_id = temp_db.record_task_run(
            task_type="code_generation",
            parameters={"temperature": 0.3, "max_tokens": 512},
            duration=5.2,
            quality_score=0.89,
        )

        assert run_id > 0

    def test_get_best_parameters(self, temp_db):
        """Test retrieving best parameters."""
        # Record multiple runs
        temp_db.record_task_run(
            "test_task",
            {"temperature": 0.5},
            duration=1.0,
            quality_score=0.7,
        )
        temp_db.record_task_run(
            "test_task",
            {"temperature": 0.7},
            duration=1.0,
            quality_score=0.9,  # Best
        )
        temp_db.record_task_run(
            "test_task",
            {"temperature": 0.3},
            duration=1.0,
            quality_score=0.8,
        )

        best = temp_db.get_best_parameters("test_task")
        assert best is not None
        assert best["temperature"] == 0.7

    def test_parameter_performance_history(self, temp_db):
        """Test retrieving performance history for parameters."""
        params = {"temperature": 0.5}

        # Record multiple runs
        for i in range(3):
            temp_db.record_task_run(
                "test_task",
                params,
                duration=1.0 + i * 0.1,
                quality_score=0.8 + i * 0.05,
            )

        history = temp_db.get_parameter_performance_history("test_task", params)
        assert len(history) == 3

    def test_optimization_stats(self, temp_db):
        """Test getting optimization statistics."""
        # Record some runs
        for i in range(5):
            temp_db.record_task_run(
                "test_task",
                {"temperature": 0.5 + i * 0.1},
                duration=1.0,
                quality_score=0.8,
                success=True,
            )

        stats = temp_db.get_optimization_stats("test_task")
        assert stats["total_runs"] == 5
        assert stats["avg_quality"] == 0.8
        assert stats["success_rate"] == 1.0

    def test_record_optimization(self, temp_db):
        """Test recording optimization run."""
        opt_id = temp_db.record_optimization(
            task_type="test_task",
            best_params={"temperature": 0.7},
            iterations=30,
            convergence_score=0.95,
            initial_quality=0.6,
            final_quality=0.9,
        )

        assert opt_id > 0

    def test_additional_metrics(self, temp_db):
        """Test recording additional metrics."""
        run_id = temp_db.record_task_run(
            "test_task",
            {"temperature": 0.5},
            duration=1.0,
            quality_score=0.8,
            additional_metrics={"tokens_generated": 512, "response_time": 2.3},
        )

        # Additional metrics should be stored
        assert run_id > 0

    def test_parameter_set_aggregation(self, temp_db):
        """Test parameter set statistics aggregation."""
        params = {"temperature": 0.5}

        # Record multiple runs with same parameters
        for _ in range(3):
            temp_db.record_task_run(
                "test_task",
                params,
                duration=1.0,
                quality_score=0.8,
            )

        param_sets = temp_db.get_all_parameter_sets("test_task")
        assert len(param_sets) > 0
        assert param_sets[0]["trials_count"] == 3

    def test_context_manager(self):
        """Test database context manager."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        with PerformanceDB(db_path) as db:
            db.record_task_run(
                "test",
                {"temp": 0.5},
                1.0,
                0.8,
            )

        db_path.unlink()

    def test_empty_database(self, temp_db):
        """Test queries on empty database."""
        best = temp_db.get_best_parameters("nonexistent_task")
        assert best is None

        stats = temp_db.get_optimization_stats("nonexistent_task")
        assert stats["total_runs"] == 0


class TestBayesianOptimizer:
    """Test suite for BayesianOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        def objective(params):
            return params["x"] ** 2

        optimizer = BayesianOptimizer(
            parameter_space={"x": (-5.0, 5.0)},
            objective_function=objective,
        )

        assert optimizer.n_dims == 1
        assert len(optimizer.bounds) == 1

    def test_random_sampling(self):
        """Test random parameter sampling."""
        def objective(params):
            return 1.0

        optimizer = BayesianOptimizer(
            parameter_space={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            objective_function=objective,
        )

        sample = optimizer._sample_random_params()
        assert len(sample) == 2
        assert 0.0 <= sample[0] <= 1.0
        assert 0.0 <= sample[1] <= 1.0

    def test_optimization_simple(self):
        """Test optimization on simple function."""
        # Minimize x^2 + y^2 (optimum at x=0, y=0)
        def objective(params):
            return -(params["x"] ** 2 + params["y"] ** 2)

        optimizer = BayesianOptimizer(
            parameter_space={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            objective_function=objective,
        )

        best = optimizer.optimize(n_iterations=20)

        # Should find parameters close to 0
        assert abs(best["x"]) < 1.0
        assert abs(best["y"]) < 1.0

    def test_acquisition_function(self):
        """Test Expected Improvement acquisition function."""
        def objective(params):
            return params["x"]

        optimizer = BayesianOptimizer(
            parameter_space={"x": (0.0, 1.0)},
            objective_function=objective,
        )

        # Add some observations
        optimizer._evaluate_and_record(np.array([0.3]))
        optimizer._evaluate_and_record(np.array([0.7]))

        # Acquisition function should return value
        ei = optimizer._acquisition_function(np.array([0.5]))
        assert ei >= 0.0

    def test_gaussian_process_prediction(self):
        """Test GP mean and std prediction."""
        def objective(params):
            return params["x"] ** 2

        optimizer = BayesianOptimizer(
            parameter_space={"x": (-1.0, 1.0)},
            objective_function=objective,
        )

        # Add observations
        optimizer._evaluate_and_record(np.array([0.0]))
        optimizer._evaluate_and_record(np.array([1.0]))

        # Predict at new point
        mu, sigma = optimizer._predict_gp(np.array([0.5]))
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma >= 0.0

    def test_convergence_detection(self):
        """Test convergence detection."""
        def objective(params):
            return 1.0  # Constant function

        optimizer = BayesianOptimizer(
            parameter_space={"x": (0.0, 1.0)},
            objective_function=objective,
        )

        # Add constant observations
        for _ in range(10):
            optimizer._evaluate_and_record(optimizer._sample_random_params())

        # Should detect convergence
        assert optimizer._check_convergence(tolerance=0.01, patience=5)

    def test_convergence_metrics(self):
        """Test convergence metrics calculation."""
        def objective(params):
            return params["x"]

        optimizer = BayesianOptimizer(
            parameter_space={"x": (0.0, 1.0)},
            objective_function=objective,
        )

        optimizer.optimize(n_iterations=10)
        metrics = optimizer.get_convergence_metrics()

        assert "best_value" in metrics
        assert "best_params" in metrics
        assert "iterations" in metrics
        assert metrics["iterations"] == 10

    def test_get_observations(self):
        """Test getting all observations."""
        def objective(params):
            return params["x"]

        optimizer = BayesianOptimizer(
            parameter_space={"x": (0.0, 1.0)},
            objective_function=objective,
        )

        optimizer.optimize(n_iterations=5)
        observations = optimizer.get_observations()

        assert len(observations) == 5
        assert all(isinstance(obs[0], dict) for obs in observations)
        assert all(isinstance(obs[1], float) for obs in observations)


class TestTaskClassifier:
    """Test suite for TaskClassifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = TaskClassifier()
        assert len(classifier.task_keywords) > 0

    def test_code_generation_classification(self):
        """Test code generation task detection."""
        classifier = TaskClassifier()

        queries = [
            "Write a Python function to sort a list",
            "Implement bubble sort in JavaScript",
            "Create a class for data validation",
        ]

        for query in queries:
            task = classifier.classify_task(query)
            assert task == "code_generation"

    def test_summarization_classification(self):
        """Test summarization task detection."""
        classifier = TaskClassifier()

        queries = [
            "Summarize this article",
            "Give me a brief overview",
            "TLDR of the document",
        ]

        for query in queries:
            task = classifier.classify_task(query)
            assert task == "summarization"

    def test_analysis_classification(self):
        """Test analysis task detection."""
        classifier = TaskClassifier()

        queries = [
            "Analyze the performance of this code",
            "Compare these two algorithms",
            "Evaluate the pros and cons",
        ]

        for query in queries:
            task = classifier.classify_task(query)
            assert task == "analysis"

    def test_creative_writing_classification(self):
        """Test creative writing task detection."""
        classifier = TaskClassifier()

        queries = [
            "Write a story about a dragon",
            "Create a poem about nature",
            "Imagine a futuristic world",
        ]

        for query in queries:
            task = classifier.classify_task(query)
            assert task == "creative_writing"

    def test_general_qa_default(self):
        """Test general QA as default."""
        classifier = TaskClassifier()

        query = "What is the capital of France?"
        task = classifier.classify_task(query)
        assert task == "general_qa"

    def test_confidence_scoring(self):
        """Test classification with confidence."""
        classifier = TaskClassifier()

        task, confidence = classifier.classify_with_confidence(
            "Write a Python sorting function"
        )

        assert task == "code_generation"
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be confident

    def test_task_characteristics(self):
        """Test getting task characteristics."""
        classifier = TaskClassifier()

        chars = classifier.get_task_characteristics("code_generation")
        assert "temperature_range" in chars
        assert "max_tokens_range" in chars
        assert "creativity" in chars
        assert chars["creativity"] == "low"


class TestSelfTuner:
    """Test suite for SelfTuner."""

    @pytest.fixture
    def tuner(self):
        """Fixture providing SelfTuner instance."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        db = PerformanceDB(db_path)
        config = {
            "model": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 512,
            }
        }

        tuner = SelfTuner(performance_db=db, config=config)

        yield tuner

        db.close()
        db_path.unlink()

    def test_initialization(self, tuner):
        """Test tuner initialization."""
        assert tuner.parameter_space is not None
        assert len(tuner.active_parameters) > 0

    def test_record_performance(self, tuner):
        """Test recording performance."""
        tuner.record_performance(
            "test_task",
            {"temperature": 0.5},
            duration=1.0,
            quality_score=0.8,
        )

        # Should be recorded in database
        stats = tuner.performance_db.get_optimization_stats("test_task")
        assert stats["total_runs"] == 1

    def test_get_optimal_parameters(self, tuner):
        """Test getting optimal parameters."""
        # Record some performance data
        tuner.record_performance(
            "code_generation",
            {"temperature": 0.3, "max_tokens": 512},
            duration=1.0,
            quality_score=0.9,
        )

        optimal = tuner.get_optimal_parameters("code_generation")
        assert "temperature" in optimal

    def test_a_b_testing(self, tuner):
        """Test A/B testing of parameter sets."""
        results = tuner.a_b_test(
            "test_task",
            {"temperature": 0.3},
            {"temperature": 0.7},
            n_trials=5,
        )

        assert "winner" in results
        assert results["winner"] in ["A", "B"]
        assert "confidence" in results

    def test_auto_adjust(self, tuner):
        """Test automatic parameter adjustment."""
        current = {"temperature": 0.8, "max_tokens": 512}

        adjusted = tuner.auto_adjust("test_task", "too_creative", current)

        # Temperature should decrease
        assert adjusted["temperature"] < current["temperature"]

    def test_optimization_history(self, tuner):
        """Test getting optimization history."""
        # Record some runs
        for i in range(3):
            tuner.record_performance(
                "test_task",
                {"temperature": 0.5 + i * 0.1},
                duration=1.0,
                quality_score=0.8,
            )

        history = tuner.get_optimization_history("test_task")
        assert len(history) >= 3

    def test_optimize_parameters(self, tuner):
        """Test parameter optimization."""
        best = tuner.optimize_parameters("test_task", iterations=10)

        assert "temperature" in best
        assert "max_tokens" in best

    def test_apply_optimal_parameters(self, tuner):
        """Test applying optimal parameters."""
        # Record best parameters
        tuner.record_performance(
            "test_task",
            {"temperature": 0.6},
            duration=1.0,
            quality_score=0.95,
        )

        tuner.apply_optimal_parameters("test_task")

        # Active parameters should be updated
        assert "test_task" in tuner.active_parameters


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        db = PerformanceDB(db_path)
        config = {"model": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 512}}
        tuner = SelfTuner(performance_db=db, config=config)

        # Classify task
        classifier = TaskClassifier()
        task_type = classifier.classify_task("Write a sorting function")

        # Optimize parameters
        best = tuner.optimize_parameters(task_type, iterations=10)

        # Apply optimal parameters
        tuner.apply_optimal_parameters(task_type)

        # Get statistics
        stats = tuner.get_statistics(task_type)

        assert stats["total_runs"] >= 10
        assert best is not None

        db.close()
        db_path.unlink()

    def test_feedback_loop(self):
        """Test feedback-driven optimization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        db = PerformanceDB(db_path)
        config = {"model": {"temperature": 0.7}}
        tuner = SelfTuner(performance_db=db, config=config)

        # Simulate user feedback loop
        current_params = {"temperature": 0.8}

        # User says "too creative"
        adjusted = tuner.auto_adjust("test_task", "too_creative", current_params)

        # Record performance
        tuner.record_performance(
            "test_task",
            adjusted,
            duration=1.0,
            quality_score=0.85,
            user_feedback="good",
        )

        db.close()
        db_path.unlink()
