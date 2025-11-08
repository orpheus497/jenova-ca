# The JENOVA Cognitive Architecture - Performance Database
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 25: Performance Database for Self-Optimization.

SQLite database for tracking performance metrics across different task types
and parameter configurations, enabling data-driven parameter optimization.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class PerformanceDB:
    """
    SQLite database for performance metrics storage and retrieval.

    Schema:
        - task_runs: Individual task executions with parameters and metrics
        - parameter_sets: Unique parameter combinations with aggregate stats
        - metrics: Detailed metrics per run (response_time, quality, etc.)
        - optimizations: Optimization run history and convergence data

    Example:
        >>> db = PerformanceDB(Path("~/.jenova-ai/performance.db"))
        >>> run_id = db.record_task_run(
        ...     task_type="code_generation",
        ...     parameters={"temperature": 0.7, "max_tokens": 512},
        ...     duration=5.2,
        ...     quality_score=0.85
        ... )
    """

    def __init__(self, db_path: Path):
        """
        Initialize performance database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row  # Enable column access by name
        self._create_schema()

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        cursor = self.connection.cursor()

        # Task runs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                duration REAL NOT NULL,
                quality_score REAL,
                parameters_json TEXT NOT NULL,
                user_feedback TEXT,
                success BOOLEAN NOT NULL DEFAULT 1
            )
        """
        )

        # Parameter sets table (unique combinations)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS parameter_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                avg_quality REAL,
                avg_duration REAL,
                trials_count INTEGER DEFAULT 0,
                success_rate REAL,
                last_used REAL,
                UNIQUE(task_type, parameters_json)
            )
        """
        )

        # Metrics table (detailed metrics per run)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES task_runs(id)
            )
        """
        )

        # Optimizations table (optimization run history)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                best_params_json TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                convergence_score REAL,
                initial_quality REAL,
                final_quality REAL
            )
        """
        )

        # Create indices for faster queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_type ON task_runs(task_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON task_runs(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_param_task ON parameter_sets(task_type)"
        )

        self.connection.commit()

    def record_task_run(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        duration: float,
        quality_score: Optional[float] = None,
        user_feedback: Optional[str] = None,
        success: bool = True,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Record a task execution.

        Args:
            task_type: Type of task (general_qa, code_generation, etc.)
            parameters: Parameter configuration used
            duration: Execution duration in seconds
            quality_score: Quality score 0.0-1.0 (optional)
            user_feedback: User feedback on result (optional)
            success: Whether task completed successfully
            additional_metrics: Additional metrics to record

        Returns:
            run_id: ID of recorded run

        Example:
            >>> run_id = db.record_task_run(
            ...     "summarization",
            ...     {"temperature": 0.3, "max_tokens": 256},
            ...     duration=3.5,
            ...     quality_score=0.92
            ... )
        """
        cursor = self.connection.cursor()
        timestamp = datetime.now().timestamp()
        params_json = json.dumps(parameters, sort_keys=True)

        # Insert task run
        cursor.execute(
            """
            INSERT INTO task_runs
            (task_type, timestamp, duration, quality_score, parameters_json,
             user_feedback, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_type,
                timestamp,
                duration,
                quality_score,
                params_json,
                user_feedback,
                success,
            ),
        )

        run_id = cursor.lastrowid

        # Record additional metrics if provided
        if additional_metrics:
            for metric_name, metric_value in additional_metrics.items():
                cursor.execute(
                    """
                    INSERT INTO metrics (run_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """,
                    (run_id, metric_name, metric_value),
                )

        # Update parameter set statistics
        self._update_parameter_set_stats(task_type, params_json)

        self.connection.commit()
        return run_id

    def get_best_parameters(
        self, task_type: str, metric: str = "quality_score"
    ) -> Optional[Dict[str, Any]]:
        """
        Get parameters with best metric value for task type.

        Args:
            task_type: Task type to query
            metric: Metric to optimize (quality_score, duration, etc.)

        Returns:
            Best parameters dict or None if no data

        Example:
            >>> params = db.get_best_parameters("code_generation")
            >>> # {'temperature': 0.7, 'max_tokens': 512, 'top_p': 0.95}
        """
        cursor = self.connection.cursor()

        if metric == "quality_score":
            # Highest quality score
            cursor.execute(
                """
                SELECT parameters_json, avg_quality
                FROM parameter_sets
                WHERE task_type = ? AND avg_quality IS NOT NULL
                ORDER BY avg_quality DESC
                LIMIT 1
            """,
                (task_type,),
            )
        elif metric == "duration":
            # Shortest duration
            cursor.execute(
                """
                SELECT parameters_json, avg_duration
                FROM parameter_sets
                WHERE task_type = ? AND avg_duration IS NOT NULL
                ORDER BY avg_duration ASC
                LIMIT 1
            """,
                (task_type,),
            )
        else:
            return None

        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def get_parameter_performance_history(
        self, task_type: str, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for specific parameter set.

        Args:
            task_type: Task type
            parameters: Parameter configuration

        Returns:
            List of performance records

        Example:
            >>> history = db.get_parameter_performance_history(
            ...     "general_qa",
            ...     {"temperature": 0.7}
            ... )
        """
        cursor = self.connection.cursor()
        params_json = json.dumps(parameters, sort_keys=True)

        cursor.execute(
            """
            SELECT id, timestamp, duration, quality_score, user_feedback, success
            FROM task_runs
            WHERE task_type = ? AND parameters_json = ?
            ORDER BY timestamp DESC
        """,
            (task_type, params_json),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "run_id": row[0],
                    "timestamp": row[1],
                    "duration": row[2],
                    "quality_score": row[3],
                    "user_feedback": row[4],
                    "success": bool(row[5]),
                }
            )

        return results

    def get_optimization_stats(self, task_type: str) -> Dict[str, Any]:
        """
        Get optimization statistics for task type.

        Args:
            task_type: Task type

        Returns:
            Dict with stats (total_runs, avg_quality, best_params, etc.)

        Example:
            >>> stats = db.get_optimization_stats("code_generation")
            >>> print(stats["total_runs"], stats["avg_quality"])
        """
        cursor = self.connection.cursor()

        # Total runs
        cursor.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_type = ?", (task_type,)
        )
        total_runs = cursor.fetchone()[0]

        # Average quality
        cursor.execute(
            """
            SELECT AVG(quality_score)
            FROM task_runs
            WHERE task_type = ? AND quality_score IS NOT NULL
        """,
            (task_type,),
        )
        avg_quality = cursor.fetchone()[0]

        # Success rate
        cursor.execute(
            """
            SELECT
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
            FROM task_runs
            WHERE task_type = ?
        """,
            (task_type,),
        )
        success_rate = cursor.fetchone()[0]

        # Get best parameters
        best_params = self.get_best_parameters(task_type)

        # Get recent optimization runs
        cursor.execute(
            """
            SELECT COUNT(*) FROM optimizations WHERE task_type = ?
        """,
            (task_type,),
        )
        optimization_runs = cursor.fetchone()[0]

        return {
            "total_runs": total_runs,
            "avg_quality": avg_quality,
            "success_rate": success_rate,
            "best_parameters": best_params,
            "optimization_runs": optimization_runs,
        }

    def record_optimization(
        self,
        task_type: str,
        best_params: Dict[str, Any],
        iterations: int,
        convergence_score: float,
        initial_quality: float,
        final_quality: float,
    ) -> int:
        """
        Record an optimization run.

        Args:
            task_type: Task type optimized
            best_params: Best parameters found
            iterations: Number of optimization iterations
            convergence_score: Convergence metric
            initial_quality: Quality before optimization
            final_quality: Quality after optimization

        Returns:
            optimization_id: ID of recorded optimization

        Example:
            >>> opt_id = db.record_optimization(
            ...     "summarization",
            ...     {"temperature": 0.3, "max_tokens": 256},
            ...     iterations=50,
            ...     convergence_score=0.95,
            ...     initial_quality=0.7,
            ...     final_quality=0.89
            ... )
        """
        cursor = self.connection.cursor()
        timestamp = datetime.now().timestamp()
        params_json = json.dumps(best_params, sort_keys=True)

        cursor.execute(
            """
            INSERT INTO optimizations
            (task_type, timestamp, best_params_json, iterations,
             convergence_score, initial_quality, final_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_type,
                timestamp,
                params_json,
                iterations,
                convergence_score,
                initial_quality,
                final_quality,
            ),
        )

        optimization_id = cursor.lastrowid
        self.connection.commit()
        return optimization_id

    def _update_parameter_set_stats(self, task_type: str, params_json: str) -> None:
        """
        Update aggregate statistics for parameter set.

        Args:
            task_type: Task type
            params_json: JSON-encoded parameters
        """
        cursor = self.connection.cursor()
        timestamp = datetime.now().timestamp()

        # Calculate stats from all runs with these parameters
        cursor.execute(
            """
            SELECT
                AVG(quality_score) as avg_quality,
                AVG(duration) as avg_duration,
                COUNT(*) as trials_count,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
            FROM task_runs
            WHERE task_type = ? AND parameters_json = ?
        """,
            (task_type, params_json),
        )

        row = cursor.fetchone()
        avg_quality, avg_duration, trials_count, success_rate = row

        # Insert or update parameter set
        cursor.execute(
            """
            INSERT INTO parameter_sets
            (task_type, parameters_json, avg_quality, avg_duration,
             trials_count, success_rate, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_type, parameters_json) DO UPDATE SET
                avg_quality = excluded.avg_quality,
                avg_duration = excluded.avg_duration,
                trials_count = excluded.trials_count,
                success_rate = excluded.success_rate,
                last_used = excluded.last_used
        """,
            (
                task_type,
                params_json,
                avg_quality,
                avg_duration,
                trials_count,
                success_rate,
                timestamp,
            ),
        )

    def get_all_parameter_sets(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Get all parameter sets for task type with statistics.

        Args:
            task_type: Task type

        Returns:
            List of parameter sets with stats

        Example:
            >>> sets = db.get_all_parameter_sets("code_generation")
            >>> for param_set in sets:
            ...     print(param_set["parameters"], param_set["avg_quality"])
        """
        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT parameters_json, avg_quality, avg_duration,
                   trials_count, success_rate
            FROM parameter_sets
            WHERE task_type = ?
            ORDER BY avg_quality DESC NULLS LAST
        """,
            (task_type,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "parameters": json.loads(row[0]),
                    "avg_quality": row[1],
                    "avg_duration": row[2],
                    "trials_count": row[3],
                    "success_rate": row[4],
                }
            )

        return results

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
