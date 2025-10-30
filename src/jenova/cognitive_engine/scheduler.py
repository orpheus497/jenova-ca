# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 5: Enhanced Cognitive Scheduler
Improvements:
- Better error handling
- Logging integration
- More robust task scheduling
- Configuration validation
"""

from datetime import datetime, timedelta


class CognitiveScheduler:
    """Schedules cognitive functions based on dynamic, context-aware triggers."""

    def __init__(self, config, cortex, insight_manager, file_logger=None):
        self.config = config
        self.cortex = cortex
        self.insight_manager = insight_manager
        self.file_logger = file_logger
        self.last_execution_times = {
            "generate_insight": None,
            "generate_assumption": None,
            "proactively_verify_assumption": None,
            "reflect": None,
        }
        self.conversation_intensity = 0

        # Load scheduler configuration with defaults
        scheduler_config = self.config.get('scheduler', {})
        self.base_intervals = {
            'generate_insight': scheduler_config.get('generate_insight_interval', 5),
            'generate_assumption': scheduler_config.get('generate_assumption_interval', 7),
            'proactively_verify_assumption': scheduler_config.get('proactively_verify_assumption_interval', 8),
            'reflect': scheduler_config.get('reflect_interval', 10),
        }

        if self.file_logger:
            self.file_logger.log_info(
                f"Cognitive Scheduler initialized with intervals: {self.base_intervals}")

    def _update_conversation_intensity(self, user_input: str):
        """Updates a simple metric for conversation intensity."""
        try:
            # Longer user inputs suggest more intense/detailed conversation
            if len(user_input) > 100:
                self.conversation_intensity += 2
            elif len(user_input) > 20:
                self.conversation_intensity += 1
            else:
                self.conversation_intensity = max(
                    0, self.conversation_intensity - 1)
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error updating conversation intensity: {e}")

    def get_cognitive_tasks(self, turn_count: int, user_input: str, username: str) -> list:
        """
        Determines which cognitive tasks to run based on the current context.

        Returns:
            list: List of tuples (task_name, kwargs) for tasks to execute
        """
        try:
            self._update_conversation_intensity(user_input)
            tasks = []

            # --- Base Intervals ---
            generate_insight_interval = self.base_intervals['generate_insight']
            generate_assumption_interval = self.base_intervals['generate_assumption']
            proactively_verify_assumption_interval = self.base_intervals['proactively_verify_assumption']
            reflect_interval = self.base_intervals['reflect']

            # --- Dynamic Adjustments ---
            # Generate insights more frequently during intense conversations
            if self.conversation_intensity > 5:
                generate_insight_interval = max(2, generate_insight_interval - 2)

            # Verify assumptions more often if many are unverified
            try:
                unverified_assumptions = self.cortex.get_all_nodes_by_type(
                    'assumption', username)
                unverified_count = len([a for a in unverified_assumptions
                                       if a.metadata.get('status') == 'unverified'])
                if unverified_count > 3:
                    proactively_verify_assumption_interval = max(
                        3, proactively_verify_assumption_interval - 3)
            except Exception as e:
                if self.file_logger:
                    self.file_logger.log_error(
                        f"Error checking unverified assumptions: {e}")

            # --- Task Scheduling ---
            if self._should_run("generate_insight", turn_count, interval=generate_insight_interval):
                tasks.append(
                    ("generate_insight_from_history", {"username": username}))
                self.conversation_intensity = 0  # Reset intensity after generating an insight

            if self._should_run("generate_assumption", turn_count, interval=generate_assumption_interval):
                tasks.append(
                    ("generate_assumption_from_history", {"username": username}))

            if self._should_run("proactively_verify_assumption", turn_count, interval=proactively_verify_assumption_interval):
                tasks.append(
                    ("proactively_verify_assumption", {"username": username}))

            if self._should_run("reflect", turn_count, interval=reflect_interval):
                tasks.append(("reflect_on_insights", {"username": username}))

            if tasks and self.file_logger:
                task_names = [name for name, _ in tasks]
                self.file_logger.log_info(
                    f"Scheduled cognitive tasks for turn {turn_count}: {task_names}")

            return tasks

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error in get_cognitive_tasks: {e}")
            return []  # Return empty list on error to prevent cascade failures

    def _should_run(self, task_name: str, turn_count: int, interval: int) -> bool:
        """
        Checks if a task should be run based on the turn count and last execution time.
        Ensures a task doesn't run twice in rapid succession even if turn count matches.

        Args:
            task_name: Name of the task
            turn_count: Current turn count
            interval: Interval in turns for task execution

        Returns:
            bool: True if task should run, False otherwise
        """
        try:
            now = datetime.now()
            last_run = self.last_execution_times.get(task_name)

            if turn_count % interval == 0:
                # Ensure at least 10 seconds have passed since last run
                if last_run is None or (now - last_run) > timedelta(seconds=10):
                    self.last_execution_times[task_name] = now
                    return True
            return False

        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error in _should_run for task '{task_name}': {e}")
            return False

    def reset_task_timer(self, task_name: str):
        """
        Manually reset the last execution time for a specific task.
        Useful for testing or manual task triggering.

        Args:
            task_name: Name of the task to reset
        """
        if task_name in self.last_execution_times:
            self.last_execution_times[task_name] = None
            if self.file_logger:
                self.file_logger.log_info(f"Reset timer for task: {task_name}")

    def get_status(self) -> dict:
        """
        Get current scheduler status including last execution times and intensity.

        Returns:
            dict: Status information
        """
        return {
            "conversation_intensity": self.conversation_intensity,
            "last_execution_times": {
                task: (time.isoformat() if time else "Never")
                for task, time in self.last_execution_times.items()
            },
            "base_intervals": self.base_intervals
        }
