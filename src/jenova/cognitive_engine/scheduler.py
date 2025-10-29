# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta


class CognitiveScheduler:
    """Schedules cognitive functions based on dynamic, context-aware triggers."""

    def __init__(self, config, cortex, insight_manager):
        self.config = config
        self.cortex = cortex
        self.insight_manager = insight_manager
        self.last_execution_times = {
            "generate_insight": None,
            "generate_assumption": None,
            "proactively_verify_assumption": None,
            "reflect": None,
        }
        self.conversation_intensity = 0

    def _update_conversation_intensity(self, user_input: str):
        """Updates a simple metric for conversation intensity."""
        # Longer user inputs suggest more intense/detailed conversation
        if len(user_input) > 100:
            self.conversation_intensity += 2
        elif len(user_input) > 20:
            self.conversation_intensity += 1
        else:
            self.conversation_intensity = max(
                0, self.conversation_intensity - 1)

    def get_cognitive_tasks(self, turn_count: int, user_input: str, username: str) -> list:
        """
        Determines which cognitive tasks to run based on the current context.
        """
        self._update_conversation_intensity(user_input)
        tasks = []
        scheduler_config = self.config.get('scheduler', {})

        # --- Base Intervals ---
        generate_insight_interval = scheduler_config.get(
            'generate_insight_interval', 5)
        generate_assumption_interval = scheduler_config.get(
            'generate_assumption_interval', 7)
        proactively_verify_assumption_interval = scheduler_config.get(
            'proactively_verify_assumption_interval', 8)
        reflect_interval = scheduler_config.get('reflect_interval', 10)

        # --- Dynamic Adjustments ---
        # Generate insights more frequently during intense conversations
        if self.conversation_intensity > 5:
            generate_insight_interval = max(2, generate_insight_interval - 2)

        # Verify assumptions more often if many are unverified
        unverified_assumptions = self.cortex.get_all_nodes_by_type(
            'assumption', username)
        if len([a for a in unverified_assumptions if a.metadata.get('status') == 'unverified']) > 3:
            proactively_verify_assumption_interval = max(
                3, proactively_verify_assumption_interval - 3)

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

        return tasks

    def _should_run(self, task_name: str, turn_count: int, interval: int) -> bool:
        """
        Checks if a task should be run based on the turn count and last execution time.
        Ensures a task doesn't run twice in rapid succession even if turn count matches.
        """
        now = datetime.now()
        last_run = self.last_execution_times.get(task_name)

        if turn_count % interval == 0:
            if last_run is None or (now - last_run) > timedelta(seconds=10):
                self.last_execution_times[task_name] = now
                return True
        return False
