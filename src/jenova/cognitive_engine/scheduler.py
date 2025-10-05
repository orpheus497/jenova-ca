
from datetime import datetime

class CognitiveScheduler:
    """Schedules cognitive functions based on the current context."""

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

    def get_cognitive_tasks(self, turn_count: int, user_input: str, username: str) -> list:
        """
        Determines which cognitive tasks to run based on the current context.
        """
        tasks = []
        scheduler_config = self.config.get('scheduler', {})

        # Get intervals from config
        generate_insight_interval = scheduler_config.get('generate_insight_interval', 5)
        generate_assumption_interval = scheduler_config.get('generate_assumption_interval', 7)
        proactively_verify_assumption_interval = scheduler_config.get('proactively_verify_assumption_interval', 8)
        reflect_interval = scheduler_config.get('reflect_interval', 10)

        # Context-aware adjustments
        if len(self.cortex.get_all_nodes_by_type('assumption', username)) > 5:
            proactively_verify_assumption_interval = 3 # Verify assumptions more frequently if there are many unverified ones

        if self._should_run("generate_insight", turn_count, interval=generate_insight_interval):
            tasks.append(("generate_insight_from_history", {"username": username}))

        if self._should_run("generate_assumption", turn_count, interval=generate_assumption_interval):
            tasks.append(("generate_assumption_from_history", {"username": username}))

        if self._should_run("proactively_verify_assumption", turn_count, interval=proactively_verify_assumption_interval):
            tasks.append(("proactively_verify_assumption", {"username": username}))

        if self._should_run("reflect", turn_count, interval=reflect_interval):
            tasks.append(("reflect", {"user": username}))

        return tasks

    def _should_run(self, task_name: str, turn_count: int, interval: int) -> bool:
        """
        Checks if a task should be run based on the turn count and the last execution time.
        """
        if turn_count % interval == 0:
            self.last_execution_times[task_name] = datetime.now()
            return True
        return False
