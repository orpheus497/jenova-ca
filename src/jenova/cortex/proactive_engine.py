
class ProactiveEngine:
    """
    The Proactive Engine analyzes the cognitive graph to identify areas of interest
    and generates proactive suggestions or questions for the user.
    """
    def __init__(self, cortex, llm, ui_logger):
        self.cortex = cortex
        self.llm = llm
        self.ui_logger = ui_logger

    def get_suggestion(self, user: str) -> str | None:
        """
        Analyzes the user's cognitive graph to find an interesting, underdeveloped,
        or highly-connected area and generates a proactive suggestion.
        """
        user_nodes = self.cortex.get_all_nodes_by_type('insight', user=user)
        if len(user_nodes) < 3:
            return None

        # Strategy: Find a cluster of related insights and ask a question that connects them.
        # This is a more advanced strategy than just looking at the most recent insight.
        # A real implementation would use graph analysis to find clusters.
        # For now, we'll simulate this by taking a few recent insights.
        recent_insights = sorted(user_nodes, key=lambda n: n['timestamp'], reverse=True)[:3]
        insight_contents = "\n".join([f"- {i['content']}" for i in recent_insights])

        prompt = f'''Based on the following related insights about the user, formulate a single, concise, and engaging question that connects these ideas or explores a potential underlying theme.

Insights:
{insight_contents}

Connecting question:'''

        with self.ui_logger.thinking_process("Considering a thought..."):
            suggestion = self.llm.generate(prompt, temperature=0.7)
        
        return suggestion
