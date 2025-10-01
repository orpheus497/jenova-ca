import random

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

        # Strategy 1: Focus on a highly central node (a well-developed topic)
        sorted_nodes = sorted(user_nodes, key=lambda n: n['metadata']['centrality'], reverse=True)
        most_central_node = sorted_nodes[0]

        # Strategy 2: Focus on a low-centrality node (an underdeveloped topic)
        least_central_node = sorted_nodes[-1]

        # Randomly choose a strategy
        if random.random() < 0.7: # 70% chance to focus on a central topic
            prompt = f'''You have a well-developed insight on the topic of '{most_central_node.get("topic", "a certain topic")}'. Based on the content "{most_central_node["content"]}", ask a follow-up question that encourages the user to explore a new dimension of this topic.

Follow-up question:'''
        else: # 30% chance to focus on an underdeveloped topic
            prompt = f'''You have a brief insight on the topic of '{least_central_node.get("topic", "a certain topic")}'. Based on the content "{least_central_node["content"]}", ask a question that encourages the user to elaborate and provide more details.

Elaboration question:'''

        with self.ui_logger.thinking_process("Considering a thought..."):
            suggestion = self.llm.generate(prompt, temperature=0.7)
        
        return suggestion
