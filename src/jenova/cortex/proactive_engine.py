##Script function and purpose: Proactive Engine for The JENOVA Cognitive Architecture
##This module generates proactive suggestions by analyzing the cognitive graph state

from datetime import datetime

##Class purpose: Analyzes cognitive graph to generate proactive suggestions for users
class ProactiveEngine:
    """
    The Proactive Engine analyzes the cognitive graph to identify areas of interest
    and generates proactive suggestions or questions for the user.
    """
    ##Function purpose: Initialize proactive engine with cortex, LLM, and logger
    def __init__(self, cortex, llm, ui_logger):
        self.cortex = cortex
        self.llm = llm
        self.ui_logger = ui_logger
        self.recent_suggestions = [] # To avoid repetition

    ##Function purpose: Generate proactive suggestion based on current cognitive graph state
    def get_suggestion(self, username: str, history: list) -> str | None:
        """Generates a proactive suggestion for the user based on current cognitive state."""
        # Get unverified assumptions
        unverified_assumptions = self.cortex.get_all_nodes_by_type('assumption', username)
        unverified_assumptions = [a['content'] for a in unverified_assumptions if a.get('metadata', {}).get('status') == 'unverified']

        # Get underdeveloped nodes (low centrality)
        self.cortex._calculate_centrality() # Ensure centrality is up-to-date
        underdeveloped_nodes = [node for node in self.cortex.get_all_nodes_by_type('insight', username) if node['metadata'].get('centrality', 0) < 0.5]
        underdeveloped_content = [node['content'] for node in underdeveloped_nodes[:3]] # Limit to a few

        # Get high-potential nodes (high centrality, potential for meta-insights)
        high_potential_nodes = [node for node in self.cortex.get_all_nodes_by_type('insight', username) if node['metadata'].get('centrality', 0) > 1.5]
        high_potential_content = [node['content'] for node in high_potential_nodes[:3]] # Limit to a few

        # Get recent suggestions to avoid repetition
        recent_suggestions = self.cortex.get_all_nodes_by_type('proactive_suggestion', username)
        recent_suggestions_content = [s['content'] for s in recent_suggestions if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days < 1]

        prompt = f"""Based on the following information about the user's cognitive graph, generate a single, concise, and engaging proactive suggestion or question for the user. The goal is to encourage deeper exploration, verify assumptions, or address underdeveloped areas of knowledge.

== Context for Proactive Suggestion ==
- Unverified Assumptions: {unverified_assumptions if unverified_assumptions else "None"}
- Underdeveloped Insights (low connections): {underdeveloped_content if underdeveloped_content else "None"}
- High-Potential Insights (well-connected): {high_potential_content if high_potential_content else "None"}
- Recent Suggestions (avoid repeating): {recent_suggestions_content if recent_suggestions_content else "None"}

What is a good proactive suggestion or question for the user to consider or act upon? Be creative and thought-provoking.

Suggestion:"""

        with self.ui_logger.thinking_process("Generating proactive suggestion..."):
            suggestion = self.llm.generate(prompt, temperature=0.7, stop=["\n\n"])
        
        if suggestion and suggestion not in recent_suggestions_content:
            self.cortex.add_node('proactive_suggestion', suggestion, username)
            return suggestion
        return None
