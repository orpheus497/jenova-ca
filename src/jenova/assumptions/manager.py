import os
import json
from datetime import datetime

class AssumptionManager:
    """Manages the lifecycle of assumptions about the user."""
    def __init__(self, config, ui_logger, file_logger, user_data_root, cortex):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.assumptions_file = os.path.join(user_data_root, 'assumptions.json')
        self.cortex = cortex
        self.assumptions = self._load_assumptions()

    def _load_assumptions(self):
        """Loads assumptions from the assumptions.json file."""
        if os.path.exists(self.assumptions_file):
            with open(self.assumptions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"verified": [], "unverified": [], "true": [], "false": []}

    def _save_assumptions(self):
        """Saves assumptions to the assumptions.json file."""
        with open(self.assumptions_file, 'w', encoding='utf-8') as f:
            json.dump(self.assumptions, f, indent=4)

    def add_assumption(self, assumption_content: str, username: str, status: str = 'unverified', linked_to: list = None):
        """Adds a new assumption."""
        if status not in self.assumptions:
            status = 'unverified'
        
        new_assumption = {
            "content": assumption_content,
            "username": username,
            "timestamp": datetime.now().isoformat()
        }
        
        self.assumptions[status].append(new_assumption)
        self._save_assumptions()

        # Add to Cortex
        node_id = self.cortex.add_node('assumption', assumption_content, username, linked_to=linked_to)
        new_assumption['cortex_id'] = node_id

        self.ui_logger.info(f"New assumption added to '{status}' list.")

    def get_all_assumptions(self) -> dict:
        """Returns all assumptions."""
        return self.assumptions

    def get_assumption_to_verify(self, llm, username: str):
        """Gets an unverified assumption and generates a question to verify it."""
        unverified = self.assumptions.get('unverified', [])
        if not unverified:
            self.ui_logger.system_message("No unverified assumptions to check.")
            return None, None

        assumption_to_verify = unverified[0]
        
        prompt = f'''You have an unverified assumption about the user '{username}'. Ask them a clear, direct question to confirm or deny this assumption. The user's response will determine if the assumption is true or false.

Assumption: "{assumption_to_verify["content"]}"

Your question to the user:'''
        
        question = llm.generate(prompt, temperature=0.3)
        return assumption_to_verify, question

    def resolve_assumption(self, assumption, user_response, username: str):
        """Moves an assumption to the 'true' or 'false' list based on user response."""
        # Simple check for now. A more sophisticated approach would use the LLM
        # to interpret the user's response.
        if user_response.lower() in ['yes', 'true', 'y', 'correct']:
            self.assumptions['true'].append(assumption)
            # Also add to cortex as a true insight
            self.cortex.add_node('insight', assumption['content'], username, linked_to=[assumption['cortex_id']])
            self.ui_logger.system_message("Assumption confirmed and converted to insight.")
        else:
            self.assumptions['false'].append(assumption)
            self.ui_logger.system_message("Assumption marked as false.")
        
        # Remove from unverified
        self.assumptions['unverified'] = [a for a in self.assumptions['unverified'] if a['content'] != assumption['content']]
        self._save_assumptions()
