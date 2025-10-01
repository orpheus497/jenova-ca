import os
import json
from datetime import datetime

class AssumptionManager:
    """Manages the lifecycle of assumptions about the user."""
    def __init__(self, config, ui_logger, file_logger, user_data_root, cortex, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.assumptions_file = os.path.join(user_data_root, 'assumptions.json')
        self.cortex = cortex
        self.llm = llm
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

    def get_assumption_to_verify(self, username: str):
        """Gets an unverified assumption and generates a question to verify it."""
        unverified = self.assumptions.get('unverified', [])
        if not unverified:
            return None, None

        assumption_to_verify = unverified[0]
        
        prompt = f'''You have an unverified assumption about the user '{username}'. Ask them a clear, direct question to confirm or deny this assumption. The user's response will determine if the assumption is true or false.

Assumption: "{assumption_to_verify["content"]}"

Your question to the user:'''
        
        question = self.llm.generate(prompt, temperature=0.3)
        return assumption_to_verify, question

    def resolve_assumption(self, assumption, user_response: str, username: str):
        """Moves an assumption to the 'true' or 'false' list based on user response."""
        prompt = f'''Analyze the user's response to determine if it confirms or denies the assumption. Respond with "true" or "false".

Assumption: "{assumption["content"]}"
User Response: "{user_response}"

Result:'''
        
        result = self.llm.generate(prompt, temperature=0.1).strip().lower()

        if result == 'true':
            self.assumptions['true'].append(assumption)
            self.assumptions['verified'].append(assumption)
            # Also add to cortex as a true insight
            self.cortex.add_node('insight', assumption['content'], username, linked_to=[assumption['cortex_id']])
            self.ui_logger.system_message("Assumption confirmed and converted to insight.")
        else:
            self.assumptions['false'].append(assumption)
            self.assumptions['verified'].append(assumption)
            self.ui_logger.system_message("Assumption marked as false.")
        
        # Remove from unverified
        self.assumptions['unverified'] = [a for a in self.assumptions['unverified'] if a['content'] != assumption['content']]
        self._save_assumptions()

    def update_assumption(self, old_assumption_content: str, new_assumption_content: str, username: str):
        """Updates an existing assumption."""
        for status in ['verified', 'unverified', 'true', 'false']:
            for assumption in self.assumptions[status]:
                if assumption['content'] == old_assumption_content and assumption['username'] == username:
                    assumption['content'] = new_assumption_content
                    assumption['timestamp'] = datetime.now().isoformat()
                    self.cortex.update_node(assumption['cortex_id'], content=new_assumption_content)
                    self._save_assumptions()
                    self.ui_logger.info(f"Assumption updated: {old_assumption_content} -> {new_assumption_content}")
                    return
        self.ui_logger.system_message("Assumption not found.")
