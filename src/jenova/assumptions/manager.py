##Script function and purpose: Assumption Manager for The JENOVA Cognitive Architecture
##This module manages the lifecycle of user assumptions including creation, verification, and resolution

import os
from datetime import datetime
from typing import Any, Dict, Optional
from jenova.utils.file_io import load_json_file, save_json_file

##Class purpose: Manages lifecycle of assumptions about users
class AssumptionManager:
    """Manages the lifecycle of assumptions about the user."""
    ##Function purpose: Initialize assumption manager with configuration and components
    def __init__(
        self, 
        config: Dict[str, Any], 
        ui_logger: Any, 
        file_logger: Any, 
        user_data_root: str, 
        cortex: Any, 
        llm: Any, 
        integration_layer: Optional[Any] = None
    ) -> None:
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.assumptions_file = os.path.join(user_data_root, 'assumptions.json')
        self.cortex = cortex
        self.llm = llm
        self.integration_layer = integration_layer  # Optional integration layer for Cortex-Memory feedback
        ##Block purpose: Load assumptions using centralized file I/O utility
        self.assumptions = load_json_file(
            self.assumptions_file, 
            {"verified": [], "unverified": [], "true": [], "false": []},
            file_logger
        )

    ##Function purpose: Save assumptions to persistent JSON file
    def _save_assumptions(self):
        """Saves assumptions to the assumptions.json file."""
        save_json_file(self.assumptions_file, self.assumptions, indent=4, file_logger=self.file_logger)

    ##Function purpose: Add a new assumption, avoiding duplicates
    def add_assumption(self, assumption_content: str, username: str, status: str = 'unverified', linked_to: list = None) -> str:
        """Adds a new assumption, avoiding duplicates."""
        # Check for duplicates across all statuses
        for s in ['unverified', 'true', 'false']:
            if s in self.assumptions:
                for existing_assumption in self.assumptions[s]:
                    if existing_assumption.get('content') == assumption_content and existing_assumption.get('username') == username:
                        if s in ['true', 'false']:
                            self.ui_logger.system_message(".. >> Assumption already exists and has been resolved.")
                            return existing_assumption.get('cortex_id')
                        else: # unverified
                            self.ui_logger.system_message(".. >> Assumption already exists and is unverified.")
                            return existing_assumption.get('cortex_id')

        if status not in self.assumptions:
            status = 'unverified'
        
        new_assumption = {
            "content": assumption_content,
            "username": username,
            "timestamp": datetime.now().isoformat()
        }
        
        node_id = self.cortex.add_node('assumption', assumption_content, username, linked_to=linked_to)
        new_assumption['cortex_id'] = node_id

        ##Block purpose: Provide feedback from Cortex to Memory (if integration layer available)
        integration_config = self.config.get('cortex', {}).get('integration', {})
        if integration_config.get('cortex_to_memory_feedback', False) and self.integration_layer:
            try:
                self.integration_layer.feedback_cortex_to_memory(node_id, username)
            except Exception as e:
                self.file_logger.log_error(f"Error providing Cortex-to-Memory feedback for assumption {node_id}: {e}")

        self.assumptions[status].append(new_assumption)
        self._save_assumptions()

        return node_id

    ##Function purpose: Return all assumptions grouped by status
    def get_all_assumptions(self) -> dict:
        """Returns all assumptions."""
        return self.assumptions

    ##Function purpose: Get an unverified assumption and generate verification question
    def get_assumption_to_verify(self, username: str):
        """Gets an unverified assumption and generates a question to verify it."""
        unverified = self.assumptions.get('unverified', [])
        if not unverified:
            return None, None

        assumption_to_verify = unverified[0]
        
        prompt = f'''You have an unverified assumption about the user '{username}'. Ask them a clear, direct question to confirm or deny this assumption. The user's response will determine if the assumption is true or false.

Assumption: "{assumption_to_verify["content"]}"

Your question to the user:'''
        
        try:
            question = self.llm.generate(prompt, temperature=0.3)
            return assumption_to_verify, question
        except Exception as e:
            self.file_logger.log_error(f"Error generating assumption verification question: {e}")
            return None, None

    ##Function purpose: Move assumption to true or false list based on user response
    def resolve_assumption(self, assumption, user_response: str, username: str):
        """Moves an assumption to the 'true' or 'false' list based on user response."""
        prompt = f'''Analyze the user's response to determine if it confirms or denies the assumption. Respond with "true" or "false".

Assumption: "{assumption["content"]}"
User Response: "{user_response}"

Result:'''
        
        try:
            result = self.llm.generate(prompt, temperature=0.1).strip().lower()
        except Exception as e:
            self.file_logger.log_error(f"Error resolving assumption: {e}")
            return

        if result == 'true':
            self.assumptions['true'].append(assumption)
            self.assumptions['verified'].append(assumption)
            # Also add to cortex as a true insight
            linked_to_cortex_id = []
            if 'cortex_id' in assumption:
                linked_to_cortex_id.append(assumption['cortex_id'])
            else:
                # If cortex_id is missing, create a new node for the assumption
                node_id = self.cortex.add_node('assumption', assumption['content'], username)
                assumption['cortex_id'] = node_id
                linked_to_cortex_id.append(node_id)
            insight_node_id = self.cortex.add_node('insight', assumption['content'], username, linked_to=linked_to_cortex_id)
            
            ##Block purpose: Provide feedback from Cortex to Memory for new insight (if integration layer available)
            integration_config = self.config.get('cortex', {}).get('integration', {})
            if integration_config.get('cortex_to_memory_feedback', False) and self.integration_layer:
                try:
                    self.integration_layer.feedback_cortex_to_memory(insight_node_id, username)
                except Exception as e:
                    self.file_logger.log_error(f"Error providing Cortex-to-Memory feedback for converted insight {insight_node_id}: {e}")
            
            self.ui_logger.system_message("Assumption confirmed and converted to insight.")
        else:
            self.assumptions['false'].append(assumption)
            self.assumptions['verified'].append(assumption)
            self.ui_logger.system_message("Assumption marked as false.")
        
        # Remove from unverified
        for i, a in enumerate(self.assumptions['unverified']):
            if a['content'] == assumption['content'] and a['username'] == username:
                self.assumptions['unverified'].pop(i)
                break
        self._save_assumptions()

    ##Function purpose: Update an existing assumption's content
    def update_assumption(self, old_assumption_content: str, new_assumption_content: str, username: str):
        """Updates an existing assumption."""
        for status in ['verified', 'unverified', 'true', 'false']:
            for assumption in self.assumptions[status]:
                if assumption['content'] == old_assumption_content and assumption['username'] == username:
                    assumption['content'] = new_assumption_content
                    assumption['timestamp'] = datetime.now().isoformat()
                    if 'cortex_id' in assumption:
                        self.cortex.update_node(assumption['cortex_id'], content=new_assumption_content)
                    else:
                        node_id = self.cortex.add_node('assumption', new_assumption_content, username)
                        assumption['cortex_id'] = node_id
                    self._save_assumptions()
                    self.ui_logger.info(f"Assumption updated: {old_assumption_content} -> {new_assumption_content}")
                    return
        self.ui_logger.system_message("Assumption not found.")
