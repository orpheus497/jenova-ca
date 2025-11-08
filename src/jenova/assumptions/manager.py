# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

import json
import os
from datetime import datetime


class AssumptionManager:
    """Manages the lifecycle of assumptions about the user."""

    def __init__(self, config, ui_logger, file_logger, user_data_root, cortex, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.assumptions_file = os.path.join(user_data_root, "assumptions.json")
        self.cortex = cortex
        self.llm = llm
        self.assumptions = self._load_assumptions()

    def _load_assumptions(self):
        """Loads assumptions from the assumptions.json file."""
        if os.path.exists(self.assumptions_file):
            try:
                with open(self.assumptions_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                if self.file_logger:
                    self.file_logger.log_error(f"Error loading assumptions file: {e}")
                return {"verified": [], "unverified": [], "true": [], "false": []}
        return {"verified": [], "unverified": [], "true": [], "false": []}

    def _save_assumptions(self):
        """Saves assumptions to the assumptions.json file."""
        try:
            with open(self.assumptions_file, "w", encoding="utf-8") as f:
                json.dump(self.assumptions, f, indent=4)
        except OSError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error saving assumptions file: {e}")
            pass

    def add_assumption(
        self,
        assumption_content: str,
        username: str,
        status: str = "unverified",
        linked_to: list = None,
    ) -> str:
        """Adds a new assumption, avoiding duplicates."""
        # Check for duplicates across all statuses
        for s in ["unverified", "true", "false"]:
            if s in self.assumptions:
                for existing_assumption in self.assumptions[s]:
                    if (
                        existing_assumption.get("content") == assumption_content
                        and existing_assumption.get("username") == username
                    ):
                        if self.ui_logger:
                            if s in ["true", "false"]:
                                self.ui_logger.system_message(
                                    ".. >> Assumption already exists and has been resolved."
                                )
                            else:  # unverified
                                self.ui_logger.system_message(
                                    ".. >> Assumption already exists and is unverified."
                                )
                        return existing_assumption.get("cortex_id")

        if status not in self.assumptions:
            status = "unverified"

        new_assumption = {
            "content": assumption_content,
            "username": username,
            "timestamp": datetime.now().isoformat(),
        }

        node_id = self.cortex.add_node(
            "assumption", assumption_content, username, linked_to=linked_to
        )
        new_assumption["cortex_id"] = node_id

        self.assumptions[status].append(new_assumption)
        self._save_assumptions()

        return node_id

    def get_all_assumptions(self) -> dict:
        """Returns all assumptions."""
        return self.assumptions

    def get_assumption_to_verify(self, username: str):
        """Gets an unverified assumption and generates a question to verify it."""
        unverified = self.assumptions.get("unverified", [])
        if not unverified:
            return None, None

        assumption_to_verify = unverified[0]

        prompt = f"""You have an unverified assumption about the user '{username}'. Ask them a clear, direct question to confirm or deny this assumption. The user's response will determine if the assumption is true or false.

Assumption: "{assumption_to_verify["content"]}"

Your question to the user:"""

        try:
            question = self.llm.generate(prompt, temperature=0.3)
            return assumption_to_verify, question
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(
                    f"Error generating assumption verification question: {e}"
                )
            return None, None

    def resolve_assumption(self, assumption, user_response: str, username: str):
        """Moves an assumption to the 'true' or 'false' list based on user response."""
        prompt = f"""Analyze the user's response to determine if it confirms or denies the assumption. Respond with "true" or "false".

Assumption: "{assumption["content"]}"
User Response: "{user_response}"

Result:"""

        try:
            result = self.llm.generate(prompt, temperature=0.1).strip().lower()
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error resolving assumption: {e}")
            return

        if result == "true":
            self.assumptions["true"].append(assumption)
            self.assumptions["verified"].append(assumption)
            # Also add to cortex as a true insight
            linked_to_cortex_id = []
            if "cortex_id" in assumption:
                linked_to_cortex_id.append(assumption["cortex_id"])
            else:
                # If cortex_id is missing, create a new node for the assumption
                node_id = self.cortex.add_node(
                    "assumption", assumption["content"], username
                )
                assumption["cortex_id"] = node_id
                linked_to_cortex_id.append(node_id)
            self.cortex.add_node(
                "insight",
                assumption["content"],
                username,
                linked_to=linked_to_cortex_id,
            )
            if self.ui_logger:
                self.ui_logger.system_message(
                    "Assumption confirmed and converted to insight."
                )
        else:
            self.assumptions["false"].append(assumption)
            self.assumptions["verified"].append(assumption)
            if self.ui_logger:
                self.ui_logger.system_message("Assumption marked as false.")

        # Remove from unverified
        for i, a in enumerate(self.assumptions["unverified"]):
            if a["content"] == assumption["content"] and a["username"] == username:
                self.assumptions["unverified"].pop(i)
                break
        self._save_assumptions()

    def update_assumption(
        self, old_assumption_content: str, new_assumption_content: str, username: str
    ):
        """Updates an existing assumption."""
        for status in ["verified", "unverified", "true", "false"]:
            for assumption in self.assumptions[status]:
                if (
                    assumption["content"] == old_assumption_content
                    and assumption["username"] == username
                ):
                    assumption["content"] = new_assumption_content
                    assumption["timestamp"] = datetime.now().isoformat()
                    if "cortex_id" in assumption:
                        self.cortex.update_node(
                            assumption["cortex_id"], content=new_assumption_content
                        )
                    else:
                        node_id = self.cortex.add_node(
                            "assumption", new_assumption_content, username
                        )
                        assumption["cortex_id"] = node_id
                    self._save_assumptions()
                    if self.ui_logger:
                        self.ui_logger.info(
                            f"Assumption updated: {old_assumption_content} -> {new_assumption_content}"
                        )
                    return
        if self.ui_logger:
            self.ui_logger.system_message("Assumption not found.")
