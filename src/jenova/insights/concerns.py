# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for managing the lifecycle of concerns.
"""

import json
import os


class ConcernManager:
    """Manages the lifecycle of concerns, including their creation, updating, and interlinking."""

    def __init__(self, config, ui_logger, file_logger, insights_root, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.insights_root = insights_root
        self.concerns_file = os.path.join(self.insights_root, 'concerns.json')
        self.concerns = self._load_concerns()
        self.llm = llm

    def _load_concerns(self):
        """Loads the concerns from the concerns.json file."""
        if os.path.exists(self.concerns_file):
            try:
                with open(self.concerns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                if self.file_logger:
                    self.file_logger.log_error(
                        f"Error loading concerns file: {e}")
                return {}
        return {}

    def _save_concerns(self):
        """Saves the concerns to the concerns.json file."""
        try:
            with open(self.concerns_file, 'w', encoding='utf-8') as f:
                json.dump(self.concerns, f, indent=4)
        except OSError as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error saving concerns file: {e}")

    def get_all_concerns(self) -> list[str]:
        """Returns a list of all existing concern topics."""
        return list(self.concerns.keys())

    def find_or_create_concern(self, insight_content: str, existing_topics: list) -> str:
        """Uses an LLM to find the most relevant concern for an insight, or creates a new one."""
        try:
            if not existing_topics:
                return self._create_new_concern(insight_content)

            existing_topics_str = "\n- ".join(existing_topics)
            prompt_template = """Analyze the following insight and determine if it belongs to any of the existing topics. Respond with the most relevant topic name from the list if a good fit is found. If no existing topic is a good fit, respond with "new".

Existing Topics:
- {existing_topics_str}

Insight: "{insight_content}"

Relevant Topic:"""
            prompt = prompt_template.format(existing_topics_str=existing_topics_str, insight_content=insight_content)
            chosen_topic = self.llm.generate(prompt, temperature=0.2).strip()

            if chosen_topic.lower() != "new" and chosen_topic in existing_topics:
                return chosen_topic
            else:
                return self._create_new_concern(insight_content)
        except Exception as e:
            self.file_logger.log_error(
                f"Error finding or creating concern: {e}")
            # Fallback to creating a new concern
            return self._create_new_concern(insight_content)

    def _create_new_concern(self, insight_content: str) -> str:
        """Creates a new concern based on the insight content."""
        prompt_template = """Create a short, one or two-word topic for the following insight:

Insight: "{insight_content}"

Topic:"""
        prompt = prompt_template.format(insight_content=insight_content)
        try:
            new_topic = self.llm.generate(
                prompt, temperature=0.3).strip().replace(' ', '_')
            if new_topic not in self.concerns:
                self.concerns[new_topic] = {
                    "description": insight_content, "related_concerns": []}
                self._save_concerns()
            return new_topic
        except Exception as e:
            self.file_logger.log_error(f"Error creating new concern: {e}")
            return "general"

    # Note: reorganize_insights() method has been removed
    # Insight reorganization is now handled by Cortex.reflect() method
    # See src/jenova/cortex/cortex.py for the new implementation
