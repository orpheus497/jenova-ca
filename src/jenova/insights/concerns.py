import os
import json

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
                self.file_logger.log_error(f"Error loading concerns file: {e}")
                return {}
        return {}

    def _save_concerns(self):
        """Saves the concerns to the concerns.json file."""
        try:
            with open(self.concerns_file, 'w', encoding='utf-8') as f:
                json.dump(self.concerns, f, indent=4)
        except OSError as e:
            self.file_logger.log_error(f"Error saving concerns file: {e}")
            pass

    def get_all_concerns(self) -> list[str]:
        """Returns a list of all existing concern topics."""
        return list(self.concerns.keys())

    def find_or_create_concern(self, insight_content: str, existing_topics: list) -> str:
        """Uses an LLM to find the most relevant concern for an insight, or creates a new one."""
        try:
            if not existing_topics:
                return self._create_new_concern(insight_content)

            prompt = f'''Analyze the following insight and determine if it belongs to any of the existing topics. Respond with the most relevant topic name from the list if a good fit is found. If no existing topic is a good fit, respond with "new".

Existing Topics:
- {"\n- ".join(existing_topics)}

Insight: "{insight_content}"

Relevant Topic:'''
            chosen_topic = self.llm.generate(prompt, temperature=0.2).strip()

            if chosen_topic.lower() != "new" and chosen_topic in existing_topics:
                return chosen_topic
            else:
                return self._create_new_concern(insight_content)
        except Exception as e:
            self.file_logger.log_error(f"Error finding or creating concern: {e}")
            return self._create_new_concern(insight_content) # Fallback to creating a new concern

    def _create_new_concern(self, insight_content: str) -> str:
        """Creates a new concern based on the insight content."""
        prompt = f'''Create a short, one or two-word topic for the following insight:

Insight: "{insight_content}"

Topic:'''
        try:
            new_topic = self.llm.generate(prompt, temperature=0.3).strip().replace(' ', '_')
            if new_topic not in self.concerns:
                self.concerns[new_topic] = {"description": insight_content, "related_concerns": []}
                self._save_concerns()
            return new_topic
        except Exception as e:
            self.file_logger.log_error(f"Error creating new concern: {e}")
            return "general"

    def reorganize_insights(self, all_insights: list) -> list:
        """DEPRECATED: This method is no longer used. Reorganization is handled by Cortex.reflect."""
        self.file_logger.log_warning("ConcernManager.reorganize_insights is deprecated and should not be called.")
        return all_insights
