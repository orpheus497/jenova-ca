import os
import json
from datetime import datetime

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
            with open(self.concerns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_concerns(self):
        """Saves the concerns to the concerns.json file."""
        with open(self.concerns_file, 'w', encoding='utf-8') as f:
            json.dump(self.concerns, f, indent=4)

    def find_or_create_concern(self, insight_content: str, existing_topics: list) -> str:
        """Uses an LLM to find the most relevant concern for an insight, or creates a new one."""
        if not existing_topics:
            # If no topics exist, create a new one based on the insight.
            prompt = f'''Create a short, one or two-word topic for the following insight:

Insight: "{insight_content}"

Topic:'''
            new_topic = self.llm.generate(prompt, temperature=0.3).strip().replace(' ', '_')
            self.concerns[new_topic] = {"description": insight_content, "related_concerns": []}
            self._save_concerns()
            return new_topic

        prompt = f'''Analyze the following insight and determine if it belongs to any of the existing topics. Respond with the most relevant topic name from the list if a good fit is found. If no existing topic is a good fit, respond with "new".

Existing Topics:
- {"\n- ".join(existing_topics)}

Insight: "{insight_content}"

Relevant Topic:'''
        chosen_topic = self.llm.generate(prompt, temperature=0.2).strip()

        if chosen_topic.lower() != "new" and chosen_topic in existing_topics:
            return chosen_topic
        else:
            # Create a new topic
            prompt = f'''Create a short, one or two-word topic for the following insight:

Insight: "{insight_content}"

Topic:'''
            new_topic = self.llm.generate(prompt, temperature=0.3).strip().replace(' ', '_')
            self.concerns[new_topic] = {"description": insight_content, "related_concerns": []}
            self._save_concerns()
            return new_topic

    def reorganize_insights(self, all_insights: list) -> list:
        """Uses an LLM to reorganize insights, re-categorize, and create interlinks."""
        if not all_insights:
            return []

        insights_str = "\n".join([f'Insight {i+1}: "{insight["content"]}" (Current Topic: {insight["topic"]})' for i, insight in enumerate(all_insights)])

        prompt = f'''Analyze the following insights. For each insight, determine the most appropriate topic from the existing list of topics. Also, identify any other topics it is related to (interlinks).

Existing Topics:
- {"\n- ".join(self.get_all_concerns())}

Insights:
{insights_str}

Respond with a JSON array, where each object has the following keys:
- "insight_number": The original number of the insight.
- "new_topic": The most appropriate topic for the insight.
- "interlinked_topics": A list of other related topics.

JSON Response:'''
        
        with self.ui_logger.thinking_process("Reorganizing and interlinking insights..."):
            reorg_json_str = self.llm.generate(prompt, temperature=0.4)

        try:
            reorg_data = json.loads(reorg_json_str)
            reorganized_insights = []
            for item in reorg_data:
                insight_index = item['insight_number'] - 1
                if 0 <= insight_index < len(all_insights):
                    insight = all_insights[insight_index]
                    insight['topic'] = item['new_topic']
                    insight['related_concerns'] = item['interlinked_topics']
                    reorganized_insights.append(insight)
            return reorganized_insights
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.ui_logger.system_message(f"Failed to reorganize insights due to an error: {e}")
            self.file_logger.log_error(f"Failed to reorganize insights: {reorg_json_str}")
            return all_insights # Return original insights on failure

    def get_all_concerns(self) -> list:
        """Returns a list of all concern names."""
        return list(self.concerns.keys())