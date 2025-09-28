import os
import json
from datetime import datetime

class InsightManager:
    """Manages the creation, storage, and retrieval of topical insights."""
    def __init__(self, config, ui_logger, file_logger, insights_root):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.insights_root = insights_root
        os.makedirs(self.insights_root, exist_ok=True)

    def save_insight_from_json(self, json_str: str, username: str):
        """Parses a JSON string from the LLM and saves the insight."""
        try:
            # Clean the string to ensure it's valid JSON
            json_str = json_str.strip().replace("`", "")
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()

            data = json.loads(json_str)
            topic = data.get('topic', 'general').strip().replace(' ', '_')
            insight = data.get('insight', '').strip()

            if not topic or not insight:
                self.ui_logger.system_message("Insight generation failed: LLM returned empty topic or insight.")
                self.file_logger.log_error("Insight generation failed: LLM returned empty topic or insight.")
                return

            user_insights_dir = os.path.join(self.insights_root, username)
            topic_dir = os.path.join(user_insights_dir, topic)
            os.makedirs(topic_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insight_{timestamp}.json"
            filepath = os.path.join(topic_dir, filename)

            insight_data = { "topic": topic, "content": insight, "timestamp": datetime.now().isoformat(), "user": username }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(insight_data, f, indent=4)
            
            self.ui_logger.info(f"New insight saved for user '{username}' under topic '{topic}'")
            self.file_logger.log_info(f"New insight saved: {filepath}")

        except json.JSONDecodeError:
            error_msg = f"Insight generation failed: Could not decode JSON from LLM response: {json_str}"
            self.ui_logger.system_message(error_msg)
            self.file_logger.log_error(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred while saving insight: {e}"
            self.ui_logger.system_message(error_msg)
            self.file_logger.log_error(error_msg)

    def get_relevant_insights(self, query: str, username: str, max_insights: int = 3) -> list[str]:
        """Scans insight topics and returns content from topics mentioned in the query."""
        relevant_insights = []
        query_lower = query.lower()
        user_insights_dir = os.path.join(self.insights_root, username)
        
        if not os.path.exists(user_insights_dir):
            return []

        for topic in os.listdir(user_insights_dir):
            topic_dir = os.path.join(user_insights_dir, topic)
            if os.path.isdir(topic_dir) and topic.replace('_', ' ') in query_lower:
                for insight_file in sorted(os.listdir(topic_dir), reverse=True):
                    if insight_file.endswith('.json'):
                        try:
                            with open(os.path.join(topic_dir, insight_file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                relevant_insights.append(f"Learned Insight on '{data['topic']}': {data['content']}")
                                if len(relevant_insights) >= max_insights:
                                    return relevant_insights
                        except Exception:
                            continue
        return relevant_insights

    def get_all_insights(self, username: str) -> list[dict]:
        """Retrieves all insights from the insights directory for a specific user."""
        all_insights = []
        user_insights_dir = os.path.join(self.insights_root, username)

        if not os.path.exists(user_insights_dir):
            return []

        for topic in os.listdir(user_insights_dir):
            topic_dir = os.path.join(user_insights_dir, topic)
            if os.path.isdir(topic_dir):
                for insight_file in os.listdir(topic_dir):
                    if insight_file.endswith('.json'):
                        try:
                            with open(os.path.join(topic_dir, insight_file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                all_insights.append(data)
                        except Exception:
                            continue
        return all_insights