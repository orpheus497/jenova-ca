import os
import json
import shutil
from datetime import datetime
from .concerns import ConcernManager

class InsightManager:
    """Manages the creation, storage, and retrieval of topical insights."""
    def __init__(self, config, ui_logger, file_logger, insights_root, llm, cortex, memory_search):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.insights_root = insights_root
        self.llm = llm
        self.cortex = cortex
        self.memory_search = memory_search
        os.makedirs(self.insights_root, exist_ok=True)
        self.concern_manager = ConcernManager(config, ui_logger, file_logger, insights_root, self.llm)

    def save_insight(self, insight_content: str, username: str, topic: str = None, linked_to: list = None, insight_data: dict = None):
        """Saves an insight, finding or creating a concern for it, and adds it to the Cortex."""
        if not topic:
            existing_topics = self.concern_manager.get_all_concerns()
            topic = self.concern_manager.find_or_create_concern(insight_content, existing_topics)

        user_insights_dir = os.path.join(self.insights_root, username)
        topic_dir = os.path.join(user_insights_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"insight_{timestamp}.json"
        filepath = os.path.join(topic_dir, filename)

        if insight_data is None:
            insight_data = { "topic": topic, "content": insight_content, "timestamp": datetime.now().isoformat(), "user": username, "related_concerns": [] }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insight_data, f, indent=4)
        
        if 'cortex_id' in insight_data:
            self.cortex.update_node(insight_data['cortex_id'], content=insight_content, linked_to=linked_to)
        else:
            node_id = self.cortex.add_node('insight', insight_content, username, linked_to=linked_to)
            insight_data['cortex_id'] = node_id

        self.ui_logger.info(f"New insight saved for user '{username}' under topic '{topic}'")
        self.file_logger.log_info(f"New insight saved: {filepath}")

    def reorganize_insights(self, username: str):
        """Reorganizes all insights for a user, re-categorizing and interlinking them."""
        self.ui_logger.system_message(f"Reorganizing insights for user '{username}'...")
        all_insights = self.get_all_insights(username)
        if not all_insights:
            self.ui_logger.system_message("No insights to reorganize.")
            return

        reorganized_insights = self.concern_manager.reorganize_insights(all_insights)

        user_insights_dir = os.path.join(self.insights_root, username)
        temp_dir = os.path.join(self.insights_root, f"{username}_temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        for insight in reorganized_insights:
            topic_dir = os.path.join(temp_dir, insight['topic'])
            os.makedirs(topic_dir, exist_ok=True)
            
            timestamp = datetime.fromisoformat(insight['timestamp']).strftime("%Y%m%d_%H%M%S")
            filename = f"insight_{timestamp}.json"
            filepath = os.path.join(topic_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(insight, f, indent=4)

        shutil.rmtree(user_insights_dir)
        os.rename(temp_dir, user_insights_dir)
        
        self.ui_logger.system_message("Insights have been reorganized.")

    def get_relevant_insights(self, query: str, username: str, max_insights: int = 3) -> list[str]:
        """Uses semantic search to find the most relevant insights for a given query."""
        return self.memory_search.search_insights(query, username, max_insights)

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
                        filepath = os.path.join(topic_dir, insight_file)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                data['filepath'] = filepath
                                all_insights.append(data)
                        except Exception:
                            continue
        return all_insights
