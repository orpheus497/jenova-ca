# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

import json
import os
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
        self.concern_manager = ConcernManager(
            config, ui_logger, file_logger, insights_root, self.llm)

    def save_insight(self, insight_content: str, username: str, topic: str = None, linked_to: list = None, insight_data: dict = None):
        """Saves an insight, finding or creating a concern for it, and adds it to the Cortex."""
        self.file_logger.log_info(
            f"Attempting to save insight for user '{username}'. Content: '{insight_content}'")
        try:
            if not topic:
                existing_topics = self.concern_manager.get_all_concerns()
                topic = self.concern_manager.find_or_create_concern(
                    insight_content, existing_topics)
            self.file_logger.log_info(
                f"Insight topic determined as: '{topic}'")

            if insight_data and 'cortex_id' in insight_data:
                self.cortex.update_node(
                    insight_data['cortex_id'], content=insight_content, linked_to=linked_to)
                node_id = insight_data['cortex_id']
                self.file_logger.log_info(
                    f"Updating existing insight node: {node_id}")
            else:
                node_id = self.cortex.add_node(
                    'insight', insight_content, username, linked_to=linked_to)
                self.file_logger.log_info(
                    f"Created new insight node: {node_id}")

            user_insights_dir = os.path.join(self.insights_root, username)
            topic_dir = os.path.join(user_insights_dir, topic)
            os.makedirs(topic_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insight_{timestamp}.json"
            filepath = os.path.join(topic_dir, filename)

            if insight_data is None:
                insight_data = {"topic": topic, "content": insight_content, "timestamp": datetime.now(
                ).isoformat(), "user": username, "related_concerns": []}

            insight_data['cortex_id'] = node_id

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(insight_data, f, indent=4)

            if self.ui_logger:
                self.ui_logger.info(
                    f"New insight saved for user '{username}' under topic '{topic}'")
            if self.file_logger:
                self.file_logger.log_info(f"New insight saved: {filepath}")
        except Exception as e:
            if self.file_logger:
                self.file_logger.log_error(f"Error saving insight: {e}")
            if self.ui_logger:
                self.ui_logger.system_message(
                    "An error occurred while saving the insight.")

    def get_relevant_insights(self, query: str, username: str, max_insights: int = 3) -> list[str]:
        """Uses semantic search to find the most relevant insights for a given query."""
        return self.memory_search.search_insights(query, username, max_insights)

    def get_all_insights(self, username: str) -> list[dict]:
        """Retrieves all insights from the insights directory for a specific user."""
        self.file_logger.log_info(
            f"Getting all insights for user '{username}'")
        all_insights = []
        user_insights_dir = os.path.join(self.insights_root, username)

        if not os.path.exists(user_insights_dir):
            self.file_logger.log_info(
                f"No insights directory found for user '{username}'")
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
                        except Exception as e:
                            self.file_logger.log_error(
                                f"Error reading insight file {filepath}: {e}")
                            continue
        self.file_logger.log_info(
            f"Found {len(all_insights)} insights for user '{username}'")
        return all_insights

    def get_latest_insight_id(self, username: str) -> str | None:
        """Retrieves the cortex_id of the most recently saved insight for a specific user."""
        all_insights = self.get_all_insights(username)
        if not all_insights:
            return None

        # Sort insights by timestamp in descending order
        latest_insight = max(
            all_insights, key=lambda x: x.get('timestamp', ''))

        return latest_insight.get('cortex_id')
