
import os
import json
from datetime import datetime
import uuid

class Cortex:
    """
    The Cortex is the central hub of the AI's cognitive architecture.
    It manages a graph of interconnected cognitive nodes, including insights,
    memories, and assumptions, fostering a deep, evolving understanding.
    """
    def __init__(self, config, ui_logger, file_logger, llm, cortex_root):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.llm = llm
        self.cortex_root = cortex_root
        os.makedirs(self.cortex_root, exist_ok=True)
        self.graph_file = os.path.join(self.cortex_root, 'cognitive_graph.json')
        self.graph = self._load_graph()

    def _load_graph(self):
        """Loads the cognitive graph from a file."""
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"nodes": {}, "links": []}

    def _save_graph(self):
        """Saves the cognitive graph to a file."""
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, indent=4)

    def add_node(self, node_type: str, content: str, user: str, linked_to: list = None):
        """Adds a new node to the cognitive graph."""
        node_id = str(uuid.uuid4())
        
        # Psychological dimension
        sentiment_prompt = f"""Analyze the sentiment of the following text. Respond with a single word: positive, negative, or neutral.

Text: "{content}"

Sentiment:"""
        sentiment = self.llm.generate(prompt=sentiment_prompt, temperature=0.1).strip().lower()

        new_node = {
            "id": node_id,
            "type": node_type,
            "content": content,
            "user": user,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "sentiment": sentiment
            }
        }
        self.graph["nodes"][node_id] = new_node
        
        if linked_to:
            for target_id in linked_to:
                if target_id in self.graph["nodes"]:
                    self.add_link(node_id, target_id, "related_to")

        self._save_graph()
        self.ui_logger.info(f"New {node_type} node created: {node_id} (Sentiment: {sentiment})")
        return node_id

    def add_link(self, source_id: str, target_id: str, relationship: str):
        """Adds a directed link between two nodes in the graph."""
        new_link = {
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "timestamp": datetime.now().isoformat()
        }
        self.graph["links"].append(new_link)
        self._save_graph()

    def get_node(self, node_id: str):
        """Retrieves a node by its ID."""
        return self.graph["nodes"].get(node_id)

    def get_all_nodes_by_type(self, node_type: str, user: str = None):
        """Retrieves all nodes of a specific type, optionally filtered by user."""
        nodes = []
        for node in self.graph["nodes"].values():
            if node["type"] == node_type:
                if user and node.get("user") == user:
                    nodes.append(node)
                elif not user:
                    nodes.append(node)
        return nodes

    def reflect(self, user: str):
        """
        Performs a deep reflection on the cognitive graph. This process analyzes
        the graph to find patterns, infer relationships, generate new links,
        and create meta-insights.
        """
        self.ui_logger.system_message(f"Initiating deep reflection for user '{user}'...")
        
        user_nodes = [n for n in self.graph["nodes"].values() if n.get("user") == user]
        if len(user_nodes) < 5:
            self.ui_logger.system_message("Insufficient data for a meaningful reflection.")
            return

        # 1. Identify unlinked nodes and try to link them
        self._link_orphans(user_nodes)

        # 2. Search for clusters of related nodes to generate meta-insights
        self._generate_meta_insights(user_nodes)

        self._save_graph()
        self.ui_logger.system_message("Reflection complete. The cognitive graph has been refined.")

    def _link_orphans(self, user_nodes: list):
        """Identifies nodes with few links and attempts to create new connections."""
        linked_node_ids = {link['source'] for link in self.graph['links']} | {link['target'] for link in self.graph['links']}
        orphan_nodes = [node for node in user_nodes if node['id'] not in linked_node_ids]

        if len(orphan_nodes) < 2:
            return

        for orphan in orphan_nodes:
            # Don't try to link the same node to itself
            other_nodes = [node for node in user_nodes if node['id'] != orphan['id']]
            if not other_nodes:
                continue

            node_contents = "\n".join([f"- Node {i+1} (ID: {node['id']}): {node['content']}" for i, node in enumerate(other_nodes)])

            prompt = f"""Analyze the following 'orphan' node and the list of other nodes. Identify the single most relevant node from the list that the orphan node could be linked to. 

Orphan Node: "{orphan['content']}"

List of other nodes:
{node_contents}

Respond with a JSON object containing the ID of the most relevant node and the relationship type (e.g., 'elaborates_on', 'conflicts_with', 'related_to').
Example: {{"relevant_node_id": "<node_id>", "relationship": "elaborates_on"}}

JSON Response:"""

            with self.ui_logger.thinking_process("Finding links for orphan nodes..."):
                response_str = self.llm.generate(prompt, temperature=0.3)
            
            try:
                response_data = json.loads(response_str)
                target_id = response_data.get('relevant_node_id')
                relationship = response_data.get('relationship')

                if target_id and relationship and target_id in self.graph["nodes"]:
                    self.add_link(orphan['id'], target_id, relationship)
                    self.ui_logger.info(f"Linked orphan node {orphan['id']} to {target_id}")
            except (json.JSONDecodeError, KeyError):
                continue

    def develop_insight(self, insight_id: str, user: str):
        """Develops an existing insight by generating a more detailed version."""
        insight_node = self.get_node(insight_id)
        if not insight_node or insight_node.get('type') != 'insight':
            self.ui_logger.system_message(f"Node {insight_id} is not a valid insight.")
            return

        prompt = f"""Analyze the following insight and generate a more detailed and developed version of it. You can add more context, examples, or connections to other ideas.

Original Insight: "{insight_node['content']}"

Developed Insight:"""

        with self.ui_logger.thinking_process("Developing insight..."):
            developed_content = self.llm.generate(prompt, temperature=0.5)

        if developed_content:
            new_insight_id = self.add_node('insight', developed_content, user)
            self.add_link(new_insight_id, insight_id, 'develops')
            self.ui_logger.info(f"Developed insight {insight_id} into {new_insight_id}")

    def _generate_meta_insights(self, user_nodes: list):
        """Analyzes clusters of nodes to generate higher-level meta-insights."""
        # Simplified clustering: find nodes with more than 2 links
        node_link_counts = {node['id']: 0 for node in user_nodes}
        for link in self.graph['links']:
            if link['source'] in node_link_counts:
                node_link_counts[link['source']] += 1
            if link['target'] in node_link_counts:
                node_link_counts[link['target']] += 1
        
        clusters = [node_id for node_id, count in node_link_counts.items() if count > 2]

        if not clusters:
            return

        for cluster_node_id in clusters:
            cluster_node = self.get_node(cluster_node_id)
            if not cluster_node or cluster_node.get('type') == 'meta-insight':
                continue

            linked_nodes_content = []
            for link in self.graph['links']:
                if link['source'] == cluster_node_id:
                    linked_node = self.get_node(link['target'])
                    if linked_node:
                        linked_nodes_content.append(linked_node['content'])
                elif link['target'] == cluster_node_id:
                    linked_node = self.get_node(link['source'])
                    if linked_node:
                        linked_nodes_content.append(linked_node['content'])
            
            if len(linked_nodes_content) < 2:
                continue

            content_str = "\n".join([f"- {content}" for content in linked_nodes_content])

            prompt = f"""Analyze the following collection of related insights and generate a single, higher-level 'meta-insight'.

Related Insights:
{content_str}

Meta-Insight:"""

            with self.ui_logger.thinking_process("Generating meta-insight..."):
                meta_insight_content = self.llm.generate(prompt, temperature=0.4)
            
            if meta_insight_content:
                meta_insight_id = self.add_node('meta-insight', meta_insight_content, cluster_node['user'])
                self.add_link(meta_insight_id, cluster_node_id, 'summarizes')
                self.ui_logger.info(f"Generated meta-insight {meta_insight_id} from cluster around {cluster_node_id}")
