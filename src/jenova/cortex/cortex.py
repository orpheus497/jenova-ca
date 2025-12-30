##Script function and purpose: Cortex Module for The JENOVA Cognitive Architecture
##This module implements the central cognitive hub managing interconnected nodes of insights, memories, and assumptions

import os
import json
from datetime import datetime
import uuid
from .graph_metrics import GraphMetrics
from .clustering import AdvancedClustering

##Class purpose: The Cortex is the central hub of the AI's cognitive architecture
##It manages a graph of interconnected cognitive nodes, including insights,
##memories, and assumptions, fostering a deep, evolving understanding
class Cortex:
    """
    The Cortex is the central hub of the AI's cognitive architecture.
    It manages a graph of interconnected cognitive nodes, including insights,
    memories, and assumptions, fostering a deep, evolving understanding.
    """
    ##Function purpose: Initialize Cortex with configuration, loggers, LLM interface, and data directory
    def __init__(self, config, ui_logger, file_logger, llm, cortex_root):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.llm = llm
        self.cortex_root = cortex_root
        os.makedirs(self.cortex_root, exist_ok=True)
        self.docs_path = os.path.join(os.getcwd(), "src", "jenova", "docs")
        self.graph_file = os.path.join(self.cortex_root, 'cognitive_graph.json')
        self.graph = self._load_graph()
        self.reflection_cycle_count = 0
        self.json_grammar = self._load_grammar()
        
        # Initialize advanced clustering and graph metrics modules
        self.clustering = AdvancedClustering(config)
        self.graph_metrics = GraphMetrics(config)

    ##Function purpose: Load the JSON grammar for structured LLM outputs
    def _load_grammar(self):
        """Loads the JSON grammar from the llama.cpp submodule."""
        grammar_path = os.path.join(os.getcwd(), "llama.cpp", "grammars", "json.gbnf")
        if os.path.exists(grammar_path):
            with open(grammar_path, 'r') as f:
                grammar_text = f.read()
            from llama_cpp.llama_grammar import LlamaGrammar
            return LlamaGrammar.from_string(grammar_text)
        self.ui_logger.system_message("JSON grammar file not found at " + grammar_path)
        return None

    ##Function purpose: Load cognitive graph from persistent file storage
    def _load_graph(self):
        """Loads the cognitive graph from a file."""
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"nodes": {}, "links": []}

    ##Function purpose: Save cognitive graph to persistent file storage
    def _save_graph(self):
        """Saves the cognitive graph to a file."""
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(self.graph, f, indent=4)

    ##Function purpose: Add a new node to the cognitive graph with emotion analysis
    def add_node(self, node_type: str, content: str, user: str, linked_to: list = None, metadata: dict = None):
        """Adds a new node to the cognitive graph."""
        node_id = str(uuid.uuid4())
        
        emotion_prompt = f"""Analyze the emotional content of the following text. Respond with a JSON object containing the detected emotions from the following list: ['Joy', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Disgust', 'Love', 'Curiosity'] and their corresponding intensity on a scale from 0.0 to 1.0.

Text: "{content}"

Emotion JSON:"""        
        emotion_str = self.llm.generate(prompt=emotion_prompt, temperature=0.2, grammar=self.json_grammar)
        try:
            emotions = json.loads(emotion_str)
        except json.JSONDecodeError:
            emotions = {}
            self.file_logger.log_error(f"Failed to decode emotion JSON: {emotion_str}")

        new_node = {
            "id": node_id,
            "type": node_type,
            "content": content,
            "user": user,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "emotions": emotions,
                "centrality": 0
            }
        }

        if metadata:
            new_node["metadata"].update(metadata)

        self.graph["nodes"][node_id] = new_node
        
        if linked_to:
            for target_id in linked_to:
                if target_id in self.graph["nodes"]:
                    self.add_link(node_id, target_id, "related_to")

        self._save_graph()
        self.ui_logger.info(f"New {node_type} node created: {node_id} (Emotions: {emotions})")
        return node_id

    ##Function purpose: Add a directed link between two nodes in the cognitive graph
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

    ##Function purpose: Retrieve a node by its unique identifier
    def get_node(self, node_id: str):
        """Retrieves a node by its ID."""
        return self.graph["nodes"].get(node_id)

    ##Function purpose: Retrieve all nodes of a specific type, optionally filtered by user
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

    ##Function purpose: Update an existing node's content or links
    def update_node(self, node_id: str, content: str = None, linked_to: list = None):
        """Updates an existing node in the cognitive graph."""
        if node_id not in self.graph["nodes"]:
            return

        if content:
            self.graph["nodes"][node_id]["content"] = content
            self.graph["nodes"][node_id]["timestamp"] = datetime.now().isoformat()

        if linked_to:
            # Remove existing links from this node
            self.graph["links"] = [link for link in self.graph["links"] if link["source"] != node_id]
            # Add new links
            for target_id in linked_to:
                if target_id in self.graph["nodes"]:
                    self.add_link(node_id, target_id, "related_to")
        
        self._save_graph()
        self.ui_logger.info(f"Node {node_id} updated.")

    ##Function purpose: Performs a deep reflection on the cognitive graph with advanced metrics and clustering
    def reflect(self, user: str) -> list[str]:
        """
        Performs a deep reflection on the cognitive graph. This process analyzes
        the graph to find patterns, infer relationships, generate new links,
        create meta-insights, and calculate comprehensive graph metrics.
        """
        messages = []
        messages.append(f"Initiating deep reflection for user '{user}'...")
        
        user_nodes = [n for n in self.graph["nodes"].values() if n.get("user") == user]
        if len(user_nodes) < 5:
            messages.append("Insufficient data for a meaningful reflection.")
            return messages

        ##Block purpose: Initial reflection pass
        self._calculate_centrality()
        self._link_orphans(user_nodes)
        self._link_external_information(user_nodes)
        self._generate_meta_insights(user_nodes)

        ##Block purpose: Iterative deepening for complex patterns (if enabled)
        reflection_config = self.config.get('cortex', {}).get('reflection', {})
        iterative_enabled = reflection_config.get('iterative_deepening', False)
        max_iterations = reflection_config.get('max_iterations', 2)
        
        if iterative_enabled and len(user_nodes) > 20:  # Only for larger graphs
            messages.append(f"Performing iterative reflection analysis (up to {max_iterations} iterations)...")
            
            ##Block purpose: Iterative analysis loop
            for iteration in range(1, max_iterations + 1):
                messages.append(f"Iteration {iteration}/{max_iterations}...")
                
                ##Block purpose: Recalculate centrality after changes
                self._calculate_centrality()
                
                ##Block purpose: Find newly created orphans from previous iteration
                updated_user_nodes = [n for n in self.graph["nodes"].values() if n.get("user") == user]
                new_orphans = [node for node in updated_user_nodes if node['metadata']['centrality'] == 0]
                if new_orphans:
                    self._link_orphans(new_orphans)
                
                ##Block purpose: Generate additional meta-insights from new connections
                self._generate_meta_insights(updated_user_nodes)
                
                ##Block purpose: Check if significant changes occurred
                if iteration < max_iterations:
                    # Continue if new links were created in this iteration
                    # This is a simple heuristic - could be enhanced
                    break  # For now, do one iteration to avoid performance issues

        # Calculate comprehensive graph metrics
        metrics_config = self.config.get('cortex', {}).get('metrics', {})
        if metrics_config.get('enabled', True):
            try:
                all_metrics = self.graph_metrics.calculate_all_metrics(self.graph, user)
                if all_metrics:
                    # Log key metrics
                    clustering_info = all_metrics.get('clustering', {})
                    modularity_info = all_metrics.get('modularity', {})
                    density_info = all_metrics.get('density', {})
                    
                    if clustering_info.get('average_coefficient'):
                        messages.append(f"Graph clustering coefficient: {clustering_info['average_coefficient']}")
                    if modularity_info.get('modularity'):
                        messages.append(f"Graph modularity: {modularity_info['modularity']} ({modularity_info.get('num_communities', 0)} communities)")
                    if density_info.get('density'):
                        messages.append(f"Graph density: {density_info['density']} ({density_info.get('num_nodes', 0)} nodes, {density_info.get('num_edges', 0)} edges)")
                    
                    # Store metrics in file logger for detailed analysis
                    self.file_logger.log_info(f"Graph metrics: {json.dumps(all_metrics, indent=2)}")
            except Exception as e:
                self.file_logger.log_error(f"Error calculating graph metrics: {e}")

        self.reflection_cycle_count += 1
        pruning_config = self.config.get('cortex', {}).get('pruning', {})
        prune_interval = pruning_config.get('prune_interval', 10)
        if self.reflection_cycle_count % prune_interval == 0:
            messages.extend(self.prune_graph(user))

        messages.extend(self._update_relationship_weights())

        self._save_graph()
        messages.append("Reflection complete. The cognitive graph has been refined.")
        return messages

    def _calculate_centrality(self):
        """Calculates the weighted degree centrality for each node in the graph."""
        weights = self.config.get('cortex', {}).get('relationship_weights', {})
        node_link_counts = {node_id: 0 for node_id in self.graph["nodes"]}
        
        for link in self.graph['links']:
            weight = weights.get(link['relationship'], 1.0)
            if link['source'] in node_link_counts:
                node_link_counts[link['source']] += weight
            if link['target'] in node_link_counts:
                node_link_counts[link['target']] += weight
        
        for node_id, count in node_link_counts.items():
            if node_id in self.graph["nodes"]:
                self.graph["nodes"][node_id]["metadata"]["centrality"] = count

    ##Function purpose: Identifies nodes with few links and attempts to create new connections using batch processing
    def _link_orphans(self, user_nodes: list):
        """
        Identifies orphan nodes (nodes with centrality 0) and links them to related nodes.
        Uses batch processing with semantic pre-filtering for efficiency.
        """
        orphan_nodes = [node for node in user_nodes if node['metadata']['centrality'] == 0]

        if len(orphan_nodes) < 2:
            return

        # Batch process orphans: group them and find candidates more efficiently
        other_nodes = [node for node in user_nodes if node['metadata']['centrality'] > 0]
        
        if not other_nodes:
            return

        # Process orphans in batches to reduce LLM calls
        batch_size = 3  # Process 3 orphans at a time
        for i in range(0, len(orphan_nodes), batch_size):
            batch_orphans = orphan_nodes[i:i + batch_size]
            
            # Create batch prompt for multiple orphans
            orphan_contents = "\n".join([f"- Orphan {j+1} (ID: {orphan['id']}): {orphan['content']}" 
                                         for j, orphan in enumerate(batch_orphans)])
            
            node_contents = "\n".join([f"- Node {j+1} (ID: {node['id']}): {node['content']}" 
                                      for j, node in enumerate(other_nodes[:50])])  # Limit to 50 nodes for efficiency

            prompt = f"""Analyze the following orphan nodes and the list of other nodes. For each orphan, identify which nodes from the list are strongly related. 

Orphan Nodes:
{orphan_contents}

List of other nodes:
{node_contents}

Respond with a valid JSON array where each element corresponds to an orphan node (in order) and contains: {{"orphan_id": "<id>", "relevant_node_ids": ["<node_id_1>", "<node_id_2>"], "relationship": "<relationship_type>"}}. Relationship types: 'elaborates_on', 'conflicts_with', 'related_to', 'develops', 'summarizes'. Do not include any other text.

JSON Response:"""

            response_str = self.llm.generate(prompt, temperature=0.3, grammar=self.json_grammar)
            
            try:
                response_data = json.loads(response_str)
                # Handle both list and single object responses
                if isinstance(response_data, dict):
                    response_data = [response_data]
                
                for response_item in response_data:
                    orphan_id = response_item.get('orphan_id')
                    target_ids = response_item.get('relevant_node_ids', [])
                    relationship = response_item.get('relationship')

                    if orphan_id and target_ids and relationship:
                        # Find the orphan node
                        orphan = next((n for n in batch_orphans if n['id'] == orphan_id), None)
                        if orphan:
                            for target_id in target_ids:
                                if target_id in self.graph["nodes"]:
                                    self.add_link(orphan_id, target_id, relationship)
                                    self.ui_logger.info(f"Linked orphan node {orphan_id} to {target_id}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.file_logger.log_error(f"Failed to link orphan batch. Invalid JSON response: {response_str}. Error: {e}")
                # Fallback to individual processing for this batch
                for orphan in batch_orphans:
                    self._link_orphan_individual(orphan, other_nodes)
    
    ##Function purpose: Fallback method to link a single orphan node individually
    def _link_orphan_individual(self, orphan: dict, other_nodes: list):
        """Fallback method for linking a single orphan when batch processing fails."""
        node_contents = "\n".join([f"- Node {i+1} (ID: {node['id']}): {node['content']}" 
                                  for i, node in enumerate(other_nodes[:30])])

        prompt = f"""Analyze the following 'orphan' node and the list of other nodes. Your task is to identify which nodes from the list are strongly related to the orphan node. 

Orphan Node: "{orphan['content']}"

List of other nodes:
{node_contents}

Respond with a valid JSON object containing a list of related node IDs and the relationship type (e.g., 'elaborates_on', 'conflicts_with', 'related_to'). The JSON object must have the following structure: {{"relevant_node_ids": ["<node_id_1>", "<node_id_2>"], "relationship": "<relationship_type>"}}. Do not include any other text or explanations in your response.

JSON Response:"""

        response_str = self.llm.generate(prompt, temperature=0.3, grammar=self.json_grammar)
        
        try:
            response_data = json.loads(response_str)
            target_ids = response_data.get('relevant_node_ids', [])
            relationship = response_data.get('relationship')

            if target_ids and relationship:
                for target_id in target_ids:
                    if target_id in self.graph["nodes"]:
                        self.add_link(orphan['id'], target_id, relationship)
                        self.ui_logger.info(f"Linked orphan node {orphan['id']} to {target_id}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.file_logger.log_error(f"Failed to link orphan node {orphan['id']}. Invalid JSON response: {response_str}. Error: {e}")

    def _link_external_information(self, user_nodes: list):
        """Links insights to external information and finds relationships between external sources."""
        insights = [node for node in user_nodes if node['type'] == 'insight']
        documents = [node for node in user_nodes if node['type'] == 'document']
        web_results = [node for node in user_nodes if node['type'] == 'web_search_result']

        external_nodes = documents + web_results

        if not external_nodes:
            return

        # Link insights to external nodes
        if insights:
            for insight in insights:
                for ext_node in external_nodes:
                    prompt = f"""Given the following insight and the summary of an external information source, what is the relationship between them (e.g., 'confirms', 'refutes', 'expands_on', 'related_to')? Respond with a JSON object containing the relationship type.

Insight: "{insight['content']}"

External Information: "{ext_node['content']}"

Relationship JSON:"""
                    response_str = self.llm.generate(prompt, temperature=0.3, grammar=self.json_grammar)
                    
                    try:
                        response_data = json.loads(response_str)
                        relationship = response_data.get('relationship')

                        if relationship:
                            self.add_link(insight['id'], ext_node['id'], relationship)
                            self.ui_logger.info(f"Linked insight {insight['id']} to {ext_node['id']} as '{relationship}'")
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        self.file_logger.log_error(f"Failed to link external information for insight {insight['id']}. Invalid JSON response: {response_str}. Error: {e}")
                        continue
        
        # Link external nodes to each other and find contradictions
        for i in range(len(external_nodes)):
            for j in range(i + 1, len(external_nodes)):
                node1 = external_nodes[i]
                node2 = external_nodes[j]

                prompt = f"""Compare the following two pieces of information. What is the relationship between them (e.g., 'agrees_with', 'contradicts', 'related_to')? If they contradict, briefly explain the contradiction. Respond with a JSON object containing 'relationship' and an optional 'contradiction' field.

Information 1: "{node1['content']}"

Information 2: "{node2['content']}"

Relationship JSON:"""
                response_str = self.llm.generate(prompt, temperature=0.3, grammar=self.json_grammar)
                
                try:
                    response_data = json.loads(response_str)
                    relationship = response_data.get('relationship')
                    contradiction = response_data.get('contradiction')

                    if relationship:
                        self.add_link(node1['id'], node2['id'], relationship)
                        self.ui_logger.info(f"Linked {node1['id']} to {node2['id']} as '{relationship}'")

                    if contradiction:
                        contradiction_insight = f"Contradiction found between {node1['id']} and {node2['id']}: {contradiction}"
                        self.add_node('insight', contradiction_insight, node1['user'], linked_to=[node1['id'], node2['id']])

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self.file_logger.log_error(f"Failed to link external sources {node1['id']} and {node2['id']}. Invalid JSON response: {response_str}. Error: {e}")
                    continue

    ##Function purpose: Develop an existing insight by generating a more detailed version
    def develop_insight(self, insight_id: str, user: str) -> list[str]:
        """Develops an existing insight by generating a more detailed version."""
        messages = []
        insight_node = self.get_node(insight_id)
        if not insight_node or insight_node.get('type') != 'insight':
            messages.append(f"Node {insight_id} is not a valid insight.")
            return messages

        prompt = f"""Analyze the following insight and generate a more detailed and developed version of it. You can add more context, examples, or connections to other ideas.\n\nOriginal Insight: "{insight_node['content']}"\n\nDeveloped Insight:"""

        developed_content = self.llm.generate(prompt, temperature=0.5)

        if developed_content:
            new_insight_id = self.add_node('insight', developed_content, user)
            self.add_link(new_insight_id, insight_id, 'develops')
            messages.append(f"Developed insight {insight_id} into {new_insight_id}")
        return messages

    ##Function purpose: Process documents from docs folder, chunk them, and integrate insights into the graph
    def develop_insights_from_docs(self, user: str) -> list[str]:
        """
        Reads documents from the docs folder, chunks them, develops insights,
        and links them to the document node.
        """
        messages = []
        self.processed_docs_file = os.path.join(self.cortex_root, 'processed_documents.json')
        self.processed_docs = self._load_processed_docs()

        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)
            messages.append(f"Docs directory created at {self.docs_path}. Please add documents to be processed.")
            # Create a sample file
            sample_doc_path = os.path.join(self.docs_path, "example.md")
            if not os.path.exists(sample_doc_path):
                with open(sample_doc_path, "w") as f:
                    f.write("# Example Document\n\nThis is an example document. Jenova can read this and generate insights from it.")
                messages.append("An example document has been created for you.")
            return messages

        for filename in os.listdir(self.docs_path):
            filepath = os.path.join(self.docs_path, filename)
            if os.path.isfile(filepath):
                try:
                    last_modified = os.path.getmtime(filepath)
                    if filename in self.processed_docs and self.processed_docs[filename]['last_modified'] >= last_modified:
                        continue

                    messages.append(f"Processing new document: {filename}")
                    self.file_logger.log_info(f"Processing new document: {filename}")
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Generate summary
                    summary_prompt = f"""Summarize the following document in 2-3 sentences, capturing its main purpose and key topics.

Document: \"""{content}\"""

Summary:"""
                    summary = self.llm.generate(summary_prompt, temperature=0.3)

                    # Create a document node with summary in metadata
                    doc_metadata = {"summary": summary} if summary else {}
                    doc_node_id = self.add_node('document', f"Content from document: {filename}", user, metadata=doc_metadata)

                    # Chunk content and generate insights
                    chunks = self._chunk_text(content)
                    insight_ids = []
                    for i, chunk in enumerate(chunks):
                        self.ui_logger.system_message(f"Analyzing chunk {i+1}/{len(chunks)} of {filename}...")
                        prompt = f"""Analyze the following text from a document. Your task is to perform a comprehensive analysis and extract the following information:
1.  A concise summary of the text.
2.  A list of key takeaways or main points (as a list of strings).
3.  A list of any questions that this text can answer (as a list of strings).
4.  A list of key entities (people, places, organizations).
5.  The overall sentiment of the text.

Respond with a single, valid JSON object containing the keys: 'summary', 'takeaways', 'questions', 'entities', and 'sentiment'.

Text: '''{chunk}'''

JSON Response:"""
                        
                        analysis_data = None
                        analysis_json_str = self.llm.generate(prompt, temperature=0.2, grammar=self.json_grammar)
                        try:
                            analysis_data = json.loads(analysis_json_str)
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            self.file_logger.log_error(f"Failed to process analysis from document chunk. Invalid JSON response: {analysis_json_str}. Error: {e}")
                            continue
                        
                        if analysis_data:
                            chunk_summary = analysis_data.get('summary')
                            if chunk_summary:
                                insight_metadata = {
                                    'entities': analysis_data.get('entities'),
                                    'sentiment': analysis_data.get('sentiment'),
                                    'source_chunk': i
                                }
                                summary_insight_id = self.add_node('insight', chunk_summary, user, metadata=insight_metadata)
                                self.add_link(summary_insight_id, doc_node_id, 'derived_from')
                                insight_ids.append(summary_insight_id)

                                # Add takeaways as separate insight nodes
                                for takeaway in analysis_data.get('takeaways', []):
                                    takeaway_id = self.add_node('insight', takeaway, user, metadata={'source_chunk': i})
                                    self.add_link(takeaway_id, summary_insight_id, 'elaborates_on')
                                    self.add_link(takeaway_id, doc_node_id, 'derived_from')

                                # Add questions as separate question nodes
                                for question in analysis_data.get('questions', []):
                                    question_id = self.add_node('question', question, user, metadata={'source_chunk': i})
                                    self.add_link(question_id, summary_insight_id, 'answered_by')
                                    self.add_link(question_id, doc_node_id, 'related_to_document')


                    # Update processed docs tracker
                    self.processed_docs[filename] = {
                        'last_modified': last_modified,
                        'summary': summary,
                        'insight_ids': insight_ids
                    }
                    self._save_processed_docs()
                    messages.append(f"Generated {len(insight_ids)} insights and a summary for {filename}.")
                except Exception as e:
                    self.file_logger.log_error(f"Error processing document {filename}: {e}")
                    messages.append(f"Error processing document {filename}: {e}")

        if not messages:
            messages.append("No new documents to process.")

        return messages

    ##Function purpose: Load processed documents tracker from persistent storage
    def _load_processed_docs(self):
        """Loads the processed documents tracker from a file."""
        if os.path.exists(self.processed_docs_file):
            with open(self.processed_docs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    ##Function purpose: Save processed documents tracker to persistent storage
    def _save_processed_docs(self):
        """Saves the processed documents tracker to a file."""
        with open(self.processed_docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_docs, f, indent=4)

    ##Function purpose: Split text into chunks of specified word count for processing
    def _chunk_text(self, text: str, chunk_size: int = 512) -> list[str]:
        """Splits text into chunks of a specified size."""
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    ##Function purpose: Analyzes clusters of nodes using advanced clustering algorithms to generate higher-level meta-insights
    def _generate_meta_insights(self, user_nodes: list):
        """
        Uses advanced clustering algorithms to identify meaningful clusters of nodes,
        then generates meta-insights from high-quality clusters.
        """
        if len(user_nodes) < 3:
            return
        
        # Use advanced clustering to detect communities
        clusters, quality_metrics = self.clustering.detect_communities(self.graph, user_nodes)
        
        # Log clustering quality metrics
        if quality_metrics:
            self.file_logger.log_info(f"Clustering quality: modularity={quality_metrics.get('modularity', 0)}, "
                                     f"clusters={quality_metrics.get('num_clusters', 0)}, "
                                     f"avg_size={quality_metrics.get('average_cluster_size', 0)}")
        
        # Identify pattern clusters for meta-insight generation
        pattern_clusters = self.clustering.identify_pattern_clusters(self.graph, user_nodes)
        
        # Use pattern clusters if available, otherwise use all clusters
        clusters_to_process = pattern_clusters if pattern_clusters else clusters
        
        for cluster in clusters_to_process:
            if len(cluster) < 3:
                continue

            # Check if a meta-insight that covers this exact cluster already exists
            cluster_set = set(cluster)
            meta_insight_exists = False
            for node_id in cluster:
                node = self.get_node(node_id)
                if node and node['type'] == 'meta-insight':
                    # Check if this meta-insight links to all nodes in the current cluster
                    linked_nodes = {link['target'] for link in self.graph['links'] if link['source'] == node_id}
                    if linked_nodes == cluster_set:
                        meta_insight_exists = True
                        break
            if meta_insight_exists:
                continue

            linked_nodes_content = [self.get_node(node_id)['content'] for node_id in cluster if self.get_node(node_id)]
            if not linked_nodes_content:
                continue
                
            content_str = "\n".join([f"- {content}" for content in linked_nodes_content])

            prompt = f"""Analyze the following collection of related insights. Your task is to synthesize them into a single, novel, and higher-level 'meta-insight'. A meta-insight is a new conclusion, pattern, or theme that emerges from the combination of the existing insights, but is not explicitly stated in any of them. Do not simply summarize the related insights. The meta-insight should be a new piece of knowledge.

Related Insights:
{content_str}

Meta-Insight:"""

            meta_insight_content = self.llm.generate(prompt, temperature=0.6)
            
            if meta_insight_content:
                existing_meta_insights = self.get_all_nodes_by_type('meta-insight', user_nodes[0]['user'])
                is_duplicate = any(existing_insight['content'] == meta_insight_content for existing_insight in existing_meta_insights)
                
                if not is_duplicate:
                    meta_insight_id = self.add_node('meta-insight', meta_insight_content, user_nodes[0]['user'])
                    for node_id in cluster:
                        self.add_link(meta_insight_id, node_id, 'summarizes')
                    self.ui_logger.info(f"Generated meta-insight {meta_insight_id} from a cluster of {len(cluster)} nodes.")

    def prune_graph(self, user: str) -> list[str]:
        """Prunes the cognitive graph to remove old, irrelevant, or low-quality nodes."""
        messages = []
        pruning_config = self.config.get('cortex', {}).get('pruning', {})
        if not pruning_config.get('enabled', False):
            return messages

        messages.append(f"Initiating graph pruning for user '{user}'...")

        now = datetime.now()
        max_age_days = pruning_config.get('max_age_days', 30)
        min_centrality = pruning_config.get('min_centrality', 0.1)

        nodes_to_prune = []
        for node_id, node in self.graph["nodes"].items():
            if node.get("user") != user:
                continue

            age = now - datetime.fromisoformat(node['timestamp'])
            if age.days > max_age_days and node['metadata']['centrality'] < min_centrality:
                nodes_to_prune.append(node_id)

        if not nodes_to_prune:
            messages.append("No nodes to prune.")
            return messages

        for node_id in nodes_to_prune:
            del self.graph["nodes"][node_id]
            self.graph["links"] = [link for link in self.graph["links"] if link["source"] != node_id and link["target"] != node_id]

        self._save_graph()
        messages.append(f"Pruned {len(nodes_to_prune)} nodes from the cognitive graph.")
        return messages

    ##Function purpose: Dynamically update relationship weights based on impact on graph evolution
    def _update_relationship_weights(self) -> list[str]:
        """Dynamically updates the relationship weights based on their impact on the graph."""
        messages = []
        messages.append("Analyzing relationship weights...")
        
        weights = self.config.get('cortex', {}).get('relationship_weights', {})
        last_updated_str = weights.get('last_updated')
        
        if last_updated_str:
            last_updated = datetime.fromisoformat(last_updated_str)
            if (datetime.now() - last_updated).days < 1:
                messages.append("Relationship weights are up-to-date.")
                return messages

        relationship_scores = {rel: {'total_centrality': 0, 'count': 0, 'meta_insights': 0} for rel in weights if rel != 'last_updated'}

        # Calculate scores for each relationship type
        for link in self.graph['links']:
            relationship = link['relationship']
            if relationship in relationship_scores:
                source_node = self.get_node(link['source'])
                target_node = self.get_node(link['target'])
                if source_node and target_node:
                    relationship_scores[relationship]['total_centrality'] += source_node['metadata']['centrality'] + target_node['metadata']['centrality']
                    relationship_scores[relationship]['count'] += 2

                    if source_node['type'] == 'meta-insight' or target_node['type'] == 'meta-insight':
                        relationship_scores[relationship]['meta_insights'] += 1

        new_weights = {}
        for rel, scores in relationship_scores.items():
            if scores['count'] > 0:
                avg_centrality = scores['total_centrality'] / scores['count']
                meta_insight_bonus = scores['meta_insights'] * 0.1 # Add a bonus for generating meta-insights
                new_weights[rel] = avg_centrality + meta_insight_bonus
            else:
                new_weights[rel] = weights.get(rel, 1.0) # Keep the old weight if the relationship type is not used

        # Normalize the weights to a scale of 0-2
        max_weight = max(new_weights.values()) if new_weights else 0
        if max_weight > 0:
            for rel, weight in new_weights.items():
                new_weights[rel] = round(weight / max_weight * 2.0, 2)

        # Update the config
        self.config['cortex']['relationship_weights'] = {'last_updated': datetime.now().isoformat(), **new_weights}
        messages.append("Relationship weights have been updated.")
        return messages