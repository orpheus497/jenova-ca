import os
import json
import chromadb
from chromadb.utils import embedding_functions

class ProceduralMemory:
    def __init__(self, config, ui_logger, file_logger, db_path, llm):
        self.config = config
        self.ui_logger = ui_logger
        self.file_logger = file_logger
        self.db_path = db_path
        self.llm = llm
        os.makedirs(self.db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config['model']['embedding_model'])
        self.collection = client.get_or_create_collection(name="procedural_steps", embedding_function=self.embedding_function)
        self.next_id = self.collection.count()

    def add_procedure(self, procedure: str, goal: str = None, steps: list = None, context: str = None):
        if not goal or not steps or not context:
            prompt = f'''Analyze the following procedure and extract the goal, the steps, and the context. Respond with a JSON object containing "goal" (a string), "steps" (a list of strings), and "context" (a string).

Procedure: "{procedure}"

JSON Response:'''
            response_str = self.llm.generate(prompt, temperature=0.2)
            try:
                response_data = json.loads(response_str)
                goal = response_data.get('goal')
                steps = response_data.get('steps', [])
                context = response_data.get('context')
            except (json.JSONDecodeError, KeyError):
                goal = procedure
                steps = []
                context = 'general'

        metadata = {
            "goal": goal,
            "steps": json.dumps(steps),
            "context": context
        }
        
        doc_id = f"proc_{self.next_id}"
        self.collection.add(ids=[doc_id], documents=[procedure], metadatas=[metadata])
        self.next_id += 1
        self.file_logger.log_info(f"Added procedure {doc_id} to procedural memory.")

    def search(self, query: str, n_results: int = 2) -> list[tuple[str, float]]:
        if self.collection.count() == 0: return []
        n_results = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results['documents']: return []
        return list(zip(results['documents'][0], results['distances'][0]))