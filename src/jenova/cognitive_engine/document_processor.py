import os
import glob
import json
from jenova.memory.semantic import SemanticMemory
from jenova.insights.manager import InsightManager
from jenova.assumptions.manager import AssumptionManager

class DocumentProcessor:
    """Processes documents from the docs folder and integrates them into the AI's knowledge base."""
    def __init__(self, docs_path: str, semantic_memory: SemanticMemory, insight_manager: InsightManager, assumption_manager: AssumptionManager, llm):
        self.docs_path = docs_path
        self.semantic_memory = semantic_memory
        self.insight_manager = insight_manager
        self.assumption_manager = assumption_manager
        self.llm = llm
        self.processed_files = {}

    def process_documents(self, username: str):
        """Scans the docs folder for new or updated documents and processes them."""
        for filepath in glob.glob(os.path.join(self.docs_path, "**", "*.md"), recursive=True):
            last_modified = os.path.getmtime(filepath)
            if filepath not in self.processed_files or self.processed_files[filepath] < last_modified:
                self.process_document(filepath, username)
                self.processed_files[filepath] = last_modified

    def process_document(self, filepath: str, username: str):
        """Reads a document, splits it into chunks, and processes each chunk."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = [content[i:i+1000] for i in range(0, len(content), 900)]
        for chunk in chunks:
            self.process_chunk(chunk, username, os.path.basename(filepath))

    def process_chunk(self, chunk: str, username: str, source: str):
        """Processes a chunk of a document to generate new insights, assumptions, or memories."""
        prompt = f'''Analyze the following text from the document '{source}'. Determine if it contains a concrete fact, a potential insight, or a new assumption. Respond with a JSON object containing "type" (fact, insight, or assumption) and "content".

Text: "{chunk}"

JSON Response:'''
        response_str = self.llm.generate(prompt, temperature=0.3)
        try:
            response_data = json.loads(response_str)
            content_type = response_data.get('type')
            content = response_data.get('content')

            if content_type == 'fact':
                self.semantic_memory.add_fact(content, source=source, confidence=0.9)
            elif content_type == 'insight':
                self.insight_manager.save_insight(content, username, topic=source)
            elif content_type == 'assumption':
                self.assumption_manager.add_assumption(content, username)
        except (json.JSONDecodeError, KeyError):
            pass
