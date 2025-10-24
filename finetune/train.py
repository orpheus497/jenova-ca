
import os
import json
import glob
import argparse
import sqlite3
from datetime import datetime

def load_chromadb_memory(db_path, memory_type):
    """
    Load memory data from ChromaDB sqlite database.
    Returns list of (content, metadata) tuples.
    """
    memories = []
    try:
        # ChromaDB stores data in chroma.sqlite3
        chroma_db = os.path.join(db_path, "chroma.sqlite3")
        if not os.path.exists(chroma_db):
            return memories
        
        conn = sqlite3.connect(chroma_db)
        cursor = conn.cursor()
        
        # Query embeddings and documents
        cursor.execute("""
            SELECT documents, metadatas 
            FROM embeddings 
            WHERE collection_id IN (SELECT id FROM collections)
        """)
        
        for row in cursor.fetchall():
            if row[0]:  # documents
                content = row[0]
                metadata = json.loads(row[1]) if row[1] else {}
                memories.append((content, metadata))
        
        conn.close()
    except Exception as e:
        print(f"Warning: Could not load {memory_type} memory from {db_path}: {e}")
    
    return memories

def create_comprehensive_training_data(user_data_dir, output_file):
    """
    Scans the user's complete cognitive architecture and creates comprehensive
    training data from:
    - Insights (organized by concerns/topics)
    - Episodic memory (conversation history)
    - Semantic memory (factual knowledge)
    - Procedural memory (how-to procedures)
    - Assumptions (verified knowledge about user)
    - Cognitive graph (document knowledge)
    """
    if not os.path.exists(user_data_dir):
        print(f"Error: User data directory not found at '{user_data_dir}'")
        return 0
    
    print(f"Scanning cognitive architecture in: {user_data_dir}")
    count = 0
    training_entries = []
    
    # 1. Load insights
    insights_dir = os.path.join(user_data_dir, "insights")
    if os.path.exists(insights_dir):
        print("Loading insights...")
        insight_files = glob.glob(os.path.join(insights_dir, "**", "*.json"), recursive=True)
        print(f"  Found {len(insight_files)} insight files")
        
        for file_path in insight_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    topic = data.get("topic", "general")
                    content = data.get("content")
                    user = data.get("user", "user")
                    
                    if content:
                        entry = {
                            "messages": [
                                {"role": "user", "content": f"What insights have you developed about '{topic}'?"},
                                {"role": "assistant", "content": f"Based on our conversations about '{topic}', I've learned: {content}"}
                            ],
                            "source": "insights",
                            "topic": topic
                        }
                        training_entries.append(entry)
            except json.JSONDecodeError:
                print(f"  Warning: Skipping corrupted file: {file_path}")
    
    # 2. Load episodic memory (conversations)
    episodic_path = os.path.join(user_data_dir, "memory", "episodic")
    if os.path.exists(episodic_path):
        print("Loading episodic memory...")
        episodes = load_chromadb_memory(episodic_path, "episodic")
        print(f"  Found {len(episodes)} episodic memories")
        
        for content, metadata in episodes[:500]:  # Limit to recent 500
            if content and len(content) > 20:
                entities = metadata.get('entities', [])
                entity_str = f" involving {', '.join(entities[:3])}" if entities else ""
                entry = {
                    "messages": [
                        {"role": "user", "content": f"Recall our conversation{entity_str}"},
                        {"role": "assistant", "content": content}
                    ],
                    "source": "episodic",
                    "metadata": metadata
                }
                training_entries.append(entry)
    
    # 3. Load semantic memory (facts)
    semantic_path = os.path.join(user_data_dir, "memory", "semantic")
    if os.path.exists(semantic_path):
        print("Loading semantic memory...")
        facts = load_chromadb_memory(semantic_path, "semantic")
        print(f"  Found {len(facts)} semantic facts")
        
        for content, metadata in facts[:300]:  # Limit to 300 most important
            if content and len(content) > 10:
                source = metadata.get('source', 'conversation')
                entry = {
                    "messages": [
                        {"role": "user", "content": f"What do you know about this topic?"},
                        {"role": "assistant", "content": f"I know that: {content} (Source: {source})"}
                    ],
                    "source": "semantic",
                    "metadata": metadata
                }
                training_entries.append(entry)
    
    # 4. Load procedural memory (procedures)
    procedural_path = os.path.join(user_data_dir, "memory", "procedural")
    if os.path.exists(procedural_path):
        print("Loading procedural memory...")
        procedures = load_chromadb_memory(procedural_path, "procedural")
        print(f"  Found {len(procedures)} procedures")
        
        for content, metadata in procedures:
            if content and len(content) > 20:
                goal = metadata.get('goal', 'complete a task')
                entry = {
                    "messages": [
                        {"role": "user", "content": f"How do I {goal}?"},
                        {"role": "assistant", "content": content}
                    ],
                    "source": "procedural",
                    "metadata": metadata
                }
                training_entries.append(entry)
    
    # 5. Load assumptions
    assumptions_file = os.path.join(user_data_dir, "assumptions.json")
    if os.path.exists(assumptions_file):
        print("Loading assumptions...")
        try:
            with open(assumptions_file, 'r', encoding='utf-8') as f:
                assumptions_data = json.load(f)
                verified = assumptions_data.get('verified', []) + assumptions_data.get('true', [])
                print(f"  Found {len(verified)} verified assumptions")
                
                for assumption in verified[:100]:  # Limit to 100
                    content = assumption.get('content')
                    if content:
                        entry = {
                            "messages": [
                                {"role": "user", "content": "What have you learned about my preferences?"},
                                {"role": "assistant", "content": f"I've learned that: {content}"}
                            ],
                            "source": "assumptions"
                        }
                        training_entries.append(entry)
        except json.JSONDecodeError:
            print("  Warning: Could not load assumptions")
    
    # 6. Load cognitive graph (documents)
    cortex_path = os.path.join(user_data_dir, "cortex", "cognitive_graph.json")
    if os.path.exists(cortex_path):
        print("Loading cognitive graph...")
        try:
            with open(cortex_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
                nodes = graph.get('nodes', {})
                
                # Extract document nodes
                doc_count = 0
                for node_id, node in nodes.items():
                    if node.get('type') in ['document', 'document_chunk']:
                        content = node.get('content')
                        metadata = node.get('metadata', {})
                        filename = metadata.get('filename', 'document')
                        
                        if content and len(content) > 50:
                            entry = {
                                "messages": [
                                    {"role": "user", "content": f"What do you know from {filename}?"},
                                    {"role": "assistant", "content": content}
                                ],
                                "source": "documents",
                                "metadata": metadata
                            }
                            training_entries.append(entry)
                            doc_count += 1
                
                print(f"  Found {doc_count} document nodes")
        except json.JSONDecodeError:
            print("  Warning: Could not load cognitive graph")
    
    # Write all training entries
    print(f"\nCompiling {len(training_entries)} training entries...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in training_entries:
            outfile.write(json.dumps(entry) + "\n")
            count += 1
    
    print(f"✓ Successfully created '{output_file}' with {count} training entries from:")
    source_counts = {}
    for entry in training_entries:
        source = entry.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count_val in sorted(source_counts.items()):
        print(f"  - {source}: {count_val} entries")
    
    return count

def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive fine-tuning dataset from JENOVA's complete cognitive architecture."
    )
    
    # Get the username of the current user
    try:
        username = os.getlogin()
    except:
        username = os.environ.get('USER', 'default')
    
    default_user_data_dir = os.path.join(os.path.expanduser("~"), ".jenova-ai", "users", username)
    
    parser.add_argument("--user-data-dir", default=default_user_data_dir, 
                       help=f"User data directory containing all cognitive data. Defaults to: {default_user_data_dir}")
    parser.add_argument("--output-file", default="finetune_train.jsonl", 
                       help="Output file for the fine-tuning data. Defaults to 'finetune_train.jsonl'")
    
    args = parser.parse_args()
    
    print("="*70)
    print("JENOVA Fine-Tuning Data Generation")
    print("="*70)
    print()
    print("This script creates a comprehensive training dataset from your")
    print("complete cognitive architecture including:")
    print("  • Insights and learned knowledge")
    print("  • Conversation history (episodic memory)")
    print("  • Factual knowledge (semantic memory)")
    print("  • Procedures and how-to knowledge")
    print("  • Verified assumptions about you")
    print("  • Document knowledge from cognitive graph")
    print()
    print("The resulting .jsonl file can be used to fine-tune a language")
    print("model using external tools like Axolotl, llama.cpp training,")
    print("or HuggingFace Transformers (convert GGUF to native format first).")
    print()
    print("="*70)
    print()
    
    count = create_comprehensive_training_data(args.user_data_dir, args.output_file)
    
    if count == 0:
        print("\n❌ No training data generated. Interact with JENOVA first to build cognitive data.")
        return
    
    print()
    print("="*70)
    print("✓ Training data generation complete!")
    print("="*70)
    print()
    print(f"Output file: {args.output_file}")
    print(f"Total entries: {count}")
    print()
    print("Next steps:")
    print("  1. Review the generated .jsonl file")
    print("  2. Use external fine-tuning tools:")
    print("     - For GGUF: llama.cpp training utilities")
    print("     - For PyTorch: Convert GGUF to safetensors, then use Transformers/Axolotl")
    print("     - For LoRA: Use PEFT with native model format")
    print("  3. Convert fine-tuned model back to GGUF for use with JENOVA")
    print()
    print("Note: GGUF models must be fine-tuned in native format (safetensors/PyTorch),")
    print("then converted to GGUF. See llama.cpp documentation for details.")
    print()

if __name__ == "__main__":
    main()
