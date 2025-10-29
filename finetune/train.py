# The JENOVA Cognitive Architecture
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""This module is responsible for creating a fine-tuning dataset from the JENOVA Cognitive Architecture.
"""

import argparse
import glob
import json
import os
import sqlite3


def load_chromadb_memory(db_path, memory_type):
    """
    Load memory data from ChromaDB sqlite database.
    Returns list of (content, metadata) tuples.
    """
    memories = []
    try:
        chroma_db = os.path.join(db_path, "chroma.sqlite3")
        if not os.path.exists(chroma_db):
            return memories

        conn = sqlite3.connect(chroma_db)
        cursor = conn.cursor()

        if cursor:
            cursor.execute("""
                SELECT documents, metadatas 
                FROM embeddings 
                WHERE collection_id IN (SELECT id FROM collections)
            """)

            for row in cursor.fetchall():
                if row[0]:
                    content = row[0]
                    metadata = json.loads(row[1]) if row[1] else {}
                    memories.append((content, metadata))

        if conn:
            conn.close()
    except Exception as e:
        print(
            f"Warning: Could not load {memory_type} memory from {db_path}: {e}")

    return memories


def create_comprehensive_training_data(user_data_dir, output_file):
    """
    Scans the user's complete cognitive architecture and creates comprehensive
    training data.
    """
    if not os.path.exists(user_data_dir):
        print(f"Error: User data directory not found at '{user_data_dir}'")
        return 0

    print(f"Scanning cognitive architecture in: {user_data_dir}")
    training_entries = []

    # 1. Load insights from files (legacy support, will be superseded by graph)
    insights_dir = os.path.join(user_data_dir, "insights")
    if os.path.exists(insights_dir):
        insight_files = glob.glob(os.path.join(
            insights_dir, "**", "*.json"), recursive=True)
        for file_path in insight_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entry = {
                        "messages": [
                            {"role": "user",
                                "content": f"""What insights have you developed about '{data.get("topic", "general")}'?"""},
                            {"role": "assistant",
                                "content": f"""Based on our conversations, I've learned: {data.get('content')}"""}
                        ],
                        "source": "insights_file"
                    }
                    training_entries.append(entry)
            except json.JSONDecodeError:
                print(
                    f"  Warning: Skipping corrupted insight file: {file_path}")

    # 2. Load memories from ChromaDB
    for mem_type in ["episodic", "semantic", "procedural"]:
        mem_path = os.path.join(user_data_dir, "memory", mem_type)
        if os.path.exists(mem_path):
            print(f"Loading {mem_type} memory...")
            memories = load_chromadb_memory(mem_path, mem_type)
            for content, metadata in memories:
                if content and len(content) > 20:
                    entry = {
                        "messages": [
                            {"role": "user",
                                "content": f"Recall information related to: {content[:50]}..."},
                            {"role": "assistant", "content": content}
                        ],
                        "source": mem_type,
                        "metadata": metadata
                    }
                    training_entries.append(entry)

    # 3. Load verified assumptions
    assumptions_file = os.path.join(user_data_dir, "assumptions.json")
    if os.path.exists(assumptions_file):
        print("Loading assumptions...")
        try:
            with open(assumptions_file, 'r', encoding='utf-8') as f:
                assumptions_data = json.load(f)
                verified = assumptions_data.get('true', [])
                for assumption in verified:
                    if content := assumption.get('content'):
                        entry = {
                            "messages": [
                                {"role": "user",
                                    "content": "What have you learned about my preferences?"},
                                {"role": "assistant",
                                    "content": f"I've learned that: {content}"}
                            ],
                            "source": "assumptions"
                        }
                        training_entries.append(entry)
        except json.JSONDecodeError:
            print("  Warning: Could not load assumptions file.")

    # 4. Load from the complete Cognitive Graph
    cortex_path = os.path.join(user_data_dir, "cortex", "cognitive_graph.json")
    if os.path.exists(cortex_path):
        print("Loading cognitive graph...")
        try:
            with open(cortex_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
                nodes = graph.get('nodes', {})
                for node_id, node in nodes.items():
                    node_type = node.get('type')
                    content = node.get('content')
                    if not content or len(content) < 20:
                        continue

                    # Create training data from different node types
                    if node_type in ['insight', 'meta-insight']:
                        user_prompt = f"What can you tell me about the insight: {content[:60]}..."
                        assistant_response = content
                    elif node_type == 'document_chunk':
                        user_prompt = "What information is in this document chunk?"
                        assistant_response = content
                    elif node_type == 'question':
                        # Find the answer by traversing links
                        continue  # Skip for now to avoid complexity
                    else:
                        continue  # Skip other node types

                    entry = {
                        "messages": [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_response}
                        ],
                        "source": f"cortex_{node_type}",
                        "metadata": node.get('metadata', {})
                    }
                    training_entries.append(entry)
        except json.JSONDecodeError:
            print("  Warning: Could not load cognitive graph.")

    # Write all training entries
    print(f"\nCompiling {len(training_entries)} training entries...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in training_entries:
            outfile.write(json.dumps(entry) + "\n")

    source_counts = {}
    for entry in training_entries:
        source = entry.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1

    print(
        f"✓ Successfully created '{output_file}' with {len(training_entries)} entries from:")
    for source, count_val in sorted(source_counts.items()):
        print(f"  - {source}: {count_val} entries")

    return len(training_entries)


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive fine-tuning dataset from JENOVA's complete cognitive architecture."
    )

    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER', 'default')

    default_user_data_dir = os.path.join(
        os.path.expanduser("~"), ".jenova-ai", "users", username)

    parser.add_argument("--user-data-dir", default=default_user_data_dir,
                        help=f"User data directory. Defaults to: {default_user_data_dir}")
    parser.add_argument("--output-file", default="finetune_train.jsonl",
                        help="Output file for the fine-tuning data. Defaults to 'finetune_train.jsonl'")

    args = parser.parse_args()

    print("="*70)
    print("JENOVA Fine-Tuning Data Generation")
    print("="*70)
    print("This script creates a training dataset from your complete cognitive architecture.")
    print("The resulting .jsonl file can be used to fine-tune a language model.")
    print("="*70)

    count = create_comprehensive_training_data(
        args.user_data_dir, args.output_file)

    if count == 0:
        print(
            "\n❌ No training data generated. Interact with JENOVA to build cognitive data.")
        return

    print("\n" + "="*70)
    print("✓ Training data generation complete!")
    print("="*70 + "\n")
    print(f"Output file: {args.output_file}")
    print(f"Total entries: {count}")
    print("\nNext steps:")
    print("  1. Review the generated .jsonl file.")
    print("  2. Use external fine-tuning tools (e.g., Axolotl, llama.cpp training).")
    print("  3. Convert the fine-tuned model back to GGUF for use with JENOVA.")


if __name__ == "__main__":
    main()
