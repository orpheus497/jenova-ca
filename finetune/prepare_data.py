import os
import json

def prepare_finetuning_data():
    """
    Scans the user's insights directory and compiles all generated insights
    into a single JSONL file suitable for fine-tuning with llama.cpp.
    """
    user_home = os.path.expanduser("~")
    insights_root = os.path.join(user_home, ".jenova-ai", "insights")
    output_file = "train.jsonl"
    
    if not os.path.exists(insights_root):
        print(f"Error: Insights directory not found at '{insights_root}'")
        return

    instruction = "Based on the following insight, provide a thoughtful and detailed response or elaboration."
    count = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for topic in os.listdir(insights_root):
            topic_dir = os.path.join(insights_root, topic)
            if os.path.isdir(topic_dir):
                for insight_file in os.listdir(topic_dir):
                    if insight_file.endswith('.json'):
                        file_path = os.path.join(topic_dir, insight_file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                                # This format matches a common instruction-tuning template
                                formatted_entry = {
                                    "text": f"<s>[INST] {instruction}\n\n{data['content']} [/INST] Elaborating on this insight, it's clear that... </s>"
                                }
                                outfile.write(json.dumps(formatted_entry) + "\n")
                                count += 1
                            except json.JSONDecodeError:
                                print(f"Warning: Skipping corrupted JSON file: {file_path}")
                                continue
    
    print(f"Successfully prepared {count} insights into '{output_file}' for fine-tuning.")

if __name__ == "__main__":
    prepare_finetuning_data()