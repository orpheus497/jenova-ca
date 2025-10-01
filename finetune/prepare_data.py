import os
import json
import glob

def prepare_finetuning_data():
    """
    Scans the user's insights directory and compiles all generated insights
    into a single JSONL file suitable for fine-tuning.
    This version creates a more structured instruction prompt.
    """
    user_home = os.path.expanduser("~")
    # It's better to read the config to get the root path, but for now, this is fine
    insights_root = os.path.join(user_home, ".jenova-ai", "insights")
    output_file = "finetune_train.jsonl"
    
    if not os.path.exists(insights_root):
        print(f"Error: Insights directory not found at '{insights_root}'")
        return

    count = 0
    # Using glob to find all insight files recursively
    insight_files = glob.glob(os.path.join(insights_root, "**", "*.json"), recursive=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in insight_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    topic = data.get("topic", "general")
                    insight_content = data.get("content")

                    if not insight_content:
                        continue

                    # Create a more descriptive, structured instruction
                    instruction = f"Given the following insight on the topic '{topic}', provide a detailed elaboration. Expand on the core idea, offer examples, and explain its implications."
                    
                    # This format is common for instruction-following models
                    formatted_entry = {
                        "text": f"<s>[INST] {instruction}\n\nInsight: {insight_content} [/INST] Elaborating on the insight regarding '{topic}', one could conclude that... </s>"
                    }
                    outfile.write(json.dumps(formatted_entry) + "\n")
                    count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping corrupted JSON file: {file_path}")
                    continue
    
    print(f"Successfully prepared {count} insights into '{output_file}' for fine-tuning.")

if __name__ == "__main__":
    prepare_finetuning_data()