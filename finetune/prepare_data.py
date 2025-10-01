import os
import json
import glob
import argparse

def prepare_insight_data(insights_dir, outfile):
    """
    Scans the user's insights directory and compiles all generated insights
    into a single JSONL file suitable for fine-tuning.
    """
    if not os.path.exists(insights_dir):
        print(f"Error: Insights directory not found at '{insights_dir}'")
        return 0

    count = 0
    insight_files = glob.glob(os.path.join(insights_dir, "**", "*.json"), recursive=True)

    for file_path in insight_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                topic = data.get("topic", "general")
                insight_content = data.get("content")

                if not insight_content:
                    continue

                instruction = f"Given the following insight on the topic '{topic}', provide a detailed elaboration. Expand on the core idea, offer examples, and explain its implications."
                
                formatted_entry = {
                    "text": f"<s>[INST] {instruction}\n\nInsight: {insight_content} [/INST] Elaborating on the insight regarding '{topic}', one could conclude that... </s>"
                }
                outfile.write(json.dumps(formatted_entry) + "\n")
                count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted JSON file: {file_path}")
                continue
    return count

def prepare_history_data(history_file, outfile):
    """
    Processes a conversation history file and adds it to the fine-tuning data.
    """
    if not os.path.exists(history_file):
        print(f"Warning: History file not found at '{history_file}'")
        return 0

    count = 0
    with open(history_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        user_message = None
        for line in lines:
            if "New query received from" in line:
                user_message = line.split(":", 2)[-1].strip()
            elif "Generated Response:" in line and user_message:
                ai_message = line.split(":", 1)[-1].strip()
                formatted_entry = {
                    "text": f"<s>[INST] {user_message} [/INST] {ai_message} </s>"
                }
                outfile.write(json.dumps(formatted_entry) + "\n")
                count += 1
                user_message = None
    return count

def main():
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning.")
    parser.add_argument("--insights-dir", default=os.path.join(os.path.expanduser("~"), ".jenova-ai", "insights"), help="Directory containing the user's insights.")
    parser.add_argument("--output-file", default="finetune_train.jsonl", help="Output file for the fine-tuning data.")
    parser.add_argument("--include-history", help="Path to a conversation history file to include in the fine-tuning data.")
    args = parser.parse_args()

    total_entries = 0
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        total_entries += prepare_insight_data(args.insights_dir, outfile)
        if args.include_history:
            total_entries += prepare_history_data(args.include_history, outfile)

    print(f"Successfully prepared {total_entries} entries into '{args.output_file}' for fine-tuning.")

if __name__ == "__main__":
    main()