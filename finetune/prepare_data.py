import os
import json
import glob
import argparse

def prepare_insight_data(insights_dir, outfile):
    """
    Scans the user's insights directory and compiles all generated insights
    into a more sophisticated conversational format suitable for fine-tuning.
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
                user = data.get("user", "user")

                if not insight_content:
                    continue

                # Create a more realistic, synthesized user prompt that would lead to the insight
                instruction = f"Hey Jenova, I've been thinking about our conversations on '{topic}'. Can you synthesize your understanding of this for me? What have you learned about my perspective on it?"
                
                # The assistant's response is the insight itself, framed as a thoughtful response
                assistant_response = f"Of course, {user}. Based on our discussions, I've developed an insight regarding '{topic}'. My understanding is that: {insight_content}"

                formatted_entry = {
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                outfile.write(json.dumps(formatted_entry) + "\n")
                count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted JSON file: {file_path}")
                continue
    return count

def prepare_history_data(history_file, outfile):
    """
    Processes a conversation history log in JSONL format, preserving multi-turn context.
    Each line in the file should be a JSON object with 'user_message' and 'ai_message'.
    """
    if not os.path.exists(history_file):
        print(f"Warning: History file not found at '{history_file}'")
        return 0

    count = 0
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                user_message = data.get("user_message")
                ai_message = data.get("ai_message")

                if not user_message or not ai_message:
                    continue

                formatted_entry = {
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": ai_message}
                    ]
                }
                outfile.write(json.dumps(formatted_entry) + "\n")
                count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted JSON line in history file: {line}")
                continue
    return count

def prepare_data(output_file, insights_dir, history_file=None):
    """Prepares the finetuning data and writes it to the output file."""
    total_entries = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        total_entries += prepare_insight_data(insights_dir, outfile)
        if history_file:
            total_entries += prepare_history_data(history_file, outfile)

    print(f"Successfully prepared {total_entries} entries into '{output_file}' for fine-tuning.")
    return total_entries

def main():
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning.")
    parser.add_argument("--insights-dir", default=os.path.join(os.path.expanduser("~"), ".jenova-ai", "insights"), help="Directory containing the user's insights.")
    parser.add_argument("--output-file", default="finetune_train.jsonl", help="Output file for the fine-tuning data.")
    parser.add_argument("--include-history", help="Path to a conversation history file to include in the fine-tuning data.")
    args = parser.parse_args()

    prepare_data(args.output_file, args.insights_dir, args.include_history)

if __name__ == "__main__":
    main()