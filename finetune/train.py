##Script function and purpose: Fine-tuning Data Generator for The JENOVA Cognitive Architecture
##This script converts user insights into a JSONL format suitable for fine-tuning LLMs
##It synthesizes conversational pairs from stored insights to enable personalized model training

import os
import json
import glob
import argparse

##Function purpose: Scan insights directory and compile into fine-tuning format
def create_training_data(insights_dir, output_file):
    """
    Scans the user's insights directory and compiles all generated insights
    into a conversational format suitable for fine-tuning.
    """
    if not os.path.exists(insights_dir):
        print(f"Error: Insights directory not found at '{insights_dir}'")
        return

    print(f"Scanning for insights in: {insights_dir}")
    count = 0
    insight_files = glob.glob(os.path.join(insights_dir, "**", "*.json"), recursive=True)
    print(f"Found {len(insight_files)} insight files.")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in insight_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    topic = data.get("topic", "general")
                    insight_content = data.get("content")
                    user = data.get("user", "user")

                    if not insight_content:
                        continue

                    # Synthesize a user prompt that would lead to the insight
                    instruction = f"Let's reflect on the topic of '{topic}'. Based on our conversations, can you summarize your understanding of my perspective?"
                    
                    # The assistant's response is the insight itself
                    assistant_response = f"Of course, {user}. I've developed an insight regarding '{topic}'. My understanding is that: {insight_content}"

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
    
    print(f"Successfully created '{output_file}' with {count} training entries.")

##Function purpose: Parse command line arguments and run the training data generator
def main():
    parser = argparse.ArgumentParser(description="Create a fine-tuning dataset from JENOVA's insights.")
    
    # Get the username of the current user
    username = os.getlogin()
    
    default_insights_dir = os.path.join(os.path.expanduser("~"), ".jenova-ai", "users", username, "insights")
    
    parser.add_argument("--insights-dir", default=default_insights_dir, help=f"Directory containing the user's insights. Defaults to the current user's insights directory: {default_insights_dir}")
    parser.add_argument("--output-file", default="finetune_train.jsonl", help="Output file for the fine-tuning data. Defaults to 'finetune_train.jsonl' in the current directory.")
    
    args = parser.parse_args()

    create_training_data(args.insights_dir, args.output_file)

if __name__ == "__main__":
    main()
