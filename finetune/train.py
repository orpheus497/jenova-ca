
import os
import json
import glob
import argparse

def create_training_data(insights_dir, output_file):
    """
    Scans the user's insights directory and compiles all generated insights
    into a conversational format suitable for fine-tuning with transformers.
    """
    if not os.path.exists(insights_dir):
        print(f"Error: Insights directory not found at '{insights_dir}'")
        return 0

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
    return count

def finetune_with_lora(train_file, output_dir, epochs=3, batch_size=4, learning_rate=2e-4):
    """
    Fine-tune the base model with LoRA using the prepared training data.
    Requires: transformers, peft, accelerate, bitsandbytes
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        import torch
    except ImportError as e:
        print(f"Error: Missing required libraries for fine-tuning: {e}")
        print("Install with: pip install peft bitsandbytes accelerate datasets")
        return
    
    print("Loading Phi-3.5 Mini Instruct model and tokenizer...")
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"Loading training data from {train_file}...")
    dataset = load_dataset('json', data_files=train_file, split='train')
    
    def tokenize_function(examples):
        # Format messages into prompt
        prompts = []
        for messages in examples['messages']:
            prompt = ""
            for msg in messages:
                if msg['role'] == 'user':
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
            prompts.append(prompt)
        
        return tokenizer(prompts, truncation=True, padding='max_length', max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the LoRA adapter
    print(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✓ Fine-tuning complete!")

def main():
    parser = argparse.ArgumentParser(description="Create a fine-tuning dataset from JENOVA's insights and optionally fine-tune with LoRA.")
    
    # Get the username of the current user
    try:
        username = os.getlogin()
    except:
        username = os.environ.get('USER', 'default')
    
    default_insights_dir = os.path.join(os.path.expanduser("~"), ".jenova-ai", "users", username, "insights")
    
    parser.add_argument("--insights-dir", default=default_insights_dir, 
                       help=f"Directory containing the user's insights. Defaults to: {default_insights_dir}")
    parser.add_argument("--output-file", default="finetune_train.jsonl", 
                       help="Output file for the fine-tuning data. Defaults to 'finetune_train.jsonl'")
    parser.add_argument("--prepare-only", action="store_true",
                       help="Only prepare the training data, don't run fine-tuning")
    parser.add_argument("--lora-output", default="/usr/local/share/jenova-ai/lora",
                       help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate for training")
    
    args = parser.parse_args()

    # Always prepare the training data
    count = create_training_data(args.insights_dir, args.output_file)
    
    if count == 0:
        print("No insights found. Train JENOVA by interacting with it first.")
        return
    
    # Optionally run fine-tuning
    if not args.prepare_only:
        print(f"\nStarting LoRA fine-tuning with {count} training examples...")
        finetune_with_lora(
            args.output_file,
            args.lora_output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

if __name__ == "__main__":
    main()
