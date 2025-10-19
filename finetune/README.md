# JENOVA - Fine-Tuning Framework

This directory contains the tools to prepare your AI's learned insights for fine-tuning with transformers and LoRA/PEFT. This allows you to create an adapter that enhances the base model with knowledge from your past interactions.

## How It Works

1.  **Insight Generation:** As you interact with JENOVA, the `InsightManager` automatically saves novel conclusions and summaries of your conversations as structured `.json` files in your `~/.jenova-ai/users/<username>/insights/` directory.

2.  **Data Preparation:** The `train.py` script reads all of these individual insight files and compiles them into a single `finetune_train.jsonl` file. This file is formatted for instruction-based fine-tuning with HuggingFace transformers.

## Fine-Tuning Workflow

1.  **Interact with JENOVA:** Use the AI normally. The more you interact and the more topics you cover, the more insight files will be generated.

2.  **Prepare the Training Data:** Once you have a sufficient number of insights (e.g., 50+), run the preparation script from the project root:

    ```bash
    python finetune/train.py --prepare-only
    ```

    This will create a `finetune_train.jsonl` file.

3.  **Fine-Tune with LoRA:** Use the prepared data with PEFT (Parameter-Efficient Fine-Tuning) to create a LoRA adapter:

    ```bash
    # Install additional dependencies if needed
    pip install peft bitsandbytes accelerate
    
    # Run fine-tuning (requires GPU recommended)
    python finetune/train.py --epochs 3 --batch-size 4 --learning-rate 2e-4
    ```

    This creates a LoRA adapter in `/usr/local/share/jenova-ai/lora/` that can be loaded alongside the base model.

4.  **Integrate the Adapter:** The system can automatically load LoRA adapters if configured. The adapter will enhance the base model with your personalized knowledge without modifying the original model weights.