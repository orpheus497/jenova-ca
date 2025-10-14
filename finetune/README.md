# Jenova AI - Fine-Tuning Framework

This directory contains the tools to prepare your AI's learned insights for fine-tuning. This allows you to create a new `jenova.gguf` model that has "baked-in" knowledge from your past interactions.

## How It Works

1.  **Insight Generation:** As you interact with Jenova AI, the `InsightManager` automatically saves novel conclusions and summaries of your conversations as structured `.json` files in your `~/.jenova-ai/users/<username>/insights/` directory.

2.  **Cognitive Process Accelerator (CPA):** The CPA continuously learns from your interactions and maintains persistent state in `~/.jenova-ai/users/<username>/.cpa_state/`, including:
    - Query patterns and conversation themes
    - Memory access patterns
    - Hot functions for optimization
    - Compilation history

3.  **Data Preparation:** The `prepare_data.py` script reads all of these individual insight files and compiles them into a single `train.jsonl` file. This file is formatted specifically for instruction-based fine-tuning with tools like `llama.cpp`.

## Fine-Tuning Workflow

1.  **Interact with Jenova:** Use the AI normally. The more you interact and the more topics you cover, the more insight files will be generated. The CPA continuously learns patterns in the background.

2.  **Prepare the Data:** Once you have a sufficient number of insights (e.g., 50+), run the preparation script from the project root:

    ```bash
    source venv/bin/activate
    python finetune/prepare_data.py
    ```

    This will create a `train.jsonl` file in the `finetune/` directory.

3.  **Fine-Tune the Model:** Use your `train.jsonl` file with `llama.cpp`'s fine-tuning process. This is an advanced step that requires compiling `llama.cpp` and following their documentation. A typical command might look like this:

    ```bash
    # This is an example command and may require modification
    ./train-text-from-file --model-base models/phi-2.Q4_K_M.gguf --train-data finetune/train.jsonl --model-out models/jenova.gguf
    ```

4.  **Integrate the New Model:** Once the fine-tuning process is complete, you will have a new `jenova.gguf` file. Simply replace your old model file in the `models/` directory with this new one. The next time you launch Jenova AI, it will automatically use the new, smarter model.

## CPA State and Continuity

The Cognitive Process Accelerator maintains persistent learning across sessions. Even after fine-tuning:
- Query patterns continue to be tracked and optimized
- Memory access statistics are preserved
- JIT compilation improvements persist
- The AI maintains true continuity between sessions

Your fine-tuned model will benefit from both the baked-in knowledge (from fine-tuning) and the continuous learning (from CPA state).