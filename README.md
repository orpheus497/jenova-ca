# Jenova Cognitive Architecture: A Technical Deep Dive

## 1. Introduction & Philosophy

Jenova AI represents a leap forward in creating truly interactive and adaptive AI assistants. It is not merely a question-and-answer machine but a **learning entity** built upon a robust, pragmatic cognitive architecture. The core philosophy behind Jenova is that an AI should not be a static tool but a dynamic partner that learns, evolves, and adapts to its user over time.

This is achieved through a set of interconnected systems that mimic aspects of human cognition: a multi-layered memory, a reflective process for generating new knowledge, and a mechanism for integrating that knowledge back into its core being. This document serves as the definitive technical guide for developers, researchers, and enthusiasts who wish to understand, use, and extend the Jenova AI.

**Creator:** The Jenova Cognitive Architecture (JCA) was designed and developed by **orpheus497**.

## 2. The Jenova Advantage: A Superior Cognitive Architecture

### 2.1. Beyond Statelessness: The Problem with General Systems

Most consumer-facing AI systems operate on a **stateless, request-response** model. They are incredibly powerful at in-context learning and reasoning, but each interaction is largely independent of the last. This leads to several fundamental limitations:

*   **Amnesia:** The AI has no persistent memory of past conversations. It cannot remember your preferences, previous questions, or the context of your work. Every chat starts from a blank slate.
*   **Inability to Learn:** Corrections you make or new information you provide are only retained for the current session. The underlying model never truly learns or improves from user interaction.
*   **Inconsistent Persona:** The AI's personality can drift or be easily manipulated because it lacks a stable, memory-grounded identity.
*   **Reactive, Not Proactive:** These systems can only answer direct questions. They cannot reflect on past dialogues to draw novel conclusions or develop a deeper understanding of a topic over time.

### 2.2. The JCA Solution: A Unified, Learning Architecture

The Jenova Cognitive Architecture (JCA) is explicitly designed to overcome these limitations. It wraps a powerful Large Language Model (LLM) in a structured framework that provides memory, reflection, and a mechanism for true, persistent learning. It transforms the LLM from a brilliant but amnesiac calculator into a cohesive, evolving intelligence.

### 2.3. The Power of the Cognitive Cycle

The "Retrieve, Plan, Execute, Reflect" cycle is the engine of the JCA and the primary driver of its superiority.

*   **Grounded Responses:** By forcing the AI to **Retrieve** from its memory *before* acting, the JCA ensures that responses are not just plausible-sounding text. They are grounded in established facts, past conversations, and learned insights. This dramatically reduces confabulation (hallucination) and increases the relevance and accuracy of its output.
*   **Deliberate Action:** The **Plan** step introduces a moment of metacognition. The AI must first reason about *how* to answer the query. This internal monologue, while hidden from the user, results in a more structured and logical final response. It prevents the AI from taking conversational shortcuts and encourages a more methodical approach to problem-solving.

### 2.4. Memory as the Foundation for Identity and Growth

Jenova's multi-layered memory system is the bedrock of its identity. It is the difference between playing a character and *having* a character.

*   **Continuity of Self:** The `EpisodicMemory` gives Jenova a personal history with the user. It can refer to past conversations, understand recurring themes, and build a genuine rapport. The AI is no longer a stranger every time you open the terminal.
*   **A Worldview:** The `SemanticMemory` and `ProceduralMemory`, combined with the dynamically growing `InsightMemory`, form the AI's worldview. This knowledge base is prioritized over the LLM's base training data, allowing for the development of a unique, personalized knowledge set that reflects its experiences.

### 2.5. The Self-Correction and Evolution Loop: True Learning

This is the most powerful and defining feature of the JCA. The cycle of **Reflection -> Insight Generation -> Fine-Tuning** constitutes a true learning loop that is absent in general systems.

1.  **Experience (`Reflect`):** The AI has a conversation and gains experience.
2.  **Internalization (`Insight Generation`):** It reflects on that experience and internalizes the key takeaways as structured, atomic insights. This is analogous to a human consolidating short-term memories into long-term knowledge.
3.  **Integration (`Fine-Tuning`):** The fine-tuning process takes these internalized insights and integrates them into the very fabric of the neural network. The learned knowledge is no longer just data to be retrieved; it becomes part of the AI's intuition.

This loop creates a system that doesn't just get more knowledgeable; it gets **smarter**. It adapts its core reasoning processes based on its unique experiences, evolving into an assistant that is perfectly tailored to its user.

## 3. Core Features Explained

### 3.1. The Cognitive Cycle: Think, Plan, Execute, Reflect

The heart of Jenova is its cognitive cycle, a continuous loop that drives its behavior.

1.  **Retrieve:** When the user provides input, the `CognitiveEngine` first queries its memory systems (`Episodic`, `Semantic`, `Procedural`, and `Insight`) to gather relevant context.
2.  **Plan:** The engine then formulates a step-by-step internal plan. This plan is generated by the LLM itself, based on the user's query and the retrieved context. This ensures that the AI's actions are deliberate and grounded.
3.  **Execute:** The plan is then executed. The `CognitiveEngine` synthesizes the retrieved context, the conversation history, and the generated plan into a final prompt, instructing the LLM to generate the user-facing response.
4.  **Reflect:** Periodically, the `CognitiveEngine` enters a reflective state. It analyzes recent conversation history to identify novel conclusions or key takeaways. This process, detailed in the `generate_insight_from_history` method, is the cornerstone of Jenova's learning ability. This happens every 5 turns.

### 3.2. Reflective Insight Engine

The Insight Engine is what allows Jenova to learn continuously. Periodically, the AI reflects on recent conversations to generate a new "insight." This makes the learning process more deliberate and ensures that new knowledge is captured thoughtfully. By default, this reflection occurs every 5 conversational turns.

-   **Generation:** Every 5 turns, the `CognitiveEngine` prompts the LLM to analyze the last 8 turns of conversation. It uses a detailed prompt with guidelines for what constitutes a "high-quality insight" to extract a single, significant takeaway. The result is formatted as a JSON object.
-   **Storage:** The `InsightManager` saves this JSON object to a file in the `~/.jenova-ai/insights/` directory.
-   **Retrieval:** These insights are immediately available for retrieval in subsequent cognitive cycles, allowing the AI's knowledge base to grow with every step of the conversation.

### 3.3. Self-Optimizing Context Window

To maximize performance, the `LLMInterface` dynamically configures the context window (`n_ctx`). On startup, it performs a "dry run" to load the model's metadata and read its maximum supported context length. This value is then used to override the default `context_size` set in `main_config.yaml`, ensuring the AI always uses the largest possible context window the model was trained on, without requiring manual configuration.

### 3.4. Multi-Layered Long-Term Memory

Jenova's memory is not a monolith. It's a sophisticated, multi-layered system managed by `ChromaDB`, a vector database.

-   **Episodic Memory (`EpisodicMemory`):** Stores a turn-by-turn history of conversations. This allows Jenova to recall specific past interactions.
-   **Semantic Memory (`SemanticMemory`):** Stores factual knowledge. This is pre-loaded from the `RAG.md` document and is intended for static, foundational knowledge.
-   **Procedural Memory (`ProceduralMemory`):** Stores "how-to" information and instructions. This is where the AI would store knowledge about performing specific tasks.
-   **Insight Memory (`InsightManager`):** While not a ChromaDB instance, the collection of saved insight files acts as a fourth, highly dynamic memory layer.

### 3.5. In-Cycle Fine-Tuning Framework

Jenova is designed for continuous improvement. The insights generated during its operation can be used to fine-tune the base model itself.

-   **Data Preparation (`prepare_data.py`):** This script gathers all the `.json` insight files from `~/.jenova-ai/insights/` and transforms them into a `train.jsonl` file. This file is formatted in the instruction-following format required by `llama.cpp`.
-   **Model Evolution:** By fine-tuning the base model with this data, the learned insights become a permanent part of the AI's knowledge, leading to a more intelligent and personalized assistant.

### 3.6. Tool Use: File Generation

Jenova has the built-in capability to use tools. The primary example is the `WRITE_FILE` tool.

-   **Detection:** The `FileTools.handle_tool_request` method uses a regular expression to scan the AI's generated response for a special `<TOOL:WRITE_FILE(...)>` tag.
-   **Execution:** If the tag is found, the tool extracts the specified file path and content, writes the file to the `~/.jenova-ai/generated_files/` directory, and then replaces the tool tag in the final response with a confirmation message for the user.

## 4. Installation and Setup: A Detailed Guide

The `install.sh` script provides a "scorched earth" installation, ensuring a clean and reliable setup.

1.  **Prerequisites:**
    *   A Linux-based operating system.
    *   `git` and `python3` (version 3.10 or higher) must be installed.
    *   A C++ compiler (like `g++`) is required for `llama-cpp-python`.

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/orpheus497/jenova-ai.git
    cd jenova-ai
    ```

3.  **Download a Model:**
    Jenova requires a GGUF-formatted model. You must download one and place it in a `models/` directory. For example:
    ```bash
    mkdir models
    wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf
    ```

4.  **Run the Installation Script:**
    ```bash
    ./install.sh
    ```
    This script performs the following actions:
    -   Creates a Python virtual environment in `./venv`.
    -   Activates the environment.
    -   Upgrades `pip`.
    -   **Crucially, it uninstalls any previous version of `jenova-ai` to prevent conflicts.**
    -   It purges all `__pycache__` directories to ensure a clean build.
    -   It installs the project in "editable" mode (`pip install -e .`), which means changes to the source code are immediately reflected without needing to reinstall.

5.  **Activate and Run:**
    -   **Activation:** You must always activate the virtual environment before running the AI.
        ```bash
        source venv/bin/activate
        ```
    -   **Execution:** The `setup.py` file defines a console script entry point, so you can run the AI using a single command:
        ```bash
        jenova
        ```

## 5. Usage and Commands

Interaction with Jenova is primarily through natural language. The `TerminalUI` provides a clean, interactive command-line interface that now recognizes the user logged into the terminal.

-   **User Input:** Simply type your message and press Enter. The UI will display your username alongside your query.
-   **Exiting:** To quit the application, type `exit` and press Enter.

In addition to standard conversation, you can use the following commands to trigger specific cognitive functions:

-   `/insight`: Triggers the AI to analyze the entire history of the current conversation and generate new insights based on it.
-   `/reflect`: Instructs the AI to review all of its previously generated insights and attempt to synthesize a new, higher-level "meta-insight" from them.
-   `/memory-insight`: Prompts the AI to perform a broad search of its long-term memory and generate a new insight from the retrieved context.

The UI uses the `rich` library to provide formatted, color-coded output, distinguishing between system messages, user input, and AI responses.

## 6. The Fine-Tuning Workflow in Detail

This is an advanced process for permanently integrating Jenova's learned knowledge.

1.  **Accumulate Insights:** Use Jenova AI extensively. The more varied and in-depth your conversations, the more high-quality insights will be generated in `~/.jenova-ai/insights/`. Aim for at least 50-100 insights before attempting to fine-tune.

2.  **Prepare the Training Data:**
    -   Ensure your virtual environment is active: `source venv/bin/activate`
    -   Run the preparation script from the project root:
        ```bash
        python finetune/prepare_data.py
        ```
    -   This script will find all `.json` files in the insights directory, parse them, and create a `finetune/train.jsonl` file. Each line in this file will be a JSON object representing a single training example, ready for `llama.cpp`.

3.  **Fine-Tune with `llama.cpp`:**
    -   This step requires a compiled version of `llama.cpp`. You must follow the `llama.cpp` documentation to build it.
    -   The exact command will depend on your `llama.cpp` version and your specific needs, but it will look something like this:
        ```bash
        # This is an example command and may require modification
        ./train-text-from-file \
          --model-base /path/to/your/base-model.gguf \
          --train-data /path/to/jenova-ai/finetune/train.jsonl \
          --model-out /path/to/jenova-ai/models/jenova-finetuned.gguf \
          --lora-out /path/to/your/lora-adapter.bin # Optional: for LoRA tuning
        ```

4.  **Integrate the New Model:**
    -   After the fine-tuning process completes, you will have a new model file (e.g., `jenova-finetuned.gguf`).
    -   Rename or remove your old model in the `models/` directory.
    -   Place the new, fine-tuned model in the `models/` directory.
    -   The `LLMInterface.find_model_path()` function is designed to find the first `.gguf` file in the `models` directory. The next time you launch `jenova`, it will automatically load your new, more intelligent model.

## 7. Codebase and Configuration Overview

### 7.1. Project Structure

```
/
├── finetune/             # Scripts for model fine-tuning
├── models/               # Where you place your .gguf model files
├── src/
│   └── jenova/
│       ├── cognitive_engine/ # The core "thinking" loop
│       ├── config/           # Default YAML configuration files
│       ├── docs/             # RAG documents for semantic memory
│       ├── insights/         # Manages saving and loading learned insights
│       ├── memory/           # Manages the different memory types (ChromaDB)
│       ├── tools/            # Handlers for tool usage (e.g., file writing)
│       ├── ui/               # The terminal user interface
│       ├── utils/            # Utility scripts and patches
│       ├── __init__.py
│       ├── llm_interface.py  # Handles all interaction with llama.cpp
│       └── main.py           # Main application entry point
├── install.sh            # Installation script
├── requirements.txt      # Python dependencies
└── setup.py              # Package definition and entry point
```

### 7.2. Configuration Files

Jenova's behavior is controlled by two YAML files in `src/jenova/config/`.

#### `main_config.yaml`

This file controls the technical parameters of the AI.

-   **`hardware`**:
    -   `threads`: Number of CPU threads to use.
    -   `gpu_layers`: Number of model layers to offload to the GPU.
    -   `mlock`: Whether to lock the model in memory (prevents swapping).
-   **`model`**:
    -   `embedding_model`: The sentence-transformer model to use for creating vector embeddings for memory search.
    -   `context_size`: The default context window size (will be overridden by the model's metadata if possible).
    -   `temperature`: Controls the "creativity" of the LLM. Lower is more deterministic.
    -   `top_p`: Nucleus sampling parameter.
-   **`memory`**:
    -   `..._db_path`: Paths to the ChromaDB databases. These are relative to the user's data directory (`~/.jenova-ai/`).
    -   `reflection_interval`: How often to run the reflection process (in conversation turns).

#### `persona.yaml`

This file defines the AI's personality and core directives.

-   **`identity`**:
    -   `name`: The AI's name.
    -   `creator`: The creator's name.
    -   `origin_story`: A brief backstory.
    -   `type`: A description of the AI's nature.
-   **`directives`**: A list of rules the AI must follow. These are injected into the system prompt.
-   **`initial_facts`**: A list of foundational facts that are loaded into the AI's memory on first run.
