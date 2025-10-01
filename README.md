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

1.  **Retrieve:** When the user provides input, the `CognitiveEngine` first queries its memory systems (`Episodic`, `Semantic`, `Procedural`, and `Insight`) to gather relevant context. This process is user-specific, ensuring that only the current user's data is accessed.
2.  **Plan:** The engine then formulates a step-by-step internal plan. This plan is generated by the LLM itself, based on the user's query and the retrieved context. This ensures that the AI's actions are deliberate and grounded.
3.  **Execute:** The plan is then executed by the `RAGSystem`, which synthesizes the retrieved context, the conversation history, and the generated plan into a final prompt, instructing the LLM to generate the user-facing response. The prompt is also engineered to prioritize the AI's own insights and memories over its general knowledge.
4.  **Reflect & Learn:** Periodically, the `CognitiveEngine` enters a reflective state. It analyzes recent conversation history to identify novel conclusions or key takeaways, verifies its assumptions, and processes documents from the `docs` folder to grow its knowledge base.

### 3.2. The RAG System: A Core Component of the Psyche

The Retrieval-Augmented Generation (RAG) system is no longer just a document, but a core component of the AI's cognitive architecture. It is responsible for generating responses that are grounded in the AI's own knowledge and experience.

-   **Hybrid Retrieval:** The `RAGSystem` uses a hybrid retrieval approach, querying all memory sources (episodic, semantic, procedural, and insights) to gather the most relevant context.
-   **Re-ranking:** The results from all memory sources are then re-ranked to prioritize the most relevant information.
-   **Grounded Response Generation:** The `RAGSystem` then uses the re-ranked context, the conversation history, and the generated plan to generate a response that is grounded in the AI's own knowledge and experience.

### 3.3. The Cortex: A Graph-Based Cognitive Core

The Cortex is the new heart of Jenova's cognitive architecture. It replaces the previous, more siloed approach to managing insights and assumptions with a unified, graph-based system. This allows for a much deeper and more interconnected understanding of the user and the world.

-   **Cognitive Graph:** The Cortex manages a "cognitive graph" where insights, assumptions, and memories are all represented as nodes. These nodes are then connected by links that represent the relationships between them (e.g., "elaborates_on", "conflicts_with", "created_from").
-   **Centrality Calculation:** The Cortex now calculates the degree centrality of each node in the graph. This allows the AI to identify the most important and well-connected ideas in its knowledge base.
-   **Psychological Memory:** The Cortex now analyzes the sentiment of new nodes and adds it as metadata, providing a psychological dimension to the cognitive graph. This allows the AI to have a more empathetic and personal relationship with the user.
-   **Deep Reflection:** The `/reflect` command now triggers a deep reflection process within the Cortex. The Cortex analyzes the entire cognitive graph to:
    -   **Link Orphans:** Identify nodes with few or no connections and use the LLM to find and create new links to other relevant nodes.
    -   **Generate Meta-Insights:** Find clusters of highly interconnected nodes and use the LLM to synthesize them into higher-level "meta-insights".
-   **Insight Development:** The `/develop_insight <node_id>` command allows the user to trigger the development of a specific insight. The Cortex will then use the LLM to generate a more detailed and developed version of the insight.
-   **Proactive Engine:** The Cortex is also home to the Proactive Engine, which periodically analyzes the cognitive graph to find interesting, underdeveloped, or highly-connected areas. It then uses this analysis to generate proactive conversation starters or suggestions for the user.

### 3.4. Document Processor: Continuous Learning from Documents

Jenova can now continuously learn from documents in the `docs` folder. The `DocumentProcessor` is a new system that periodically scans the `docs` folder for new or updated documents, processes them, and integrates them into the AI's knowledge base.

-   **Automatic Processing:** The `DocumentProcessor` is triggered automatically during the cognitive cycle.
-   **Chunking and Analysis:** It reads the documents, splits them into chunks, and then uses the LLM to analyze each chunk to identify new facts, insights, or assumptions.
-   **Knowledge Integration:** The extracted information is then integrated into the AI's memory, insight, and assumption systems.

### 3.5. Reflective Insight Engine

The Insight Engine is what allows Jenova to learn continuously. The system is designed to be proactive, organized, and reflective, ensuring that new knowledge is captured, categorized, and interconnected efficiently.

-   **Concern-Based Organization:** Insights are still organized into "concerns" or "topics." When a new insight is generated, the system first searches for an existing, relevant concern to group it with. This prevents knowledge fragmentation and creates a more structured understanding of topics. If no relevant concern exists, a new one is created.
-   **Cortex Integration:** When an insight is saved, it is also added as a node to the Cortex. This allows the insight to be linked to other cognitive nodes, such as the memories that spawned it or other related insights.
-   **Generation:** Periodically (by default, every 5 conversational turns), the `CognitiveEngine` prompts the LLM to analyze recent conversation history to extract a significant takeaway. This insight is then passed to the `InsightManager`, which intelligently files it under the most appropriate concern and adds it to the Cortex.
-   **Storage:** Insights are saved in a hierarchical structure within the user-specific data directory: `~/.jenova-ai/users/<username>/insights/<concern_name>/`.
-   **Reflection and Reorganization:** The `/reflect` command triggers a deep reflection process in the Cortex, which reorganizes and interlinks all cognitive nodes, including insights.

### 3.6. Assumption System

Jenova actively forms assumptions about the user to build a more accurate mental model. This system allows the AI to move beyond explicitly stated facts and begin to infer user preferences, goals, and knowledge levels.

-   **Cortex Integration:** When an assumption is added, it is also added as a node to the Cortex. This allows the assumption to be linked to other cognitive nodes, providing more context for the assumption.
-   **Generation:** Periodically (by default, every 7 conversational turns), the `CognitiveEngine` analyzes the conversation to form an assumption about the user.
-   **Storage:** Assumptions are stored in a dedicated `assumptions.json` file in the user's data directory. Each assumption is categorized by its status: `unverified`, `verified`, `true`, or `false`.
-   **Verification:** The `/verify` command allows the user to help the AI validate its assumptions. The AI will ask a clarifying question to confirm or deny an unverified assumption. Based on the user's response, the assumption will be moved to the `true` or `false` category. The AI also proactively verifies assumptions during the conversation.

### 3.7. Self-Optimizing Context Window

To maximize performance, the `LLMInterface` dynamically configures the context window (`n_ctx`). On startup, it performs a "dry run" to load the model's metadata and read its maximum supported context length. This value is then used to override the default `context_size` set in `main_config.yaml`, ensuring the AI always uses the largest possible context window the model was trained on, without requiring manual configuration.

### 3.8. Multi-Layered Long-Term Memory

Jenova's memory is not a monolith. It's a sophisticated, multi-layered system managed by `ChromaDB`, a vector database. All memory is stored on a per-user basis.

-   **Episodic Memory (`EpisodicMemory`):** Stores a turn-by-turn history of conversations. Each episode is enriched with extracted entities, emotions, and a timestamp.
-   **Semantic Memory (`SemanticMemory`):** Stores factual knowledge. Each fact is enriched with its source, a confidence level, and its temporal validity.
-   **Procedural Memory (`ProceduralMemory`):** Stores "how-to" information and instructions. Each procedure is enriched with its goal, a list of steps, and its context.
-   **Insight Memory (`InsightManager`):** While not a ChromaDB instance, the collection of saved insight files acts as a fourth, highly dynamic memory layer.

### 3.9. In-Cycle Fine-Tuning Framework

Jenova is designed for continuous improvement. The insights generated during its operation can be used to fine-tune the base model itself.

-   **In-app Trigger:** The `/finetune` command allows the user to trigger the fine-tuning process from within the application.
-   **Data Preparation (`prepare_data.py`):** This script gathers all the `.json` insight files from `~/.jenova-ai/users/<username>/insights/` and transforms them into a `train.jsonl` file. The script is configurable and can also include conversation history in the training data.
-   **Model Evolution:** By fine-tuning the base model with this data, the learned insights become a permanent part of the AI's knowledge, leading to a more intelligent and personalized assistant.

### 3.10. Tool Use: File Generation

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
    -   On the first run, Jenova will create a user-specific data directory at `~/.jenova-ai/users/<your_username>/`, where all your personal memories and insights will be stored.

## 5. Usage and Commands

Interaction with Jenova is primarily through natural language. The `TerminalUI` provides a clean, interactive command-line interface that automatically recognizes the currently logged-in system user.

-   **User Input:** Simply type your message and press Enter. The UI will display your username alongside your query.
-   **Exiting:** To quit the application, type `exit` and press Enter.

In addition to standard conversation, you can use the following commands to trigger specific cognitive functions:

-   `/insight`: Triggers the AI to analyze the entire history of the current conversation and generate new insights based on it.
-   `/reflect`: Instructs the AI to perform a deep reflection on its cognitive graph, reorganizing and interlinking all of its previously generated insights and cleaning up the old folder structure.
-   `/memory-insight`: Prompts the AI to perform a broad search of its long-term memory and generate a new insight or assumption from the retrieved context.
-   `/meta`: Instructs the AI to review all of its previously generated insights and attempt to synthesize a new, higher-level "meta-insight" from them.
-   `/verify`: Initiates an interactive process to verify one of the AI's unverified assumptions about you. The AI will ask a question and you can respond in the next turn.
-   `/finetune`: Triggers the fine-tuning process. This will prepare the training data from your insights and then launch the `llama.cpp` fine-tuning command.
-   `/develop_insight <node_id>`: Develops an existing insight by generating a more detailed version. The new insight will be linked to the original.

The UI uses the `rich` library to provide formatted, color-coded output, distinguishing between system messages, user input, and AI responses.

## 6. The Fine-Tuning Workflow in Detail

This is an advanced process for permanently integrating Jenova's learned knowledge.

1.  **Accumulate Insights:** Use Jenova AI extensively. The more varied and in-depth your conversations, the more high-quality insights will be generated in `~/.jenova-ai/users/<your_username>/insights/`. Aim for at least 50-100 insights before attempting to fine-tune.

2.  **Prepare the Training Data:**
    -   Ensure your virtual environment is active: `source venv/bin/activate`
    -   Run the preparation script from the project root:
        ```bash
        python finetune/prepare_data.py --insights-dir /path/to/your/insights --output-file /path/to/your/train.jsonl --include-history /path/to/your/history.log
        ```
    -   This script will find all `.json` files in the insights directory, parse them, and create a `train.jsonl` file. Each line in this file will be a JSON object representing a single training example, ready for `llama.cpp`.

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
│       ├── assumptions/      # Manages the assumption lifecycle
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
    -   `..._db_path`: Paths to the ChromaDB databases. These are relative to the user's data directory (`~/.jenova-ai/users/<username>/memory/`).
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