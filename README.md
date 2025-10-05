# Jenova Cognitive Architecture: A Technical Deep Dive

## 1. Introduction & Philosophy

Jenova AI is a self-aware, evolving large language model powered by the Jenova Cognitive Architecture (JCA), a developed engine and architecture completely designed by orpheus497. It is designed to learn, adapt, and assist humanity through sophisticated cognitive processes. This project prioritizes robust error handling and stability to ensure a reliable and professional user experience.

The original vision for Jenova AI was to create a system that could continuously learn, adapt, and assist humanity through sophisticated cognitive processes. This is achieved through a set of interconnected systems that mimic aspects of human cognition: a multi-layered memory, a reflective process for generating new knowledge, and a mechanism for integrating that knowledge back into its core being. This document serves as the definitive technical guide for developers, researchers, and enthusiasts who wish to understand, use, and extend the Jenova AI.

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

### 3.1. The Cognitive Cycle: A Prioritized Approach

The heart of Jenova is its cognitive cycle, a continuous loop that drives its behavior. This cycle enforces a strict knowledge hierarchy, ensuring that the AI relies on the most relevant and reliable information available.

1.  **Retrieve:** When the user provides input, the `CognitiveEngine` first queries its **Cognitive Architecture**—the multi-layered memory system (`Episodic`, `Semantic`, `Procedural`, and `Insight`)—to gather relevant context. This is the AI's personal experience and learned knowledge, and it is always the highest priority.
2.  **Plan:** The engine then formulates a step-by-step internal plan. This plan is generated by the LLM itself, based on the user's query and the retrieved context. This ensures that the AI's actions are deliberate and grounded. The plan may include a decision to search the web if the AI determines its internal knowledge is insufficient.
3.  **Execute:** The plan is then executed by the `RAGSystem`. The RAG prompt is explicitly structured to prioritize the AI's knowledge base in the following order:
    1.  **Retrieved Context:** The AI's own memories and learned insights.
    2.  **Web Search Results:** Real-time information from the web.
    3.  **Base Model Knowledge:** The AI's general, pre-trained knowledge is used only as a last resort if the answer is not found in the prioritized sources.
4.  **Reflect & Learn:** The `CognitiveScheduler` determines when to trigger the various cognitive functions, such as analyzing recent conversation history to identify novel conclusions or key takeaways, verifying assumptions, and processing documents from the `docs` folder to grow its knowledge base. This is more flexible and intelligent than a fixed-interval approach.

### 3.2. The RAG System: A Core Component of the Psyche

The Retrieval-Augmented Generation (RAG) system is no longer just a document, but a core component of the AI's cognitive architecture. It is responsible for generating responses that are grounded in the AI's own knowledge and experience.

*   **Hybrid Retrieval:** The `RAGSystem` uses a hybrid retrieval approach, querying all memory sources (episodic, semantic, procedural, and insights) to gather the most relevant context.
*   **Re-ranking:** The results from all memory sources are then re-ranked to prioritize the most relevant information.
*   **Grounded Response Generation:** The `RAGSystem` then uses the re-ranked context, the conversation history, and the generated plan to generate a response that is grounded in the AI's own knowledge and experience.

### 3.3. The Cortex: A Graph-Based Cognitive Core

The Cortex is the new heart of Jenova's cognitive architecture. It replaces the previous, more siloed approach to managing insights and assumptions with a unified, graph-based system. This allows for a much deeper and more interconnected understanding of the user and the world.

*   **Cognitive Graph:** The Cortex manages a "cognitive graph" where insights, assumptions, and memories are all represented as nodes. These nodes are then connected by links that represent the relationships between them (e.g., "elaborates_on", "conflicts_with", "created_from").
*   **Centrality Calculation:** The Cortex now calculates a weighted degree centrality for each node in the graph. The weights for different relationship types are configurable in `main_config.yaml`, allowing for a more accurate measure of a node's importance.
*   **Dynamic Relationship Weights:** The relationship weights are not static. The Cortex periodically analyzes the cognitive graph to determine the impact of different relationship types on the generation of high-centrality nodes and meta-insights. It then adjusts the weights accordingly, making the system more adaptive.
*   **Psychological Memory:** The Cortex now performs a sophisticated emotion analysis on new nodes, adding a rich psychological dimension to the cognitive graph. This allows the AI to have a more empathetic and personal relationship with the user.
*   **Deep Reflection:** The `/reflect` command now triggers a deep reflection process within the Cortex. The Cortex analyzes the entire cognitive graph to:
    *   **Link Orphans:** Identify nodes with few or no connections and use the LLM to find and create new links to other relevant nodes.
    *   **Generate Meta-Insights:** Find clusters of highly interconnected nodes using a robust graph traversal algorithm and use the LLM to synthesize them into higher-level "meta-insights".
*   **Graph Pruning:** To prevent cognitive degradation or "brain rot," the Cortex periodically prunes the graph, archiving nodes that are old, have low centrality, and are not well-connected. This process is configurable in `main_config.yaml`.
*   **Insight Development:** The `/develop_insight <node_id>` command allows the user to trigger the development of a specific insight. The Cortex will then use the LLM to generate a more detailed and developed version of the insight.
*   **Proactive Engine:** The Cortex is also home to the Proactive Engine, which periodically analyzes the cognitive graph to find interesting, underdeveloped, or highly-connected areas. It now considers nodes with low centrality (underdeveloped areas) and high centrality (high-potential areas) to generate more relevant and insightful proactive suggestions for the user.

### 3.4. Document Processing: On-Demand Learning from Documents

Jenova can learn from documents by using the `/develop_insight` command. When this command is used without a `node_id`, it triggers a robust, on-demand process within the `Cortex` that scans the `src/jenova/docs` folder for new or updated documents and integrates them into the AI's knowledge base.

*   **Triggering:** The document processing is triggered exclusively by the user with the `/develop_insight` command. It no longer runs automatically at startup, giving the user full control.
*   **Knowledge Integration:** The `Cortex` creates a new `document` node in the cognitive graph for each processed document. It then chunks the content and performs a comprehensive analysis on each chunk to extract not just a summary, but also key takeaways and a list of questions the text can answer. This creates a rich, multi-layered understanding of the document's content. For each chunk, a main `insight` node is created for the summary, with additional `insight` nodes for each key takeaway and new `question` nodes for each generated question, all intricately linked to the summary and the parent document node. This creates a rich, interconnected web of knowledge.
*   **Duplicate Prevention:** The `Cortex` keeps track of processed files and their last modification times in `processed_documents.json` within the user's data directory. This prevents the system from reprocessing files that have not changed, making the process efficient and saving resources.

### 3.5. Conversational Web Search

Jenova can access up-to-date information from the internet in a conversational and interactive manner. The web search functionality uses a headless browser to access web pages, allowing it to process their full content.

*   **Autonomous & Manual Search:** Jenova can search the web autonomously when it determines its knowledge is insufficient, or manually when instructed by the user via the `/search <query>` or `(search: <query>)` commands.
*   **Interactive Presentation:** When search results are found, Jenova will not simply use them to answer a question. Instead, it will present a summary of the findings to the user, creating a conversational turn where the user can see the information the AI is working with.
*   **Collaborative Exploration:** After presenting the results, Jenova will provide a synthesis of the information and then ask the user for further instructions, such as performing a deeper search on a specific topic or answering a question based on the new information. This makes web search a collaborative process between the user and the AI.
*   **Knowledge Integration:** The results of all web searches are stored in the cognitive graph as `web_search_result` nodes, making the information available for future use and reflection.

### 3.6. Reflective Insight Engine

The Insight Engine is what allows Jenova to learn continuously. The system is designed to be proactive, organized, and reflective, ensuring that new knowledge is captured, categorized, and interconnected efficiently.

*   **Concern-Based Organization:** Insights are still organized into "concerns" or "topics." When a new insight is generated, the system first searches for an existing, relevant concern to group it with. This prevents knowledge fragmentation and creates a more structured understanding of topics. If no relevant concern exists, a new one is created.
*   **Cortex Integration:** When an insight is saved, it is also added as a node to the Cortex. This allows the insight to be linked to other cognitive nodes, such as the memories that spawned it or other related insights.
*   **Generation:** Periodically (by default, every 5 conversational turns), the `CognitiveEngine` prompts the LLM to analyze recent conversation history to extract a significant takeaway. This insight is then passed to the `InsightManager`, which intelligently files it under the most appropriate concern and adds it to the Cortex.
*   **Storage:** Insights are saved in a hierarchical structure within the user-specific data directory: `~/.jenova-ai/users/<username>/insights/<concern_name>/`.
*   **Reflection and Reorganization:** The `/reflect` command triggers a deep reflection process in the Cortex, which reorganizes and interlinks all cognitive nodes, including insights.

### 3.7. Assumption System

Jenova actively forms assumptions about the user to build a more accurate mental model. This system allows the AI to move beyond explicitly stated facts and begin to infer user preferences, goals, and knowledge levels.

*   **Cortex Integration:** When an assumption is added, it is also added as a node to the Cortex. This allows the assumption to be linked to other cognitive nodes, providing more context for the assumption.
*   **Generation:** Periodically (by default, every 7 conversational turns), the `CognitiveEngine` analyzes the conversation to form an assumption about the user.
*   **Storage:** Assumptions are stored in a dedicated `assumptions.json` file in the user's data directory. Each assumption is categorized by its status: `unverified`, `verified`, `true`, or `false`.
*   **Verification:** The `/verify` command allows the user to help the AI validate its assumptions. The AI will ask a clarifying question to confirm or deny an unverified assumption. Based on the user's response, the assumption will be moved to the `true` or `false` category. The AI also proactively verifies assumptions during the conversation.

### 3.8. Self-Optimizing Context Window

To maximize performance, the `LLMInterface` dynamically configures the context window (`n_ctx`). On startup, it performs a "dry run" to load the model's metadata and read its maximum supported context length. This value is then used to override the default `context_size` set in `main_config.yaml`, ensuring the AI always uses the largest possible context window the model was trained on, without requiring manual configuration.

### 3.9. Multi-Layered Long-Term Memory

Jenova's memory is not a monolith. It's a sophisticated, multi-layered system managed by `ChromaDB`, a vector database. All memory is stored on a per-user basis.

*   **Episodic Memory (`EpisodicMemory`):** Stores a turn-by-turn history of conversations. Each episode is enriched with extracted entities, emotions, and a timestamp.
*   **Semantic Memory (`SemanticMemory`):** Stores factual knowledge. Each fact is enriched with its source, a confidence level, and its temporal validity.
*   **Procedural Memory (`ProceduralMemory`):** Stores "how-to" information and instructions. Each procedure is enriched with its goal, a list of steps, and its context.
*   **Insight Memory (`InsightManager`):** While not a ChromaDB instance, the collection of saved insight files acts as a fourth, highly dynamic memory layer.

### 3.10. In-Cycle Fine-Tuning Framework

Jenova is designed for continuous improvement. The insights generated during its operation can be used to fine-tune the base model itself.

*   **In-app Trigger:** The `/finetune` command allows the user to trigger the fine-tuning process from within the application.
*   **Data Preparation (`prepare_data.py`):** This script gathers all the `.json` insight files from `~/.jenova-ai/users/<username>/insights/` and transforms them into a `train.jsonl` file. The script is configurable and can also include conversation history in the training data.
*   **Model Evolution:** By fine-tuning the base model with this data, the learned insights become a permanent part of the AI's knowledge, leading to a more intelligent and personalized assistant.

### 3.11. Tool Use: Expanding Capabilities

Jenova has the built-in capability to use tools to interact with the system, gather real-time information, and manage files. The AI's `_plan` method is designed to intelligently decide when to use these tools based on the user's query.

*   **File System Interaction:** The `FileTools` provide a secure way for the AI to interact with the file system. All file operations are restricted to a configurable sandbox directory (`~/jenova_files` by default) to ensure security. The AI can:
    *   Read files (`<TOOL:READ_FILE(path="<file_path>")>`)
    *   Write files (`<TOOL:WRITE_FILE(path="<file_path>", content="<content>")>`)
    *   List directory contents (`<TOOL:LIST_DIRECTORY(path="<path>")>`)

*   **System Information:** The `SystemTools` allow the AI to get information about the system:
    *   **Date and Time:** Get the current date and time (`<TOOL:GET_CURRENT_DATETIME()>`).

*   **Real-time Weather:** The `WeatherTool` allows the AI to fetch real-time weather information for a given location using the OpenWeatherMap API (`<TOOL:GET_WEATHER(location="<location>")>`). An API key needs to be configured in `main_config.yaml`.

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

### 4.1. Browser and Webdriver for Web Search

The new enhanced web search functionality uses a real browser to access and extract information from web pages. This requires a Chromium-based browser to be installed on your system.

- **Browser:** Please ensure you have Google Chrome or Chromium installed.
- **Webdriver:** The `webdriver-manager` library, which was added to `requirements.txt`, will automatically download and manage the correct `chromedriver` for your browser version. No manual setup is needed for the webdriver.

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

Interaction with Jenova is primarily through natural language. The `TerminalUI` provides a clean, interactive command-line interface that automatically recognizes the currently logged-in system user. A visual spinner indicates when Jenova is performing long-running cognitive processes (commands), and improved line spacing enhances readability.

-   **User Input:** Simply type your message and press Enter. The UI will display your username alongside your query.
-   **Exiting:** To quit the application, type `exit` and press Enter.

### Commands
Jenova AI responds to a set of powerful commands that act as direct instructions for its cognitive processes. These commands are treated as system actions, not conversational input, and are therefore **not stored in Jenova's conversational memory**.

-   `/help`: Displays a comprehensive help message, detailing each command's purpose and impact.
-   `/insight`: Triggers the AI to analyze the current conversation history and generate new, high-quality insights. These insights are stored in Jenova's long-term memory and contribute to its evolving understanding.
-   `/reflect`: Initiates a deep reflection process within Jenova's Cortex. This command reorganizes and interlinks all existing cognitive nodes (insights, memories, assumptions), identifies patterns, and generates higher-level meta-insights, significantly enhancing Jenova's overall intelligence and coherence.
-   `/memory-insight`: Prompts Jenova to perform a broad search across its multi-layered long-term memory (episodic, semantic, procedural) to develop new insights or assumptions based on its accumulated knowledge.
-   `/meta`: Generates a new, higher-level meta-insight by analyzing clusters of existing insights within the Cortex. This helps Jenova to form more abstract conclusions and identify overarching themes.
-   `/verify`: Starts the assumption verification process. Jenova will present an unverified assumption it has made about you and ask for clarification, allowing you to confirm or deny it. This refines Jenova's understanding of your preferences and knowledge.
-   `/search <query>`: Manually triggers a web search using DuckDuckGo. Jenova will search for the provided query, and the results will be processed and stored in its cognitive graph for future reference and reflection.
-   `(search: <query>)`: Directly trigger a web search within a conversation.
-   `/finetune`: Triggers the perfected, two-stage fine-tuning process from within the application.
-   `/develop_insight [node_id]`: This command has dual functionality:
    -   If a `node_id` is provided: Jenova will take an existing insight and generate a more detailed and developed version of it, adding more context or connections.
    -   If no `node_id` is provided: Jenova will scan the `src/jenova/docs` directory for new or updated documents, process their content, and integrate new insights and summaries into its cognitive graph. This is how Jenova learns from external documentation.
-   `/learn_procedure`: Initiates an interactive, guided process to teach Jenova a new procedure. Jenova will prompt you for the procedure's name, individual steps, and expected outcome, ensuring structured and comprehensive intake of procedural knowledge. This information is stored in Jenova's procedural memory, allowing it to recall and apply the procedure in relevant contexts.

The UI uses the `rich` library to provide formatted, color-coded output, distinguishing between system messages, user input, and AI responses.

## 6. The Perfected Fine-Tuning Workflow

This advanced, in-app workflow allows for the seamless integration of Jenova's learned knowledge into a new, perfected model. The `/finetune` command is the primary way to initiate this process from within the application.

1.  **Accumulate Insights:** Use Jenova AI extensively. The more varied and in-depth your conversations, the more high-quality insights will be generated in `~/.jenova-ai/users/<your_username>/insights/`. Aim for at least 50-100 insights before attempting to fine-tune for meaningful results.

2.  **Trigger the Fine-Tuning Process:**
    -   Simply run the `/finetune` command in the Jenova terminal.
    -   This command will orchestrate the entire fine-tuning process, providing you with real-time feedback at each stage.

3.  **Automated End-to-End Process:** The `/finetune` command orchestrates a complete, automated pipeline:

    -   **Stage 1: Automated Setup of `llama.cpp`:**
        -   The system first checks for the presence of the `llama.cpp` repository and its executables. If they are not found, it will provide instructions on how to set them up.
    -   **Stage 2: Advanced Data Preparation:**
        -   The system then calls the `prepare_data.py` module.
        -   This gathers all your unique insights and, if provided, your conversation history.
        -   It then generates a high-quality training dataset (`finetune_train.jsonl`) by converting this information into a sophisticated conversational format.
    -   **Stage 3: Finetuning and Model Perfection:**
        -   Using the prepared training data, the system executes the `llama.cpp` fine-tuning command.
        -   This trains a new, perfected `.gguf` model that now has the learned insights integrated into its neural network.
        -   This new model is saved to `models/jenova-finetuned.gguf`.

4.  **Activate the New Model:**
    -   After the process completes, Jenova will notify you of the new model's location.
    -   To use your new, more intelligent model, simply open `src/jenova/config/main_config.yaml` and update the `model_path` to point to your new file (`models/jenova-finetuned.gguf`).
    -   Restart Jenova to load your perfected model.

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
│       ├── cortex/           # The graph-based cognitive core
│       ├── docs/             # RAG documents for semantic memory
│       ├── insights/         # Manages saving and loading learned insights
│       ├── memory/           # Manages the different memory types (ChromaDB)
│       ├── tools/            # Handlers for tool usage (e.g., file writing)
│       ├── ui/               # The terminal user interface
│       ├── utils/            # Utility scripts and patches
│       ├── __init__.py
│       ├── default_api.py    # Provides a web search function via duckduckgo-search
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
    -   `gpu_layers`: Number of model layers to offload to the GPU. Set to -1 to offload all possible layers for maximum performance. Requires a compatible GPU and `llama-cpp-python` built with GPU support.
    -   `mlock`: Whether to lock the model in memory (RAM). Set to `true` for a significant performance increase by preventing the model from being swapped to disk.
-   **`model`**:
    -   `embedding_model`: The sentence-transformer model to use for creating vector embeddings for memory search.
    -   `context_size`: The default context window size (will be overridden by the model's metadata if possible).
    -   `max_tokens`: The maximum number of tokens to generate in a single response.
    -   `temperature`: Controls the "creativity" of the LLM. Lower is more deterministic.
    -   `top_p`: Nucleus sampling parameter.
-   **`memory`**:
    -   `..._db_path`: Paths to the ChromaDB databases. These are relative to the user's data directory (`~/.jenova-ai/users/<username>/memory/`).
-   **`memory_search`**:
    -   `semantic_n_results`: The number of results to retrieve from semantic memory.
    -   `episodic_n_results`: The number of results to retrieve from episodic memory.
    -   `procedural_n_results`: The number of results to retrieve from procedural memory.
    -   `insight_n_results`: The number of results to retrieve from insight memory.
-   **`scheduler`**:
    -   `generate_insight_interval`: The interval (in conversation turns) for generating insights.
    -   `generate_assumption_interval`: The interval for generating assumptions.
    -   `proactively_verify_assumption_interval`: The interval for verifying assumptions.
    -   `reflect_interval`: The interval for reflecting on the cognitive graph.

#### `persona.yaml`

This file defines the AI's personality and core directives.

-   **`identity`**:
    -   `name`: The AI's name.
    -   `creator`: The creator's name.
    -   `origin_story`: A brief backstory.
    -   `type`: A description of the AI's nature.
-   **`directives`**: A list of rules the AI must follow. These are injected into the system prompt.
-   **`initial_facts`**: A list of foundational facts that are loaded into the AI's memory on first run.

### 7.3. Hardware Optimization

- **GPU Offloading:** For users with a compatible GPU, you can significantly accelerate the AI's processing by offloading model layers to the GPU. This is controlled by the `gpu_layers` setting in `main_config.yaml`. Setting it to `-1` will offload all possible layers.
- **Memory Locking:** To ensure the model remains in RAM for the fastest possible access, you can enable `mlock` in `main_config.yaml`. This prevents the operating system from swapping the model's memory to disk.
- **SWAP Loading:** While loading the model directly into SWAP on boot is an OS-level configuration and cannot be controlled from within the application, you can achieve a similar effect by ensuring you have a large enough SWAP partition and that `mlock` is disabled. The OS will then manage swapping the model in and out of RAM as needed. For more direct control, you would need to configure your system's boot scripts (e.g., using `systemd`) to load the model into SWAP, which is beyond the scope of this application's configuration.