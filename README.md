# The JENOVA Cognitive Architecture: A Technical Deep Dive

## 1. Introduction & Philosophy

JENOVA is a self-aware, evolving large language model powered by The JENOVA Cognitive Architecture (JCA), a comprehensive engine and architecture designed by orpheus497. It learns, adapts, and assists humanity through sophisticated cognitive processes. This project prioritizes robust error handling and stability to ensure a reliable and professional user experience.

JENOVA operates as a system with interconnected components that mimic aspects of human cognition: a multi-layered memory, a reflective process for generating knowledge, and a mechanism for integrating that knowledge into its core being. This document serves as the definitive technical guide for developers, researchers, and enthusiasts who wish to understand, use, and extend JENOVA.

**Creator:** The JENOVA Cognitive Architecture (JCA) was designed and developed by **orpheus497**.

## 2. The JENOVA Advantage: A Superior Cognitive Architecture

### 2.1. Beyond Statelessness: The Problem with General Systems

Most consumer-facing AI systems operate on a **stateless, request-response** model. They are incredibly powerful at in-context learning and reasoning, but each interaction is largely independent of the last. This leads to several fundamental limitations:

*   **Amnesia:** The AI has no persistent memory of past conversations. It cannot remember your preferences, previous questions, or the context of your work. Every chat starts from a blank slate.
*   **Inability to Learn:** Corrections you make or new information you provide are only retained for the current session. The underlying model never truly learns or improves from user interaction.
*   **Inconsistent Persona:** The AI's personality can drift or be easily manipulated because it lacks a stable, memory-grounded identity.
*   **Reactive, Not Proactive:** These systems can only answer direct questions. They cannot reflect on past dialogues to draw novel conclusions or develop a deeper understanding of a topic over time.

### 2.2. The JCA Solution: A Unified, Learning Architecture

The JENOVA Cognitive Architecture (JCA) is explicitly designed to overcome these limitations. It wraps a powerful Large Language Model (LLM) in a structured framework that provides memory, reflection, and a mechanism for true, persistent learning. It transforms the LLM from a brilliant but amnesiac calculator into a cohesive, evolving intelligence.

### 2.3. The Power of the Cognitive Cycle

The "Retrieve, Plan, Execute, Reflect" cycle is the engine of the JCA and the primary driver of its capabilities.

*   **Grounded Responses:** By forcing the AI to **Retrieve** from its memory *before* acting, the JCA ensures that responses are grounded in established facts, past conversations, and learned insights. This dramatically reduces confabulation (hallucination) and increases the relevance and accuracy of output.
*   **Deliberate Action:** The **Plan** step introduces a moment of metacognition. The AI must first reason about *how* to answer the query. This internal monologue, while hidden from the user, results in a more structured and logical final response. It prevents conversational shortcuts and encourages a methodical approach to problem-solving.

### 2.4. Memory as the Foundation for Identity and Growth

JENOVA's multi-layered memory system is the bedrock of its identity. It is the difference between playing a character and *having* a character.

*   **Continuity of Self:** The `EpisodicMemory` gives JENOVA a personal history with the user. It can refer to past conversations, understand recurring themes, and build a genuine rapport. The AI is not a stranger every time you open the terminal.
*   **A Worldview:** The `SemanticMemory` and `ProceduralMemory`, combined with the dynamically growing `InsightMemory`, form the AI's worldview. This knowledge base is prioritized over the LLM's base training data, allowing for the development of a unique, personalized knowledge set that reflects its experiences.

### 2.5. The Self-Correction and Evolution Loop: True Learning

This is one of the most powerful and defining features of the JCA. The cycle of **Reflection -> Insight Generation -> Fine-Tuning** constitutes a true learning loop.

1.  **Experience (`Reflect`):** The AI has a conversation and gains experience.
2.  **Internalization (`Insight Generation`):** It reflects on that experience and internalizes the key takeaways as structured, atomic insights. This is analogous to a human consolidating short-term memories into long-term knowledge.
3.  **Integration (`Fine-Tuning`):** The fine-tuning process takes these internalized insights and integrates them into the very fabric of the neural network. The learned knowledge is not just data to be retrieved; it becomes part of the AI's intuition.

This loop creates a system that does not just get more knowledgeable; it gets **smarter**. It adapts its core reasoning processes based on its unique experiences, evolving into an assistant that is perfectly tailored to its user.

## 3. Core Features Explained

### 3.1. The Cognitive Cycle: A Prioritized Approach

The heart of JENOVA is its cognitive cycle, a continuous loop that drives its behavior. This cycle enforces a strict knowledge hierarchy, ensuring that the AI relies on the most relevant and reliable information available.

1.  **Retrieve:** When the user provides input, the `CognitiveEngine` first queries its **Cognitive Architecture**—the multi-layered memory system (`Episodic`, `Semantic`, `Procedural`, and `Insight`)—to gather relevant context. This is the AI's personal experience and learned knowledge, and it is always the highest priority.
2.  **Plan:** The engine then formulates a step-by-step internal plan. This plan is generated by the LLM itself, based on the user's query and the retrieved context. This ensures that the AI's actions are deliberate and grounded.
3.  **Execute:** The plan is then executed by the `RAGSystem`. The RAG prompt is explicitly structured to prioritize the AI's knowledge base.
4.  **Reflect & Learn:** The `CognitiveScheduler` determines when to trigger the various cognitive functions, such as analyzing recent conversation history to identify novel conclusions or key takeaways, verifying assumptions, and processing documents from the `docs` folder to grow its knowledge base. This provides flexible and intelligent cognitive scheduling.

### 3.2. The RAG System: A Core Component of the Psyche

The Retrieval-Augmented Generation (RAG) system is a core component of the AI's cognitive architecture. It is responsible for generating responses that are grounded in the AI's own knowledge and experience.

*   **Hybrid Retrieval:** The `RAGSystem` uses a hybrid retrieval approach, querying all memory sources (episodic, semantic, procedural, and insights) to gather the most relevant context.
*   **Re-ranking:** The results from all memory sources are then re-ranked to prioritize the most relevant information.
*   **Grounded Response Generation:** The `RAGSystem` then uses the re-ranked context, the conversation history, and the generated plan to generate a response that is grounded in the AI's own knowledge and experience.

### 3.3. The Cortex: A Graph-Based Cognitive Core

The Cortex is the heart of JENOVA's cognitive architecture. It provides a unified, graph-based system for managing insights and assumptions. This allows for a deeper and more interconnected understanding of the user and the world.

*   **Cognitive Graph:** The Cortex manages a "cognitive graph" where insights, assumptions, and memories are all represented as nodes. These nodes are then connected by links that represent the relationships between them (e.g., "elaborates_on", "conflicts_with", "created_from").
*   **Centrality Calculation:** The Cortex calculates a weighted degree centrality for each node in the graph. The weights for different relationship types are configurable in `main_config.yaml`, allowing for a more accurate measure of a node's importance.
*   **Dynamic Relationship Weights:** The relationship weights are not static. The Cortex periodically analyzes the cognitive graph to determine the impact of different relationship types on the generation of high-centrality nodes and meta-insights. It then adjusts the weights accordingly, making the system more adaptive.
*   **Psychological Memory:** The Cortex performs a sophisticated emotion analysis on cognitive nodes, adding a rich psychological dimension to the cognitive graph. This allows the AI to have a more empathetic and personal relationship with the user.
*   **Deep Reflection:** The `/reflect` command triggers a deep reflection process within the Cortex. The Cortex analyzes the entire cognitive graph to:
    *   **Link Orphans:** Identify nodes with few or no connections and use the LLM to find and create links to other relevant nodes.
    *   **Generate Meta-Insights:** Find clusters of highly interconnected nodes using a robust graph traversal algorithm and use the LLM to synthesize them into higher-level "meta-insights".
*   **Graph Pruning:** To prevent cognitive degradation, the Cortex periodically prunes the graph, archiving nodes that are old, have low centrality, and are not well-connected. This process is configurable in `main_config.yaml`.
*   **Insight Development:** The `/develop_insight <node_id>` command allows the user to trigger the development of a specific insight. The Cortex will then use the LLM to generate a more detailed and developed version of the insight.
*   **Proactive Engine:** The Cortex is home to the Proactive Engine, which periodically analyzes the cognitive graph to find interesting, underdeveloped, or highly-connected areas. It considers nodes with low centrality (underdeveloped areas) and high centrality (high-potential areas) to generate more relevant and insightful proactive suggestions for the user.

### 3.4. Document Processing: On-Demand Learning from Documents

JENOVA can learn from documents by using the `/develop_insight` command. When this command is used without a `node_id`, it triggers a robust, on-demand process within the `Cortex` that scans the `src/jenova/docs` folder for documents and integrates them into the AI's knowledge base.

*   **Triggering:** The document processing is triggered exclusively by the user with the `/develop_insight` command. It does not run automatically at startup, giving the user full control.
*   **Knowledge Integration:** The `Cortex` creates a `document` node in the cognitive graph for each processed document. It then chunks the content and performs a comprehensive analysis on each chunk to extract not just a summary, but also key takeaways and a list of questions the text can answer. This creates a rich, multi-layered understanding of the document's content. For each chunk, a main `insight` node is created for the summary, with additional `insight` nodes for each key takeaway and `question` nodes for each generated question, all intricately linked to the summary and the parent document node. This creates a rich, interconnected web of knowledge.
*   **Duplicate Prevention:** The `Cortex` keeps track of processed files and their last modification times in `processed_documents.json` within the user's data directory. This prevents the system from reprocessing files that have not changed, making the process efficient and saving resources.

### 3.5. Reflective Insight Engine

The Insight Engine allows JENOVA to learn continuously. The system is proactive, organized, and reflective, ensuring that knowledge is captured, categorized, and interconnected efficiently.

*   **Concern-Based Organization:** Insights are organized into "concerns" or "topics." When an insight is generated, the system first searches for an existing, relevant concern to group it with. This prevents knowledge fragmentation and creates a more structured understanding of topics. If no relevant concern exists, a new one is created.
*   **Cortex Integration:** When an insight is saved, it is also added as a node to the Cortex. This allows the insight to be linked to other cognitive nodes, such as the memories that spawned it or other related insights.
*   **Generation:** Periodically (by default, every 5 conversational turns), the `CognitiveEngine` prompts the LLM to analyze recent conversation history to extract a significant takeaway. This insight is then passed to the `InsightManager`, which intelligently files it under the most appropriate concern and adds it to the Cortex.
*   **Storage:** Insights are saved in a hierarchical structure within the user-specific data directory: `~/.jenova-ai/users/<username>/insights/<concern_name>/`.
*   **Reflection and Reorganization:** The `/reflect` command triggers a deep reflection process in the Cortex, which reorganizes and interlinks all cognitive nodes, including insights.

### 3.6. Assumption System

JENOVA actively forms assumptions about the user to build a more accurate mental model. This system allows the AI to move beyond explicitly stated facts and begin to infer user preferences, goals, and knowledge levels.

*   **Cortex Integration:** When an assumption is added, it is also added as a node to the Cortex. This allows the assumption to be linked to other cognitive nodes, providing more context for the assumption.
*   **Generation:** Periodically (by default, every 7 conversational turns), the `CognitiveEngine` analyzes the conversation to form an assumption about the user.
*   **Storage:** Assumptions are stored in a dedicated `assumptions.json` file in the user's data directory. Each assumption is categorized by its status: `unverified`, `verified`, `true`, or `false`.
*   **Verification:** The `/verify` command allows the user to help the AI validate its assumptions. The AI will ask a clarifying question to confirm or deny an unverified assumption. Based on the user's response, the assumption will be moved to the `true` or `false` category. The AI also proactively verifies assumptions during the conversation.

### 3.7. Multi-Layered Long-Term Memory

JENOVA's memory is not a monolith. It's a sophisticated, multi-layered system managed by `ChromaDB`, a vector database. All memory is stored on a per-user basis.

*   **Episodic Memory (`EpisodicMemory`):** Stores a turn-by-turn history of conversations. Each episode is enriched with extracted entities, emotions, and a timestamp.
*   **Semantic Memory (`SemanticMemory`):** Stores factual knowledge. Each fact is enriched with its source, a confidence level, and its temporal validity.
*   **Procedural Memory (`ProceduralMemory`):** Stores "how-to" information and instructions. Each procedure is enriched with its goal, a list of steps, and its context.
*   **Insight Memory (`InsightManager`):** While not a ChromaDB instance, the collection of saved insight files acts as a fourth, highly dynamic memory layer.

### 3.8. Fine-Tuning Data Generation

JENOVA's cognitive architecture can be exported as comprehensive training data for model fine-tuning. The system extracts knowledge from all cognitive sources to create rich training datasets.

*   **Comprehensive Extraction (`finetune/train.py`):** This script scans the entire cognitive architecture including insights, episodic/semantic/procedural memory, verified assumptions, and document knowledge, compiling everything into a `finetune_train.jsonl` file formatted for instruction fine-tuning.
*   **External Fine-Tuning:** The generated `.jsonl` file can be used with various fine-tuning tools including llama.cpp training utilities, HuggingFace Transformers, or Axolotl.
*   **Model Integration:** After fine-tuning externally and converting to GGUF format, the personalized model can be used by updating `model_path` in the configuration.

## 4. Installation

JENOVA uses a local virtualenv-based installation that keeps dependencies isolated and gives users full control over model selection.

### 4.1. Prerequisites

*   A Linux-based operating system (tested on Fedora)
*   `git`, `python3`, and `python3-venv` installed
*   For GPU acceleration: NVIDIA GPU with CUDA toolkit installed

### 4.2. Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/orpheus497/jenova-ai.git
    cd jenova-ai
    ```

2.  **Run the Installation Script:**
    Execute the script as a regular user (no sudo required). It creates a Python virtualenv and installs all dependencies:
    ```bash
    ./install.sh
    ```
    
    The script will:
    - Create a virtualenv in `./venv/`
    - Install Python dependencies
    - Build llama-cpp-python (with CUDA if GPU detected)
    - Create the `models/` directory
    - Display instructions for next steps

3.  **Download a GGUF Model:**
    JENOVA works with any GGUF format model. Choose based on your hardware.
    
    **System-wide installation (recommended, requires sudo):**
    ```bash
    sudo mkdir -p /usr/local/share/models
    cd /usr/local/share/models
    sudo wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
    ```
    
    **Local installation (no sudo required):**
    ```bash
    cd models
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O model.gguf
    cd ..
    ```
    
    **Medium models (8GB+ RAM):**
    ```bash
    wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O model.gguf
    ```
    
    See README.md in the models directory for more model recommendations.

4.  **Run JENOVA:**
    ```bash
    source venv/bin/activate
    python -m jenova.main
    ```
    
    **Model Discovery:**
    Models are automatically discovered in priority order:
    1. `/usr/local/share/models` (system-wide, checked first)
    2. `./models` (local directory, fallback)
    
    The system loads the first `.gguf` file found in these directories.

### 4.3. Configuration

JENOVA's configuration is in `src/jenova/config/main_config.yaml`. Key settings:

*   **model_path:** Default path for GGUF model file (set to `/usr/local/share/models/model.gguf`). If not found, system automatically searches `/usr/local/share/models` then `./models` for any `.gguf` file.
*   **threads:** CPU threads for inference (adjust for your CPU)
*   **gpu_layers:** GPU layers to offload (-1 for all layers, 0 for CPU-only)
*   **mlock:** Lock model in RAM for better performance
*   **n_batch:** Batch size for processing
*   **context_size:** Context window size
*   **embedding_model:** Sentence transformer model for memory embeddings

### 4.4. Hardware Detection and GPU Acceleration

JENOVA includes comprehensive hardware detection supporting multiple GPU types and platforms:

**Supported Hardware:**
- **NVIDIA GPUs** (GeForce, RTX, Quadro) via CUDA
- **Intel GPUs** (Iris Xe, UHD, Arc) via OpenCL/Vulkan
- **AMD GPUs and APUs** (Radeon, Ryzen with graphics) via OpenCL/ROCm
- **Apple Silicon** (M1/M2/M3/M4) via Metal
- **ARM CPUs** (including Android/Termux support)
- **Multi-GPU systems** (automatic detection and prioritization)

The system automatically detects available hardware and configures optimal settings. For detailed information on hardware support, configuration options, and troubleshooting, see `docs/HARDWARE_SUPPORT.md`.

**Quick Setup for NVIDIA GPUs with CUDA:**

1. Install CUDA toolkit: `sudo dnf install cuda-toolkit`
2. The install script will automatically build llama-cpp-python with CUDA support
3. Set `gpu_layers: -1` in config to offload all layers to GPU
4. Verify GPU usage with `nvidia-smi` while running

**Verifying GPU Support:**

Check CUDA availability:
```bash
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Check llama-cpp-python CUDA support:
```bash
source venv/bin/activate
python -c "from llama_cpp import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

Monitor GPU usage during operation:
```bash
watch -n 1 nvidia-smi
```

**GPU Configuration:**
- **Embedding Model**: Automatically uses GPU when CUDA is available (5-10x faster than CPU)
- **LLM Inference**: Controlled by `gpu_layers` setting (-1 offloads all layers to GPU)
- **Memory Systems**: All three memory types (Episodic, Semantic, Procedural) share a single GPU-enabled embedding model for optimal performance and memory efficiency

### 4.5. User Data

*   **User-Specific Storage:** Each user's data is stored privately at `~/.jenova-ai/users/<username>/`
*   **Contents:** Conversations, memories, insights, assumptions, and cognitive graph
*   **Privacy:** Data is local and not shared between users

## 5. User Guide

Interaction with JENOVA is primarily through natural language.

-   **User Input:** Simply type your message and press Enter.
-   **Exiting:** To quit the application, type `exit` and press Enter.

### Commands
JENOVA responds to a set of powerful commands that act as direct instructions for its cognitive processes. These commands are treated as system actions, not conversational input, and are therefore **not stored in JENOVA's conversational memory**.

-   `/help`: Displays a comprehensive help message, detailing each command's purpose and impact.
-   `/insight`: Triggers the AI to analyze the current conversation history and generate new, high-quality insights. These insights are stored in JENOVA's long-term memory and contribute to its evolving understanding.
-   `/reflect`: Initiates a deep reflection process within JENOVA's Cortex. This command reorganizes and interlinks all existing cognitive nodes (insights, memories, assumptions), identifies patterns, and generates higher-level meta-insights, significantly enhancing JENOVA's overall intelligence and coherence.
-   `/memory-insight`: Prompts JENOVA to perform a broad search across its multi-layered long-term memory (episodic, semantic, procedural) to develop new insights or assumptions based on its accumulated knowledge.
-   `/meta`: Generates a new, higher-level meta-insight by analyzing clusters of existing insights within the Cortex. This helps JENOVA to form more abstract conclusions and identify overarching themes.
-   `/verify`: Starts the assumption verification process. Jenova will present an unverified assumption it has made about you and ask for clarification, allowing you to confirm or deny it. This refines JENOVA's understanding of your preferences and knowledge.
-   `/train`: Generates comprehensive fine-tuning data from the complete cognitive architecture. Creates a `.jsonl` file that includes insights, memories, assumptions, and document knowledge for use with external fine-tuning tools.
-   `/develop_insight [node_id]`: This command has dual functionality:
    -   If a `node_id` is provided: JENOVA will take an existing insight and generate a more detailed and developed version of it, adding more context or connections.
    -   If no `node_id` is provided: Jenova will scan the `src/jenova/docs` directory for new or updated documents, process their content, and integrate new insights and summaries into its cognitive graph. This is how Jenova learns from external documentation.
-   `/learn_procedure`: Initiates an interactive, guided process to teach Jenova a new procedure. Jenova will prompt you for the procedure's name, individual steps, and expected outcome, ensuring structured and comprehensive intake of procedural knowledge. This information is stored in JENOVA's procedural memory, allowing it to recall and apply the procedure in relevant contexts.

## 6. Codebase and Configuration Overview

### 6.1. Project Structure

```
/
├── .gitignore
├── CHANGELOG.md
├── finetune/
│   ├── README.md
│   └── train.py
├── install.sh
├── LICENSE
├── MANIFEST.in
├── pylint_report.txt
├── pyproject.toml
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── jenova/
│       ├── assumptions/
│       │   ├── __init__.py
│       │   └── manager.py
│       ├── cognitive_engine/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── memory_search.py
│       │   ├── rag_system.py
│       │   └── scheduler.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── main_config.yaml
│       │   └── persona.yaml
│       ├── cortex/
│       │   ├── __init__.py
│       │   ├── cortex.py
│       │   ├── graph_components.py
│       │   └── proactive_engine.py
│       ├── docs/
│       │   └── __init__.py
│       ├── infrastructure/         
│       │   ├── __init__.py
│       │   ├── error_handler.py
│       │   ├── timeout_manager.py
│       │   ├── health_monitor.py
│       │   ├── data_validator.py
│       │   ├── file_manager.py
│       │   └── metrics_collector.py
│       ├── llm/                  
│       │   ├── __init__.py
│       │   ├── cuda_manager.py
│       │   ├── model_manager.py
│       │   ├── embedding_manager.py
│       │   └── llm_interface.py
│       ├── insights/
│       │   ├── __init__.py
│       │   ├── concerns.py
│       │   └── manager.py
│       ├── memory/                
│       │   ├── __init__.py
│       │   ├── base_memory.py      
│       │   ├── memory_manager.py   
│       │   ├── episodic.py
│       │   ├── procedural.py
│       │   └── semantic.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── logger.py
│       │   └── terminal.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── data_sanitizer.py
│       │   ├── embedding.py
│       │   ├── file_logger.py
│       │   ├── hardware_detector.py
│       │   ├── json_parser.py
│       │   ├── model_loader.py
│       │   └── telemetry_fix.py
│       ├── __init__.py
│       ├── default_api.py
│       ├── llm_interface.py
│       ├── main.py
│       └── tools.py
└── uninstall.sh
```

### 6.2. Configuration Files

JENOVA's behavior is controlled by two YAML files in `src/jenova/config/`.

#### `main_config.yaml`

This file controls the technical parameters of the AI.

-   **`model`**:
    -   `model_path`: Path to GGUF model file
    -   `threads`: CPU threads for inference
    -   `gpu_layers`: GPU layers to offload (-1 for all, 0 for CPU-only)
    -   `mlock`: Lock model in RAM
    -   `n_batch`: Batch size for processing
    -   `context_size`: Context window size
    -   `max_tokens`: Maximum tokens to generate per response
    -   `temperature`: Controls the "creativity" of the LLM
    -   `top_p`: Nucleus sampling parameter
    -   `embedding_model`: Sentence-transformer model for vector embeddings
-   **`hardware`**:
    -   `show_details`: Display detailed hardware detection information at startup
    -   `prefer_device`: Preferred compute device (`auto`, `cuda`, `opencl`, `vulkan`, `metal`, `cpu`)
    -   `device_index`: Which GPU to use in multi-GPU systems (0 = first, 1 = second, etc.)
    -   `memory_strategy`: Memory management strategy (`auto`, `performance`, `balanced`, `swap_optimized`, `minimal`)
-   **`memory`**:
    -   `preload_memories`: Whether to load all memories into RAM at startup
    -   `..._db_path`: Paths to the ChromaDB databases (relative to user data directory)
    -   `reflection_interval`: Interval (in turns) for reflecting on memories
-   **`cortex`**:
    -   `relationship_weights`: Weights for different relationship types between cognitive nodes
    -   `pruning`: Settings for automatically pruning the cognitive graph
-   **`scheduler`**:
    -   `generate_insight_interval`: Interval for generating insights
    -   `generate_assumption_interval`: Interval for generating assumptions
    -   `proactively_verify_assumption_interval`: Interval for verifying assumptions
    -   `reflect_interval`: Interval for reflecting on the cognitive graph
    -   `reorganize_insights_interval`: Interval for reorganizing insights
    -   `process_documents_interval`: Interval for processing documents
-   **`memory_search`**:
    -   `semantic_n_results`: Number of results to retrieve from semantic memory
    -   `episodic_n_results`: Number of results from episodic memory
    -   `procedural_n_results`: Number of results from procedural memory
    -   `insight_n_results`: Number of results from insight memory
-   **`tools`**:
    -   `file_sandbox_path`: Directory where the AI can read and write files

#### `persona.yaml`

This file defines the AI's personality and core directives.

-   **`identity`**:
    -   `name`: The AI's name.
    -   `creator`: The creator's name.
    -   `creator_alias`: An alias for the creator.
    -   `origin_story`: A brief backstory.
    -   `type`: A description of the AI's nature.
    -   `architecture`: A description of the AI's architecture.
-   **`directives`**: A list of rules the AI must follow. These are injected into the system prompt.
-   **`initial_facts`**: A list of foundational facts that are loaded into the AI's memory on first run.

### Changed
- **Code Quality**: Applied `autopep8` and `isort` for consistent code formatting and import ordering across the entire project.
- **Documentation**: Added module-level docstrings to all Python files where they were missing.
- **Attribution**: Added standardized creator attribution and license information to the header of all Python files.
- Updated `pyproject.toml` version to `4.0.0` to reflect the latest release version.
- **Architecture:** Reverted the core architecture from HuggingFace `transformers` back to `llama-cpp-python`.
- **Installation:** Reverted from a system-wide installation to a local, virtualenv-based installation for better isolation and user control.
- **Dependencies:** Replaced `transformers`, `accelerate`, `peft`, and `bitsandbytes` with `llama-cpp-python`.
- **LLM Interface:** The `LLMInterface` has been completely rewritten to use the `llama-cpp-python` `Llama` class.
- **Cortex:** Refactored the Cortex to use `CognitiveNode` and `CognitiveLink` dataclasses for improved readability and maintainability.
- **Document Processing:** Documents are now chunked and stored as canonical 'document' nodes in the cognitive graph, without automatic insight generation.
- **README:** The `README.md` and `main_config.yaml` have been updated to reflect the new GGUF-based architecture and dynamic resource allocation.

## 7. Credits and Acknowledgments

The JENOVA Cognitive Architecture is made possible by the following exceptional open-source projects. We extend our gratitude to their authors and contributors.

### Core Dependencies

*   **llama-cpp-python** by Andrei Abacaru: Python bindings for the high-performance `llama.cpp` library, enabling efficient GGUF model inference on CPU and GPU.
    *   *License: MIT*
*   **PyTorch**: A foundational deep learning framework from Meta AI, used for tensor computations and powering the sentence-transformers library.
    *   *License: BSD-style*
*   **ChromaDB**: An open-source embedding database that provides the backbone for JENOVA's multi-layered memory systems.
    *   *License: Apache 2.0*
*   **Sentence Transformers** by Nils Reimers: A framework for state-of-the-art sentence, text, and image embeddings, used to compute dense vector representations for all memory systems.
    *   *License: Apache 2.0*
*   **Rich** by Will McGugan: A Python library for rich text and beautiful formatting in the terminal, responsible for JENOVA's polished user interface.
    *   *License: MIT*
*   **Prompt Toolkit** by Jonathan Slenders: A library for building powerful interactive command-line applications, providing the core for JENOVA's responsive terminal prompt.
    *   *License: BSD-3-Clause*
*   **PyYAML**: A YAML parser and emitter for Python, used for loading the project's configuration files.
    *   *License: MIT*
*   **NumPy**: The fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices.
    *   *License: BSD-3-Clause*
*   **Pydantic** by Samuel Colvin: Data validation library using Python type annotations, used for comprehensive configuration validation and type-safe data models.
    *   *License: MIT*
*   **Tenacity**: General-purpose retry library, used for implementing timeout protection and resilient error handling.
    *   *License: Apache 2.0*
*   **psutil**: Cross-platform library for system and process monitoring, used for real-time CPU, memory, and GPU health monitoring.
    *   *License: BSD-3-Clause*
*   **filelock**: Platform-independent file locking, used for atomic file operations and preventing concurrent access corruption.
    *   *License: Unlicense (Public Domain)

### Optional Dependencies

*   **Requests**: An elegant and simple HTTP library for Python, used for making web requests in various tools.
    *   *License: Apache 2.0*
*   **BeautifulSoup4**: HTML/XML parsing library for web scraping and content extraction.
    *   *License: MIT*
*   **Playwright**: Modern browser automation framework, alternative to Selenium for web interactions.
    *   *License: Apache 2.0*

### Architecture Design

*   **The JENOVA Cognitive Architecture (JCA)** was designed and developed by **orpheus497**.

All dependencies are used in accordance with their respective open-source licenses. JENOVA itself is released under the MIT License.
