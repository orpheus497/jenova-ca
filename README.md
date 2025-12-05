# The JENOVA Cognitive Architecture: A Complete Technical Reference

## 1. Introduction & Philosophy

JENOVA is an evolving large language model system powered by The JENOVA Cognitive Architecture (JCA), a comprehensive cognitive framework designed by orpheus497. The architecture implements a sophisticated cognitive processing system that learns, adapts, and assists through multi-layered memory, reflective reasoning, and continuous knowledge integration.

**Interface:** JENOVA is a **100% terminal-based application** with a rich, interactive command-line interface. There is no web UI or voice interface - all interaction happens through a sophisticated terminal interface powered by the `rich` library and `prompt-toolkit`.

### 1.1. What is JENOVA?

JENOVA represents a comprehensive approach to building stateful, learning-capable AI systems. Unlike traditional stateless LLM deployments, JENOVA implements:

- **Persistent Multi-Layered Memory**: Episodic, semantic, procedural, and insight-based memory systems using vector databases
- **Cognitive Processing Cycle**: Structured retrieve-plan-execute-reflect workflow for grounded, deliberate responses
- **Graph-Based Knowledge Representation**: Dynamic cognitive graph with weighted relationships and centrality-based importance
- **Distributed Computing Architecture**: LAN-based resource pooling for parallel inference and federated memory search
- **Continuous Learning Loop**: Reflection, insight generation, and knowledge integration capabilities
- **Production-Ready Infrastructure**: Comprehensive error handling, timeout protection, health monitoring, and type-safe configuration
- **Rich Terminal Interface**: Interactive CLI with real-time metrics, health monitoring, and 25+ built-in tools

The architecture is designed for local deployment with full user control over models, data, and processing. All cognitive processes run locally with optional distributed computing across trusted LAN peers.

**Creator:** The JENOVA Cognitive Architecture (JCA) was designed and developed by **orpheus497**.

**License:** MIT License - Full open source with comprehensive dependency attribution

### 1.2. Distributed Computing Capabilities

JENOVA implements comprehensive **LAN-based distributed computing**, enabling multiple instances to discover each other automatically and pool their hardware resources:

**Core Distributed Features:**
- 🚀 **Parallel LLM Inference** - Distribute generation across multiple GPUs for 3-4x faster responses
- ⚡ **Intelligent Load Balancing** - Five distribution strategies (LOCAL_FIRST, LOAD_BALANCED, FASTEST_PEER, PARALLEL_VOTING, ROUND_ROBIN)
- 🔄 **Automatic Failover** - Health monitoring and automatic peer recovery on instance failure
- 💪 **Resource Pooling** - Combine CPU, GPU, and RAM across trusted LAN machines
- 🧠 **Federated Memory Search** - Parallel memory queries across peers with privacy controls
- 🔒 **Enterprise Security** - SSL/TLS encryption, certificate pinning (TOFU), JWT authentication, and encrypted credential storage
- 🌐 **Zero-Configuration Discovery** - Automatic mDNS/Zeroconf peer discovery and lifecycle management

**Network Architecture:**
- **Protocol:** gRPC with Protocol Buffers for efficient serialization
- **Discovery:** mDNS/Zeroconf for zero-configuration LAN discovery
- **Security:** Self-signed certificates with SSL/TLS, JWT token-based authentication
- **Privacy:** Memory sharing disabled by default, configurable trust boundaries
- **Monitoring:** Comprehensive metrics for latency, bandwidth, and load distribution

## 2. Architecture Overview: Core Design Principles

### 2.1. Stateful Cognitive Processing

JENOVA implements a stateful cognitive architecture that maintains persistent context across interactions. The system addresses fundamental challenges in AI assistants through:

**Memory Persistence:**
- **ChromaDB Vector Storage**: All memory types use persistent vector databases with semantic search
- **Per-User Isolation**: Each user's data stored privately at `~/.jenova-ai/users/<username>/`
- **Multi-Layered Organization**: Episodic (conversations), Semantic (facts), Procedural (how-to), Insight (learned knowledge)
- **Rich Metadata**: Timestamps, entities, emotions, confidence levels, and temporal validity

**Continuous Learning:**
- **Insight Generation**: Periodic analysis of conversations to extract and store key takeaways
- **Assumption Formation**: The system forms and verifies assumptions about user preferences and knowledge
- **Knowledge Integration**: Fine-tuning data generation from cognitive architecture for model personalization
- **Graph-Based Synthesis**: Meta-insight generation through graph analysis and clustering

**Identity and Persona:**
- **Stable Configuration**: Identity defined in `persona.yaml` with directives and initial facts
- **Memory-Grounded Responses**: All responses prioritize personal memory over base model knowledge
- **Proactive Engagement**: Suggestion engine analyzes cognitive graph for proactive recommendations

**Reflective Reasoning:**
- **Scheduled Cognition**: Configurable cognitive scheduler for insight generation, reflection, and verification
- **Deep Reflection**: Graph traversal algorithms for pattern identification and meta-insight synthesis
- **Continuous Improvement**: Cognitive graph pruning removes outdated nodes while preserving important knowledge

### 2.2. The JCA Solution: Comprehensive Cognitive Framework

The JENOVA Cognitive Architecture implements a production-ready framework with multiple foundational layers:

1. **Infrastructure Layer**: Error handling, timeout protection, health monitoring, data validation, file management, metrics collection, circuit breaker patterns
2. **Core Layer**: Application bootstrap, dependency injection container, lifecycle management, modular architecture
3. **LLM Layer**: CUDA management, model lifecycle, embedding management, retry logic, timeout protection
4. **Memory Layer**: Abstract base classes, unified memory manager, atomic operations, cross-memory search, backup management, context compression, deduplication
5. **Cognitive Engine**: RAG system with LRU caching, configurable re-ranking, comprehensive timeout coverage
6. **UI Layer**: Health display, rich terminal interface, real-time metrics, comprehensive command system
7. **Testing Layer**: 680+ comprehensive tests across all architecture layers including CLI enhancements
8. **Distributed Layer**: gRPC services, peer management, federated operations, security infrastructure
9. **Security Layer**: Audit logging, encryption at rest, prompt sanitization, rate limiting, input validation
10. **Optimization Layer**: Bayesian optimization, performance profiling, self-tuning, task classification
11. **Observability Layer**: Metrics export, distributed tracing, structured logging
12. **Emotional Intelligence Layer**: Emotion detection, empathetic response generation, emotional state management
13. **Collaboration Layer**: Multi-user access control, session management, sync protocols
14. **Branching Layer**: Conversation branching, branch navigation, state persistence
15. **Plugin Layer**: Plugin API, sandboxed execution, plugin discovery and management
16. **Visualization Layer**: Graph analysis, export formats, terminal rendering

### 2.3. The Cognitive Processing Cycle

The "Retrieve, Plan, Execute, Reflect" cycle forms the core processing engine of JENOVA's cognitive architecture:

**1. Retrieve Phase:**
- **Memory Search**: Query all memory layers (episodic, semantic, procedural, insight) using semantic similarity
- **Context Assembly**: Gather relevant facts, past conversations, learned procedures, and insights
- **Relevance Ranking**: Optional re-ranking using LLM for precise context prioritization
- **Privacy Controls**: Distributed memory search respects user-configured sharing boundaries
- **Performance**: LRU caching provides <100ms response on frequently accessed context

**2. Plan Phase:**
- **Metacognitive Analysis**: LLM generates structured plan based on query and retrieved context
- **Tool Selection**: Determine if external tools needed (file operations, web search, time queries)
- **Strategy Formation**: Choose execution approach (direct response, multi-step reasoning, tool chain)
- **Timeout Protection**: Planning operations protected with configurable timeout (default 120s)
- **Logging**: Detailed planning logs for debugging and cognitive analysis

**3. Execute Phase:**
- **RAG-Based Generation**: Structured prompt prioritizing cognitive architecture over base model knowledge
- **Tool Integration**: Secure execution of file tools, shell commands (whitelist-based), web search
- **Response Synthesis**: Combine retrieved context, plan, and tool outputs into coherent response
- **Personalization**: Response style adapted based on user profile and expertise level
- **Quality Assurance**: Semantic analysis for intent, entities, sentiment, and topic extraction

**4. Reflect Phase:**
- **Conversation Analysis**: Periodic extraction of insights from recent interaction history
- **Knowledge Consolidation**: Store atomic insights in concern-based hierarchical structure
- **Assumption Generation**: Form and verify hypotheses about user preferences and knowledge
- **Graph Updates**: Add nodes to cognitive graph, link to related concepts
- **Proactive Suggestions**: Generate recommendations based on cognitive graph analysis

**Cognitive Benefits:**
- **Reduced Hallucination**: Memory-grounded responses prevent confabulation
- **Increased Relevance**: Context-aware generation using personal knowledge base
- **Structured Reasoning**: Explicit planning phase enforces methodical problem-solving
- **Continuous Learning**: Reflection loop enables persistent knowledge acquisition

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

*   A Linux-based operating system (tested on Fedora, Ubuntu, Debian, Arch)
*   macOS (Intel and Apple Silicon M1/M2/M3/M4 supported)
*   Windows (10/11 supported)
*   Termux (Android smartphones/tablets and iOS via Termux/iSH)
*   `git`, `python3` (3.10+), and `python3-venv` installed
*   For GPU acceleration: NVIDIA GPU with CUDA toolkit installed

### 4.2. Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/orpheus497/jenova-ai.git
    cd jenova-ai
    ```

2.  **Run the Installation Script:**

    **For Linux/macOS:**
    Execute the script as a regular user (no sudo required). It creates a Python virtualenv and installs all dependencies:
    ```bash
    ./install.sh
    ```

    **For Termux (Android/iOS):**
    Use the Termux-specific installation script:
    ```bash
    chmod +x install-termux.sh
    ./install-termux.sh
    ```

    The installation script will:
    - Create a virtualenv in `./venv/` (Linux/macOS) or install globally (Termux)
    - Install Python dependencies
    - Build llama-cpp-python (with CUDA if GPU detected on Linux)
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

The system automatically detects available hardware and configures optimal settings for your specific hardware tier.

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

JENOVA provides a **rich, interactive terminal interface** powered by the `rich` library and `prompt-toolkit`. All interaction happens through the command line - there is no web UI or voice interface.

### 5.0. Terminal Interface

**Starting JENOVA:**
```bash
source venv/bin/activate
python -m jenova.main
```

**Terminal Features:**
- **Rich Text Formatting**: Syntax highlighting, markdown rendering, and color-coded output
- **Interactive Prompt**: Auto-completion, command history, and multi-line input support
- **Real-Time Metrics**: Live display of system health, memory usage, and response times
- **Command System**: 25+ built-in commands for system control and cognitive management
- **Health Monitoring**: Visual indicators for CPU, GPU, memory, and model status
- **Conversation History**: Automatic saving of all interactions to episodic memory

**Basic Interaction:**
-   **User Input:** Simply type your message and press Enter.
-   **Multi-line Input:** Press `Alt+Enter` or `Esc+Enter` for new lines without submitting
-   **Exiting:** Type `exit` and press Enter, or press `Ctrl+D`
-   **Command Prefix:** All system commands start with `/` (e.g., `/help`, `/status`, `/health`)
-   **Help System:** Type `/help` at any time to see available commands and usage

**Visual Elements:**
- **Color-Coded Messages**: 
  - User input: Cyan
  - AI responses: Green
  - System messages: Yellow
  - Errors: Red
- **Progress Indicators**: Animated spinners during model loading and processing
- **Tables and Panels**: Organized display of metrics, status, and health information
- **Markdown Support**: AI can render formatted text, code blocks, and lists

### 5.1. Command Reference

JENOVA provides a comprehensive command system for direct control of cognitive processes and system management. Commands are treated as system actions and are **not stored in conversational memory**.

#### System Commands

-   **`/help`** - Display comprehensive command reference
    - Shows all available commands with detailed descriptions and usage examples
    - Organized into categories: System, Network, Memory, Learning, Settings
    - Includes usage tips and keyboard shortcuts

-   **`/health`** - Display real-time system health metrics
    - CPU usage, memory consumption, GPU utilization
    - LLM model status and embedding model health
    - ChromaDB connection status for all memory layers
    - Performance metrics and degradation warnings

-   **`/metrics`** - Show detailed performance metrics
    - Query response times and token generation rates
    - Memory search performance and cache hit rates
    - Cognitive cycle timing breakdown
    - Network metrics for distributed operations

-   **`/status`** - Display current system status
    - Active cognitive processes and scheduled tasks
    - Memory layer statistics (entry counts, storage size)
    - Configuration summary and active features
    - Uptime and session information

-   **`/cache`** - Display RAG system cache statistics
    - Cache hit/miss ratios and eviction counts
    - Most frequently accessed queries
    - Cache size and memory usage
    - Performance impact analysis

#### Network Commands (Distributed Mode)

-   **`/network [status|enable|disable|info]`** - Network management
    - `status` - Show network status and peer count
    - `enable` - Enable distributed computing mode
    - `disable` - Disable distributed mode (local only)
    - `info` - Detailed network configuration and metrics

-   **`/peers [list|info|trust|disconnect]`** - Peer management
    - `list` - Show all discovered peers with health status
    - `info <peer_id>` - Detailed peer information and capabilities
    - `trust <peer_id>` - Mark peer as trusted for memory sharing
    - `disconnect <peer_id>` - Disconnect from specific peer

#### Memory Commands

-   **`/insight`** - Generate insights from recent conversations
    - Analyzes conversation history to extract key takeaways
    - Stores insights in concern-based hierarchical structure
    - Links insights to cognitive graph for context
    - Output: Summary of generated insights and their storage locations

-   **`/memory-insight`** - Generate insights from long-term memory
    - Performs broad search across all memory layers
    - Identifies patterns and connections in accumulated knowledge
    - Creates assumptions based on memory analysis
    - Output: New insights and assumptions discovered

-   **`/reflect`** - Initiate deep cognitive reflection
    - Reorganizes and interlinks all cognitive nodes
    - Identifies orphan nodes and creates missing connections
    - Generates meta-insights from clustered insights
    - Prunes outdated or low-value nodes
    - Output: Summary of meta-insights, new links, and pruned nodes

-   **`/meta`** - Generate meta-insight from insight clusters
    - Analyzes high-centrality insight clusters
    - Synthesizes higher-level abstract conclusions
    - Creates meta-insight node in cognitive graph
    - Output: Generated meta-insight and cluster composition

-   **`/develop_insight [node_id]`** - Develop existing insight or process documents
    - **With node_id**: Generates detailed, expanded version of specific insight
    - **Without node_id**: Scans `src/jenova/docs/` for new/modified documents
        - Chunks documents and extracts summaries
        - Generates key takeaways and answerable questions
        - Creates interconnected document nodes in cognitive graph
        - Tracks processed files to prevent re-processing
    - Output: Developed insight or document processing summary

#### Learning Commands

-   **`/learn [stats|insights|gaps|skills]`** - Learning system interface
    - `stats` - Performance metrics, accuracy trends, learning rate
    - `insights` - Learning progress insights and patterns recognized
    - `gaps` - Identified knowledge gaps and improvement opportunities
    - `skills` - Acquired skills with proficiency levels (visual progress bars)

-   **`/verify`** - Assumption verification process
    - Presents unverified assumptions for user confirmation
    - Interactive clarification questions
    - Updates assumption status (true/false) based on response
    - Refines user profile and mental model
    - Output: Assumption verification result and profile update

-   **`/learn_procedure`** - Interactive procedure learning
    - Guided prompts for procedure name, steps, and expected outcome
    - Structured intake ensures comprehensive procedural knowledge
    - Stores in procedural memory with searchable metadata
    - Output: Confirmation of stored procedure

-   **`/train`** - Generate fine-tuning dataset
    - Extracts comprehensive training data from cognitive architecture
    - Includes insights, memories, assumptions, and document knowledge
    - Outputs `finetune_train.jsonl` in instruction-tuning format
    - Compatible with llama.cpp, HuggingFace, Axolotl fine-tuning tools
    - Output: Dataset location and entry count

#### Settings Commands

-   **`/settings`** - Interactive settings configuration menu
    - Five categories: Network, LLM, Memory, Learning, Privacy
    - Runtime configuration changes without restart
    - Type-safe validation before applying changes
    - Preview mode and pending changes management
    - Import/export settings to JSON
    - Undo/redo support with change history
    - Settings persistence to user profile

-   **`/profile`** - User profile viewer
    - Comprehensive interaction statistics
    - Vocabulary tracking and expertise level
    - Topic interests and preferred subjects
    - Command usage patterns
    - Correction history and suggestion feedback
    - Response style preferences

#### Code & Development Commands

-   **`/edit <file_path>`** - File editing with diff-based preview
    - Multi-file editing support
    - Syntax-aware line operations
    - Automatic backup creation before edits
    - Preview changes before applying
    - Output: Applied changes summary with line numbers

-   **`/parse <file_path>`** - Code structure and AST analysis
    - AST-based Python code parsing
    - Symbol extraction (classes, functions, methods, variables)
    - Dependency graph generation
    - Code structure visualization
    - Output: Parsed code structure and symbols

-   **`/refactor <operation> <target>`** - Code refactoring operations
    - Symbol renaming across codebase
    - Extract method/function
    - Inline variable
    - Import organization
    - Output: Refactoring summary and affected files

-   **`/analyze <file_or_directory>`** - Code quality and complexity analysis
    - Cyclomatic complexity metrics (McCabe)
    - Halstead metrics calculation
    - Maintainability index with quality grading (A-F)
    - Issue detection and recommendations
    - Output: Comprehensive code quality report

-   **`/scan <file_or_directory>`** - Security vulnerability scanning
    - Python security issue detection via Bandit
    - Pattern matching for common vulnerabilities (SQL injection, XSS, hardcoded secrets)
    - Multiple output formats (text, JSON, HTML)
    - Severity-based issue categorization
    - Output: Security scan report with recommendations

#### Git Commands

-   **`/git <operation> [args]`** - Git operations with AI assistance
    - `status` - Show working tree status
    - `diff [file]` - Show changes with analysis
    - `commit [message]` - Commit with AI-generated message (if message omitted)
    - `branch [name]` - Create, list, or delete branches
    - `log [count]` - Show commit history
    - Output: Git operation result with intelligent summaries

#### Orchestration Commands

-   **`/task <action>`** - Multi-step task planning and execution
    - `create <description>` - Create task plan with dependency graph
    - `execute <plan_id>` - Execute task plan with progress tracking
    - `pause <task_id>` - Pause running task
    - `resume <task_id>` - Resume paused task
    - `cancel <task_id>` - Cancel task execution
    - `list` - Show all task plans and their status
    - Output: Task execution progress and results

-   **`/workflow <workflow_name> [args]`** - Execute predefined workflows
    - Available workflows: `code_review`, `testing`, `deployment`, `refactoring`, `documentation`, `analysis`
    - Each workflow includes validation, execution steps, and rollback capability
    - Progress tracking with step-by-step status
    - Output: Workflow execution report

-   **`/command <action>`** - Custom command management
    - `create <name>` - Create custom Markdown template command
    - `execute <name> [vars]` - Execute custom command with variables
    - `list` - Show all available custom commands
    - `delete <name>` - Remove custom command
    - Output: Command execution result or management confirmation

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
├── install-termux.sh
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── build_proto.py
├── verify_build.py
├── verify_install.py
├── src/
│   └── jenova/
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── context_optimizer.py
│       │   ├── code_metrics.py
│       │   ├── security_scanner.py
│       │   ├── intent_classifier.py
│       │   └── command_disambiguator.py
│       ├── assumptions/
│       │   ├── __init__.py
│       │   └── manager.py
│       ├── automation/
│       │   ├── __init__.py
│       │   ├── custom_commands.py
│       │   ├── hooks_system.py
│       │   ├── template_engine.py
│       │   └── workflow_library.py
│       ├── branching/
│       │   ├── __init__.py
│       │   ├── branch_manager.py
│       │   └── branch_navigator.py
│       ├── code_tools/
│       │   ├── __init__.py
│       │   ├── file_editor.py
│       │   ├── code_parser.py
│       │   ├── refactoring_engine.py
│       │   ├── syntax_highlighter.py
│       │   ├── codebase_mapper.py
│       │   └── interactive_terminal.py
│       ├── cognitive_engine/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── memory_search.py
│       │   ├── rag_system.py
│       │   ├── scheduler.py
│       │   └── semantic_analyzer.py
│       ├── collaboration/
│       │   ├── __init__.py
│       │   ├── access_control.py
│       │   ├── collaboration_manager.py
│       │   ├── sync_protocol.py
│       │   └── user_session.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── config_schema.py
│       │   ├── main_config.yaml
│       │   └── persona.yaml
│       ├── core/
│       │   ├── __init__.py
│       │   ├── adapters.py
│       │   ├── application.py
│       │   ├── architecture.py
│       │   ├── bootstrap.py
│       │   ├── container.py
│       │   ├── interfaces/
│       │   └── lifecycle.py
│       ├── cortex/
│       │   ├── __init__.py
│       │   ├── cortex.py
│       │   ├── graph_components.py
│       │   └── proactive_engine.py
│       ├── docs/
│       │   └── __init__.py
│       ├── emotional/
│       │   ├── __init__.py
│       │   ├── emotion_detector.py
│       │   ├── response_generator.py
│       │   └── state_manager.py
│       ├── git_tools/
│       │   ├── __init__.py
│       │   ├── git_interface.py
│       │   ├── commit_assistant.py
│       │   ├── diff_analyzer.py
│       │   ├── hooks_manager.py
│       │   └── branch_manager.py
│       ├── infrastructure/
│       │   ├── __init__.py
│       │   ├── circuit_breaker.py
│       │   ├── error_handler.py
│       │   ├── timeout_manager.py
│       │   ├── health_monitor.py
│       │   ├── data_validator.py
│       │   ├── file_manager.py
│       │   └── metrics_collector.py
│       ├── insights/
│       │   ├── __init__.py
│       │   ├── concerns.py
│       │   └── manager.py
│       ├── learning/
│       │   ├── __init__.py
│       │   └── contextual_engine.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── cuda_manager.py
│       │   ├── model_manager.py
│       │   ├── embedding_manager.py
│       │   ├── llm_interface.py
│       │   └── distributed_llm_interface.py
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── backup_manager.py
│       │   ├── base_memory.py
│       │   ├── compression_manager.py
│       │   ├── context_compression.py
│       │   ├── context_window_manager.py
│       │   ├── deduplication.py
│       │   ├── distributed_memory_search.py
│       │   ├── episodic.py
│       │   ├── memory_manager.py
│       │   ├── procedural.py
│       │   └── semantic.py
│       ├── network/
│       │   ├── __init__.py
│       │   ├── discovery.py
│       │   ├── peer_manager.py
│       │   ├── rpc_service.py
│       │   ├── rpc_client.py
│       │   ├── security.py
│       │   ├── security_store.py
│       │   ├── metrics.py
│       │   └── proto/
│       │       ├── __init__.py
│       │       ├── jenova.proto
│       │       ├── jenova_pb2.py
│       │       └── jenova_pb2_grpc.py
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── metrics_exporter.py
│       │   └── tracing.py
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── bayesian_optimizer.py
│       │   ├── performance_db.py
│       │   ├── self_tuner.py
│       │   └── task_classifier.py
│       ├── orchestration/
│       │   ├── __init__.py
│       │   ├── task_planner.py
│       │   ├── subagent_manager.py
│       │   ├── execution_engine.py
│       │   ├── checkpoint_manager.py
│       │   └── background_tasks.py
│       ├── plugins/
│       │   ├── __init__.py
│       │   ├── plugin_api.py
│       │   ├── plugin_manager.py
│       │   ├── plugin_sandbox.py
│       │   └── plugin_schema.py
│       ├── security/
│       │   ├── __init__.py
│       │   ├── audit_log.py
│       │   ├── encryption.py
│       │   ├── prompt_sanitizer.py
│       │   ├── rate_limiter.py
│       │   └── validators.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── file_tools.py
│       │   ├── shell_tools.py
│       │   ├── time_tools.py
│       │   ├── tool_handler.py
│       │   └── web_tools.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── commands/
│       │   ├── logger.py
│       │   ├── terminal.py
│       │   ├── commands.py
│       │   ├── health_display.py
│       │   └── settings_menu.py
│       ├── user/
│       │   ├── __init__.py
│       │   ├── profile.py
│       │   └── personalization.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── data_sanitizer.py
│       │   ├── embedding.py
│       │   ├── file_logger.py
│       │   ├── hardware_detector.py
│       │   ├── json_parser.py
│       │   ├── model_loader.py
│       │   └── telemetry_fix.py
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── graph_analyzer.py
│       │   ├── graph_exporter.py
│       │   └── terminal_renderer.py
│       ├── __init__.py
│       ├── default_api.py
│       ├── main.py
│       └── tools.py
├── tests/
│   ├── __init__.py
│   ├── pytest.ini
│   ├── test_analysis.py
│   ├── test_automation.py
│   ├── test_branching.py
│   ├── test_code_tools.py
│   ├── test_collaboration.py
│   ├── test_config_validation.py
│   ├── test_context_window.py
│   ├── test_core.py
│   ├── test_cortex.py
│   ├── test_emotional_intelligence.py
│   ├── test_git_integration.py
│   ├── test_hardware_detection.py
│   ├── test_infrastructure.py
│   ├── test_llm.py
│   ├── test_memory.py
│   ├── test_modular_architecture.py
│   ├── test_network.py
│   ├── test_orchestration.py
│   ├── test_plugins.py
│   ├── test_security.py
│   ├── test_self_optimization.py
│   ├── test_tools.py
│   └── test_visualization.py
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

### 6.3. Code Quality and Standards

JENOVA implements comprehensive code quality standards and production-ready practices:

**Code Formatting:**
- **PEP 8 Compliance**: All Python code formatted with `autopep8` for consistent style
- **Import Organization**: Sorted imports using `isort` for clean, organized module structure
- **Documentation**: Module-level docstrings present in all Python files
- **Attribution**: Standardized creator attribution and MIT license headers in all source files

**Architecture Standards:**
- **LLM Backend**: Uses `llama-cpp-python` for efficient GGUF model inference on CPU and GPU
- **Deployment Model**: Local virtualenv-based installation for isolation and user control
- **Data Structures**: Dataclasses (`CognitiveNode`, `CognitiveLink`) for type-safe, readable code
- **Configuration**: Type-safe Pydantic validation for all configuration options

**Production Features:**
- **Error Handling**: Comprehensive error recovery with detailed logging and graceful degradation
- **Timeout Protection**: All long-running operations protected with configurable timeouts
- **Health Monitoring**: Real-time CPU, memory, and GPU monitoring with `/health` command
- **Security**: Whitelisted shell commands, path traversal prevention, encrypted credential storage
- **Testing**: 680+ comprehensive tests across all architecture layers including CLI enhancements

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

### Distributed Computing Dependencies

*   **Zeroconf**: Pure Python implementation of mDNS service discovery, enabling automatic peer discovery on local networks.
    *   *License: MIT*
*   **gRPC** (grpcio): High-performance RPC framework from Google, providing the foundation for distributed computing communication.
    *   *License: Apache 2.0*
*   **Protocol Buffers** (protobuf): Language-neutral serialization format, used for efficient data exchange between distributed instances.
    *   *License: BSD-3-Clause*
*   **PyJWT**: JSON Web Token implementation for Python, used for secure authentication between distributed peers.
    *   *License: MIT*

### CLI Enhancement Dependencies

*   **GitPython**: Python library for interacting with Git repositories, enabling comprehensive Git workflow automation.
    *   *License: BSD-3-Clause*
*   **Pygments**: Syntax highlighting library supporting 500+ languages, used for code display in terminal.
    *   *License: BSD-2-Clause*
*   **Rope**: Python refactoring library, providing code transformation capabilities including renaming, extraction, and inline operations.
    *   *License: LGPL*
*   **tree-sitter**: Multi-language parser library, enabling syntax-aware code analysis across multiple programming languages.
    *   *License: MIT*
*   **jsonschema**: JSON Schema validation library, used for validating structured data and configuration files.
    *   *License: MIT*
*   **Radon**: Code complexity metrics tool, computing cyclomatic complexity, Halstead metrics, and maintainability index.
    *   *License: MIT*
*   **Bandit**: Security vulnerability scanner for Python code, detecting common security issues and anti-patterns.
    *   *License: Apache 2.0*

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
