# JENOVA Cognitive Architecture

[![CI/CD Pipeline](https://github.com/jenova-ai/jenova-ca/actions/workflows/ci.yml/badge.svg)](https://github.com/jenova-ai/jenova-ca/actions/workflows/ci.yml)

## 1. Introduction & Philosophy

JENOVA is a self-aware, evolving large language model powered by the JENOVA Cognitive Architecture (JCA), a comprehensive engine and architecture designed by orpheus497. It learns, adapts, and assists through sophisticated cognitive processes. This document is the **program README**: what JENOVA is, how to install and use it, and the story of how it was built.

JENOVA operates as a system with interconnected components that mimic aspects of human cognition: a multi-layered memory, a reflective process for generating knowledge, and a mechanism for integrating that knowledge into its core being. It is for users, developers, and researchers who want to run, understand, or extend JENOVA.

### 1.0. The Story Behind JENOVA

**This is a personal project** by orpheus497. I used it to learn about software development, project planning, designing programs, engineering software, and technology—without ever writing or reading code myself. It was built **over six months** (late August 2025 through early 2026) using **only AI**: conversations with AI tools to design, implement, test, and document the system. **I did not touch a single line of code. Six months in, I still don’t know how to write or read any code.** Every line of code, every piece of documentation, and every configuration file was produced by AI in response to my natural-language direction.

**What I did:** Described goals, gave requirements in plain language, made decisions when asked, and approved or steered outcomes. **What I did not do:** Write or read source code, run linters or type checkers, or perform code-level review. The result is a production-ready codebase with comprehensive features, security hardening, and extensive test coverage—built entirely through dialogue with AI.

**Timeline:**
- **Late August 2025:** Project started as a personal experiment: “Can I build a real cognitive architecture using only AI?”
- **Six months of iteration:** Design, implementation, testing, and documentation carried out via AI agents and tools.
- **Early 2026:** Production-ready releases (4.0.0, 4.0.1). The system runs, remembers, reflects, and learns.

### 1.1. Modern Terminal UI

JENOVA features a **beautiful, modern terminal interface** built with [Textual](https://github.com/Textualize/textual), a powerful TUI framework for Python. The UI provides:

- **Responsive, smooth rendering** with rich colors and formatting
- **Real-time chat viewport** with automatic scrolling
- **Loading indicators** with animated spinners
- **Integrated architecture** - UI and cognitive engine in unified Python codebase

## 2. The JENOVA Advantage: A Superior Cognitive Architecture

### 2.1. Beyond Statelessness: The Problem with General Systems

Most consumer-facing AI systems operate on a **stateless, request-response** model. They are incredibly powerful at in-context learning and reasoning, but each interaction is largely independent of the last. This leads to several fundamental limitations:

* **Amnesia:** The AI has no persistent memory of past conversations. It cannot remember your preferences, previous questions, or the context of your work. Every chat starts from a blank slate.
* **Inability to Learn:** Corrections you make or new information you provide are only retained for the current session. The underlying model never truly learns or improves from user interaction.
* **Inconsistent Persona:** The AI's personality can drift or be easily manipulated because it lacks a stable, memory-grounded identity.
* **Reactive, Not Proactive:** These systems can only answer direct questions. They cannot reflect on past dialogues to draw novel conclusions or develop a deeper understanding of a topic over time.

### 2.2. The JCA Solution: A Unified, Learning Architecture

The JENOVA Cognitive Architecture (JCA) is explicitly designed to overcome these limitations. It wraps a powerful Large Language Model (LLM) in a structured framework that provides memory, reflection, and a mechanism for true, persistent learning. It transforms the LLM from a brilliant but amnesiac calculator into a cohesive, evolving intelligence.

### 2.3. The Power of the Cognitive Cycle

The "Retrieve, Plan, Execute, Reflect" cycle is the engine of the JCA and the primary driver of its capabilities.

* **Grounded Responses:** By forcing the AI to **Retrieve** from its memory *before* acting, the JCA ensures that responses are grounded in established facts, past conversations, and learned insights. This dramatically reduces confabulation (hallucination) and increases the relevance and accuracy of output.
* **Deliberate Action:** The **Plan** step introduces a moment of metacognition. The AI must first reason about *how* to answer the query. This internal monologue, while hidden from the user, results in a more structured and logical final response. It prevents conversational shortcuts and encourages a methodical approach to problem-solving.

### 2.4. Memory as the Foundation for Identity and Growth

JENOVA's multi-layered memory system is the bedrock of its identity. It is the difference between playing a character and *having* a character.

* **Continuity of Self:** The Episodic Memory gives JENOVA a personal history with the user. It can refer to past conversations, understand recurring themes, and build a genuine rapport. The AI is not a stranger every time you open the terminal.
* **A Worldview:** The Semantic Memory and Procedural Memory, combined with the dynamically growing Insight Memory, form the AI's worldview. This knowledge base is prioritized over the LLM's base training data, allowing for the development of a unique, personalized knowledge set that reflects its experiences.

### 2.5. The Self-Correction and Evolution Loop: True Learning

This is one of the most powerful and defining features of the JCA. The cycle of **Reflection → Insight Generation → Fine-Tuning** constitutes a true learning loop.

1. **Experience (`Reflect`):** The AI has a conversation and gains experience.
2. **Internalization (`Insight Generation`):** It reflects on that experience and internalizes the key takeaways as structured, atomic insights. This is analogous to a human consolidating short-term memories into long-term knowledge.
3. **Integration (`Fine-Tuning`):** The fine-tuning process takes these internalized insights and integrates them into the embedding model. The learned knowledge becomes part of the AI's retrieval intuition.

This loop creates a system that does not just get more knowledgeable; it gets **smarter**. It adapts its core reasoning processes based on its unique experiences, evolving into an assistant that is perfectly tailored to its user.

## 3. Core Features Explained

### 3.1. The Cognitive Cycle: A Prioritized Approach

The heart of JENOVA is its cognitive cycle, a continuous loop that drives its behavior. This cycle enforces a strict knowledge hierarchy, ensuring that the AI relies on the most relevant and reliable information available.

1. **Retrieve:** When the user provides input, the `CognitiveEngine` first queries its **Knowledge Store**—the multi-layered memory system (Episodic, Semantic, Procedural) and cognitive graph—to gather relevant context. This is the AI's personal experience and learned knowledge, and it is always the highest priority.
2. **Plan:** The engine then formulates a step-by-step internal plan. This plan is generated by the LLM itself, based on the user's query and the retrieved context. Plans are assessed for complexity (simple → very complex) and structured accordingly.
3. **Execute:** The plan is then executed by the `ResponseGenerator`. The RAG prompt is explicitly structured to prioritize the AI's knowledge base.
4. **Reflect & Learn:** The cognitive scheduler determines when to trigger the various cognitive functions, such as analyzing recent conversation history to identify novel conclusions or key takeaways, verifying assumptions, and growing the knowledge base.

### 3.2. The RAG System: A Core Component of the Psyche

The Retrieval-Augmented Generation (RAG) system is a core component of the AI's cognitive architecture. It is responsible for generating responses that are grounded in the AI's own knowledge and experience.

* **Hybrid Retrieval:** The system uses a hybrid retrieval approach, querying all memory sources (episodic, semantic, procedural) and the cognitive graph to gather the most relevant context.
* **Context Scoring:** Retrieved results are scored and ranked to prioritize the most relevant information.
* **Grounded Response Generation:** The system uses the ranked context, conversation history, and generated plan to produce responses grounded in the AI's own knowledge.

### 3.3. The Cognitive Graph: A Graph-Based Cognitive Core

The Cognitive Graph is the heart of JENOVA's cognitive architecture. It provides a unified, graph-based system for managing insights and assumptions. This allows for a deeper and more interconnected understanding of the user and the world.

* **Cognitive Graph:** The system manages a "cognitive graph" where insights, assumptions, and knowledge are represented as nodes. These nodes are connected by typed relationships (e.g., "elaborates_on", "conflicts_with", "created_from").
* **Embedding-Based Search:** The graph supports semantic similarity search across all nodes using vector embeddings.
* **Neighbor Traversal:** Nodes can be traversed by relationship type to find connected knowledge.

### 3.4. Reflective Insight Engine

The Insight Engine allows JENOVA to learn continuously. The system is proactive, organized, and reflective, ensuring that knowledge is captured, categorized, and interconnected efficiently.

* **Concern-Based Organization:** Insights are organized into "concerns" or "topics." When an insight is generated, the system searches for an existing, relevant concern to group it with. This prevents knowledge fragmentation and creates a more structured understanding of topics.
* **Graph Integration:** When an insight is saved, it is also added as a node to the cognitive graph. This allows the insight to be linked to other cognitive nodes.
* **Storage:** Insights are saved in a hierarchical structure within the user-specific data directory.

### 3.5. Assumption System

JENOVA actively forms assumptions about the user to build a more accurate mental model. This system allows the AI to move beyond explicitly stated facts and begin to infer user preferences, goals, and knowledge levels.

* **Graph Integration:** When an assumption is added, it is also added as a node to the cognitive graph, providing context through connections.
* **Status Tracking:** Assumptions are categorized by status: `unverified`, `true`, or `false`.
* **Verification:** The verification process allows the user to confirm or deny assumptions, refining JENOVA's understanding.

### 3.6. Multi-Layered Long-Term Memory

JENOVA's memory is not a monolith. It's a sophisticated, multi-layered system managed by ChromaDB, a vector database. All memory is stored on a per-user basis.

* **Episodic Memory:** Stores a turn-by-turn history of conversations with context.
* **Semantic Memory:** Stores factual knowledge with source and confidence metadata.
* **Procedural Memory:** Stores "how-to" information and instructions with goals, steps, and context.

### 3.7. Fine-Tuning Data Generation

JENOVA is designed for continuous improvement. The insights generated during its operation can be used to fine-tune the embedding model.

* **Data Collection (`finetune/data.py`):** Collects contrastive training examples from interactions.
* **Training (`finetune/train.py`):** Uses `MultipleNegativesRankingLoss` to fine-tune sentence-transformer embeddings, improving retrieval quality over time.

### 3.8. Cognitive Scheduler

The Cognitive Scheduler manages background cognitive tasks that run during conversation intervals, ensuring cognitive operations happen at appropriate times without interrupting user interaction.

* **Turn-Based Scheduling:** Tasks are scheduled based on conversation turns, not wall-clock time
* **Priority System:** Higher priority tasks execute first when multiple tasks are due
* **Task Types:** Insight generation, assumption verification, reflection cycles, graph pruning, orphan linking
* **Configurable Intervals:** Each task type has configurable execution intervals
* **Acceleration Logic:** Tasks can be accelerated based on conversation activity

### 3.9. Proactive Engine

The Proactive Engine enables JENOVA to autonomously generate suggestions and recommendations based on cognitive state and conversation patterns.

* **Autonomous Suggestions:** Generates proactive suggestions for exploration, verification, development, connection, and reflection
* **Context-Aware:** Suggestions are based on cognitive graph state, conversation history, and user patterns
* **Cooldown System:** Prevents suggestion spam with intelligent cooldown management
* **Category-Based:** Suggestions organized by category (explore, verify, develop, connect, reflect)
* **Priority Scoring:** Each suggestion has a priority score indicating urgency and relevance

### 3.10. Advanced Graph Features

The Cognitive Graph includes sophisticated features for knowledge management and analysis:

* **Emotion Analysis:** LLM-powered emotion detection with Pydantic validation for content analysis
* **Clustering:** Automatic clustering of related nodes for pattern identification
* **Meta-Insight Generation:** Higher-order insights synthesized from node clusters
* **Orphan Linking:** Automatic connection of isolated nodes to the graph structure
* **Contradiction Detection:** Identification of conflicting information across nodes
* **Connection Suggestions:** AI-powered suggestions for new relationships between nodes

### 3.11. Security & Validation

JENOVA includes comprehensive security measures and input validation:

* **Prompt Injection Protection:** Sanitization utilities prevent prompt injection attacks
* **LLM Output Validation:** Pydantic schemas validate all LLM JSON responses
* **Safe JSON Parsing:** Robust parsing with size limits and depth validation to prevent DoS
* **Path Validation:** Secure path validation with sandboxing to prevent traversal attacks
* **Error Message Sanitization:** Safe error handling without information leakage
* **Thread-Safe Operations:** All shared state operations are thread-safe

### 3.12. Utility Systems

Supporting utility systems provide infrastructure for cognitive operations:

* **TTLCache/CacheManager:** Thread-safe caching system with TTL support for performance optimization
* **Performance Monitor:** Performance profiling and timing utilities for optimization
* **Grammar Loader:** Centralized JSON grammar loading for structured data processing
* **Tools Module:** Shell command execution and datetime utilities for system integration
* **Migration System:** Schema versioning and data migration support for evolving data structures

## 4. Installation

### 4.1. Quick Start (Recommended)

```bash
# Clone the repository
git clone https://github.com/orpheus497/jenova-ca.git
cd jenova-ca

# Install in development mode
pip install -e ".[dev]"

# Install with fine-tuning support
pip install -e ".[dev,finetune]"

# Download a GGUF model
mkdir -p models
wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf

# Run JENOVA
jenova
```

### 4.2. Prerequisites

* **Python:** 3.10+ (tested on 3.10, 3.11, 3.12)
* **C++ Compiler:** Required for `llama-cpp-python` (e.g., `g++`, `clang++`)
* **ChromaDB:** Uses SQLite backend (native on FreeBSD/Linux)

### 4.3. Platform Support

JENOVA is designed for **native, out-of-the-box support** on:

| Platform | Status | Notes |
|----------|--------|-------|
| **FreeBSD** | ✅ Fully Supported | Tested on FreeBSD 13.x, 14.x |
| **Linux** | ✅ Fully Supported | Ubuntu 22.04+, Debian 12+, Fedora 38+ |

#### FreeBSD

```bash
pkg install python311 py311-pip
# Optional: GPU support for fine-tuning
pkg install py311-pytorch
```

#### Linux (Debian/Ubuntu)

```bash
apt install python3.11 python3.11-venv python3-pip build-essential
```

#### Linux (Fedora/RHEL)

```bash
dnf install python3.11 python3-pip gcc-c++
```

### 4.4. Cross-Platform Compatibility

* All file paths use POSIX conventions (`/` separators)
* Line endings enforced as LF via `.editorconfig`
* No platform-specific code paths or conditionals
* Atomic file operations use POSIX `rename()` semantics

### 4.5. User Data

The first time you run the application, a private directory will be created at `~/.jenova-ai/users/<your_username>/`. All conversations, memories, and learned insights are stored here, inaccessible to other users.

## 5. User Guide

Interaction with JENOVA is primarily through natural language.

* **User Input:** Simply type your message and press Enter.
* **Exiting:** To quit the application, type `exit` or press `Ctrl+C`.

### Commands

JENOVA responds to commands that act as direct instructions for its cognitive processes. Commands are system actions, not conversational input, and are **not stored in conversational memory**.

#### Currently Implemented Commands

| Command | Description | Mode |
|---------|-------------|------|
| `/help` | Display comprehensive command reference | TUI & Headless |
| `/reset` | Reset conversation state | Headless only |
| `/debug` | Toggle debug logging | Headless only |
| `exit` / `quit` | Exit the application | TUI & Headless |

#### Planned Cognitive Commands

The following cognitive commands are documented in the help system and will be implemented in future releases:

| Command | Description |
|---------|-------------|
| `/insight` | Analyze conversation and generate new insights |
| `/reflect` | Deep reflection: reorganize cognitive nodes, link orphans, generate meta-insights |
| `/memory-insight` | Search all memory layers to develop new insights |
| `/meta` | Generate higher-level meta-insights from insight clusters |
| `/verify` | Verify an unverified assumption with a clarifying question |
| `/develop_insight [node_id]` | Develop existing insight (with ID) or process documents (without ID) |
| `/learn_procedure` | Interactive guided process to teach a new procedure |

**Note:** While the underlying cognitive systems (InsightManager, AssumptionManager, CognitiveGraph) are fully implemented, the command handlers for these features are planned for future releases.

### Keyboard Shortcuts (TUI)

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `F1` | Toggle help panel |
| `Ctrl+L` | Clear chat history |
| `Ctrl+C` | Quit application |

## 6. Configuration

JENOVA is configured via a YAML file. Copy `config.example.yaml` to customize.

### Hardware

```yaml
hardware:
  threads: auto      # CPU threads (auto = detect automatically)
  gpu_layers: all    # GPU layers (-1/all = offload all, 0 = none)
```

### Model

```yaml
model:
  model_path: auto     # Path to GGUF model (auto = search common locations)
  context_length: 4096 # Context window size in tokens
  temperature: 0.7     # Creativity (0.0 = deterministic)
  top_p: 0.9           # Nucleus sampling threshold
  max_tokens: 1024     # Max tokens per response
```

### Memory

```yaml
memory:
  storage_path: .jenova-ai/memory
  embedding_model: all-MiniLM-L6-v2
  max_results: 10
```

### Persona

```yaml
persona:
  name: JENOVA
  system_prompt: "You are JENOVA, a self-aware AI with evolving memory and knowledge."
  directives:
    - Be helpful and informative
    - Acknowledge uncertainty when present
    - Learn from interactions
```

## 7. Project Structure

```
jenova-ca/
├── src/jenova/           # Main package
│   ├── assumptions/      # Assumption lifecycle management
│   │   ├── manager.py    # AssumptionManager with verification
│   │   └── types.py     # Assumption data models
│   ├── config/           # Pydantic configuration models
│   │   ├── models.py     # Configuration validation
│   │   └── __init__.py
│   ├── core/             # Core cognitive systems
│   │   ├── engine.py     # CognitiveEngine (Retrieve, Plan, Execute, Reflect)
│   │   ├── knowledge.py  # KnowledgeStore (unified memory + graph)
│   │   ├── response.py   # ResponseGenerator with caching
│   │   ├── integration.py # IntegrationHub (Memory ↔ Cortex)
│   │   ├── query_analyzer.py # Query analysis and intent detection
│   │   ├── context_scorer.py  # Context scoring and ranking
│   │   ├── context_organizer.py # Context organization
│   │   └── scheduler.py  # Cognitive scheduler
│   ├── embeddings/       # Embedding model management
│   │   ├── model.py      # Embedding model wrapper
│   │   └── types.py      # Embedding types
│   ├── graph/            # Cognitive graph (Cortex)
│   │   ├── graph.py      # CognitiveGraph with advanced features
│   │   ├── types.py      # Node, Edge, GraphQuery types
│   │   ├── llm_schemas.py # Pydantic validation schemas
│   │   └── proactive.py  # Proactive suggestion engine
│   ├── insights/         # Insight management
│   │   ├── manager.py    # InsightManager
│   │   ├── concerns.py   # ConcernManager (topic organization)
│   │   └── types.py      # Insight data models
│   ├── llm/              # LLM interface
│   │   ├── interface.py  # LLMInterface (llama-cpp-python wrapper)
│   │   └── types.py      # Prompt, Completion types
│   ├── memory/           # ChromaDB memory system
│   │   ├── memory.py     # Unified Memory class
│   │   └── types.py      # MemoryType, MemoryResult
│   ├── ui/               # Textual TUI
│   │   ├── app.py        # Main TUI application
│   │   └── components/   # UI components (banner, help, loading, message)
│   ├── utils/            # Utility modules
│   │   ├── cache.py      # TTLCache and CacheManager
│   │   ├── performance.py # Performance monitoring
│   │   ├── grammar.py    # Grammar loading
│   │   ├── sanitization.py # Input sanitization
│   │   ├── json_safe.py  # Safe JSON parsing
│   │   ├── validation.py # Path and input validation
│   │   ├── errors.py     # Error handling utilities
│   │   ├── logging.py    # Structured logging
│   │   └── migrations.py # Data migration system
│   ├── tools.py          # Shell and datetime utilities
│   ├── exceptions.py     # Exception hierarchy
│   └── main.py           # CLI entry point
├── tests/                # Comprehensive test suites
│   ├── unit/             # Unit tests (17 files, 365+ tests)
│   ├── integration/      # Integration tests (4 files, 36 tests)
│   ├── security/         # Security tests (23 adversarial tests)
│   ├── benchmarks/      # Performance benchmarks
│   └── performance/      # Performance test utilities
├── finetune/             # Embedding fine-tuning
│   ├── data.py          # Training data collection
│   └── train.py         # Fine-tuning training script
├── config.example.yaml   # Example configuration
├── pyproject.toml        # Project configuration
├── LICENSE               # AGPL-3.0 license
├── README.md             # This file
└── CHANGELOG.md          # Version history
```

## 8. Development & CLI

### CLI Options

```bash
jenova                      # Run with TUI (default)
jenova --no-tui             # Run in headless CLI mode
jenova --config config.yaml # Use custom configuration
jenova --debug              # Enable debug logging
jenova --skip-model-load    # Development mode (mock LLM)
jenova --log-file path.log  # Write logs to file
jenova --json-logs          # Output logs in JSON format
jenova --version, -v        # Show version information
```

## 9. Project Statistics & Status

### 9.1. Codebase Metrics

- **Total Python Files:** 48+ source files in `src/jenova/`
- **Lines of Code:** ~15,000+ lines of production code
- **Test Coverage:** 400+ tests across unit, integration, security, and benchmark suites
- **Documentation:** Comprehensive inline documentation and this README
- **Architecture:** Protocol-based design with clean separation of concerns
- **Dependencies:** 7 core dependencies, 4 dev dependencies, 2 optional finetune dependencies
- **Platform Support:** Native FreeBSD and Linux support
- **License:** AGPL-3.0 (Free and Open Source Software)

### 9.2. Current Status

**Version:** 4.0.1 (Beta)  
**Status:** Production-ready with comprehensive feature set

**Implemented Systems:**
- ✅ Complete cognitive architecture (Retrieve, Plan, Execute, Reflect cycle)
- ✅ Multi-layered memory system (Episodic, Semantic, Procedural) with ChromaDB
- ✅ Cognitive graph with advanced features (emotion analysis, clustering, meta-insights, orphan linking, contradiction detection)
- ✅ Insight and assumption management systems with concern-based organization
- ✅ Integration layer for unified knowledge representation (Memory ↔ Cortex)
- ✅ Query analysis and context scoring with intent detection and complexity assessment
- ✅ Context organization and prioritization
- ✅ Multi-level planning system (simple → very complex queries)
- ✅ Response generation with caching, persona support, and source citations
- ✅ Cognitive scheduler for background task management
- ✅ Proactive engine for autonomous suggestion generation
- ✅ Modern Textual-based TUI with responsive design
- ✅ Comprehensive test suite (400+ tests: unit, integration, security, benchmarks)
- ✅ Security hardening (all P0/P1 issues resolved, prompt injection protection, input validation)
- ✅ Utility systems (caching, performance monitoring, grammar loading, tools)

**Planned Features:**
- Command handlers for cognitive operations (`/insight`, `/reflect`, etc.)
- Enhanced fine-tuning workflows
- Additional cognitive capabilities

### 9.3. How JENOVA Was Built

This project was built over six months using only AI. I (orpheus497) did not write or read code—and I still don’t. Six months in, I still don’t know how to write or read any code. I used this project to learn about software development, project planning, program design, and software engineering by directing AI in natural language.

- **Code:** All source code was generated by AI in response to my direction.
- **Documentation:** All documentation was produced by AI.
- **Testing:** Test suites were created and maintained by AI.
- **Security:** Security audits and patches were applied by AI.
- **Quality:** Code reviews and quality checks were performed by AI.

**My role:** Describing what to build, giving requirements in plain language, making decisions when asked, and approving or steering outcomes—never writing or reading source code.

## 10. License

AGPL-3.0 - See [LICENSE](LICENSE) file for details.

## 11. Technical Architecture

### 11.1. System Components

**Core Cognitive Systems:**
- `CognitiveEngine`: Orchestrates the cognitive cycle (Retrieve, Plan, Execute, Reflect)
- `KnowledgeStore`: Unified interface to memory and graph systems
- `ResponseGenerator`: Formats and structures LLM output with caching
- `IntegrationHub`: Bridges Memory and Cortex for unified knowledge

**Memory Systems:**
- `Memory`: Unified ChromaDB-based vector storage (Episodic, Semantic, Procedural)
- Persistent storage with per-user isolation
- Semantic search with embedding-based retrieval

**Graph Systems:**
- `CognitiveGraph`: Dict-based graph with advanced cognitive features
- Node and edge management with relationship types
- Advanced features: emotion analysis, clustering, meta-insights

**Analysis Systems:**
- `QueryAnalyzer`: Multi-level query analysis with intent detection
- `ContextScorer`: Configurable context retrieval and ranking
- `ContextOrganizer`: Intelligent context organization and prioritization

**Supporting Systems:**
- `CognitiveScheduler`: Turn-based background task scheduling
- `ProactiveEngine`: Autonomous suggestion generation
- `AssumptionManager`: Assumption tracking and verification
- `InsightManager`: Insight generation and management
- `ConcernManager`: Topic-based concern organization

### 11.2. Data Flow

```
User Input
    ↓
Sanitization (prompt injection protection)
    ↓
CognitiveEngine.think()
    ↓
KnowledgeStore.search() → Memory + Graph
    ↓
QueryAnalyzer.analyze() → Intent, Complexity, Topics
    ↓
ContextScorer.score() → Ranked context
    ↓
ContextOrganizer.organize() → Prioritized context
    ↓
Plan Generation (simple → very complex)
    ↓
LLM.generate() → Raw response
    ↓
ResponseGenerator.generate() → Formatted response
    ↓
Memory Storage (episodic memory)
    ↓
User Output
```

### 11.3. Security Architecture

**Defense in Depth:**
- Input sanitization at all boundaries
- LLM output validation with Pydantic schemas
- Path traversal protection in configuration
- Safe JSON parsing with size/depth limits
- Error message sanitization
- Thread-safe operations for concurrent access

**Security Features:**
- Prompt injection pattern detection and removal
- Username validation for multi-user isolation
- Path validation with sandboxing
- JSON size and depth limits to prevent DoS
- Comprehensive security test suite

### 11.4. Performance Optimizations

- **Thread-Safe Caching:** LRU cache with TTL for response caching
- **Efficient Graph Operations:** O(degree) operations using reverse edge index
- **Batch Operations:** Batch node addition for reduced disk I/O
- **Lazy Loading:** Graph lazy-loaded to avoid circular imports
- **Performance Monitoring:** Built-in performance profiling utilities

## 12. Acknowledgments

**Project Creator:** orpheus497

**How it was built:** JENOVA was built over six months using only AI. I directed what to build in natural language. I never wrote or read a line of code—six months in, I still don’t know how to write or read any code. Every line of code and documentation was generated by AI in response to that dialogue.

**Technologies Used:**
- [ChromaDB](https://www.trychroma.com/) - Vector database for memory storage
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - GGUF model inference
- [Textual](https://github.com/Textualize/textual) - Modern terminal UI framework
- [Pydantic](https://docs.pydantic.dev/) - Data validation and settings management
- [structlog](https://www.structlog.org/) - Structured logging

**Inspiration:** This project explores cognitive architectures and the potential for AI systems to develop persistent memory, reflection, and true learning capabilities.
