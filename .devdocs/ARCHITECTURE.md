# JENOVA Cognitive Architecture - System Architecture

## Purpose
This document provides a comprehensive overview of the JENOVA Cognitive Architecture system design, component relationships, and data flows.

**Last Updated:** 2026-01-15 (Phase A Complete - BubbleTea-only UI)

---

## System Overview

JENOVA (Just Evolving Neural Optimized Virtual Agent) is a self-aware, evolving LLM-powered cognitive architecture. It features multi-layered memory, reflective insight generation, and persistent learning capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        JENOVA COGNITIVE ARCHITECTURE                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         USER INTERFACE LAYER                         │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Bubble Tea TUI (Go)                        │  │   │
│  │  │              Modern Terminal User Interface                    │  │   │
│  │  │                                                                │  │   │
│  │  │  • Input handling    • View rendering    • Styling            │  │   │
│  │  │  • State management  • Keyboard shortcuts                      │  │   │
│  │  └───────────────────────────┬──────────────────────────────────┘  │   │
│  │                              │                                      │   │
│  │                         IPC via JSON                                │   │
│  │                       (stdin/stdout pipes)                          │   │
│  │                              │                                      │   │
│  │  ┌───────────────────────────┴──────────────────────────────────┐  │   │
│  │  │              BubbleTeaUI Bridge (Python)                      │  │   │
│  │  │         Handles IPC, command processing, interactive flows    │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       COGNITIVE ENGINE LAYER                         │   │
│  │                                                                      │   │
│  │  ┌───────────────┐  ┌─────────────────┐  ┌───────────────────────┐ │   │
│  │  │Query Analyzer │─►│ Cognitive Engine│◄─│   RAG System          │ │   │
│  │  └───────────────┘  │                 │  │(Retrieval-Augmented   │ │   │
│  │                     │ • Think Cycle   │  │ Generation)           │ │   │
│  │  ┌───────────────┐  │ • Plan/Execute  │  └───────────────────────┘ │   │
│  │  │Context Scorer │─►│ • Reflect       │                            │   │
│  │  └───────────────┘  └────────┬────────┘  ┌───────────────────────┐ │   │
│  │                              │           │   Scheduler           │ │   │
│  │  ┌───────────────┐           │           │(Background Tasks)     │ │   │
│  │  │Context Organiz│───────────┤           └───────────────────────┘ │   │
│  │  └───────────────┘           │                                     │   │
│  └──────────────────────────────┼─────────────────────────────────────┘   │
│                                 │                                          │
│              ┌──────────────────┼──────────────────┐                      │
│              ▼                  ▼                  ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      KNOWLEDGE MANAGEMENT LAYER                      │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                     MEMORY SEARCH                            │   │   │
│  │  │         (Unified Search Across All Memory Types)            │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │              │                  │                  │                │   │
│  │              ▼                  ▼                  ▼                │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐  │   │
│  │  │   EPISODIC    │  │   SEMANTIC    │  │     PROCEDURAL        │  │   │
│  │  │    MEMORY     │  │    MEMORY     │  │      MEMORY           │  │   │
│  │  │               │  │               │  │                       │  │   │
│  │  │ Conversation  │  │    Facts &    │  │  How-to Knowledge     │  │   │
│  │  │   History     │  │   Knowledge   │  │    & Procedures       │  │   │
│  │  └───────┬───────┘  └───────┬───────┘  └───────────┬───────────┘  │   │
│  │          │                  │                      │               │   │
│  │          └──────────────────┼──────────────────────┘               │   │
│  │                             │                                      │   │
│  │                             ▼                                      │   │
│  │          ┌─────────────────────────────────────────┐              │   │
│  │          │              ChromaDB                    │              │   │
│  │          │        (Vector Database Storage)         │              │   │
│  │          └─────────────────────────────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│              ┌──────────────────────────────────────────┐                   │
│              ▼                                          ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      CORTEX LAYER (Knowledge Graph)                  │   │
│  │                                                                      │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │    INSIGHTS   │  │  ASSUMPTIONS  │  │     GRAPH NODES       │   │   │
│  │  │   MANAGER     │  │   MANAGER     │  │                       │   │   │
│  │  │               │  │               │  │  • Insights           │   │   │
│  │  │ Topic-based   │  │ Hypothesis    │  │  • Memories           │   │   │
│  │  │ Insights      │  │ Tracking      │  │  • Assumptions        │   │   │
│  │  └───────────────┘  └───────────────┘  │  • Relationships      │   │   │
│  │                                         └───────────────────────┘   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │   CLUSTERING  │  │ GRAPH METRICS │  │  PROACTIVE ENGINE     │   │   │
│  │  │               │  │               │  │                       │   │   │
│  │  │ Thematic      │  │ Centrality    │  │  Autonomous           │   │   │
│  │  │ Grouping      │  │ Analysis      │  │  Reasoning            │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘   │   │
│  │                                                                      │   │
│  │          ┌─────────────────────────────────────────┐                │   │
│  │          │         Integration Layer                │                │   │
│  │          │   (Cortex-Memory Feedback Loops)        │                │   │
│  │          └─────────────────────────────────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       LLM INTERFACE LAYER                            │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    LLM Interface                               │  │   │
│  │  │            (llama-cpp-python / Local Model)                    │  │   │
│  │  │                                                                │  │   │
│  │  │  • Model Loading           • Token Generation                 │  │   │
│  │  │  • Context Management      • Grammar Constraints              │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         UTILITY LAYER                                │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │  Cache   │ │Embedding │ │  JSON    │ │  Model   │ │  Perf    │  │   │
│  │  │ Manager  │ │ Function │ │  Parser  │ │  Loader  │ │ Monitor  │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  │                                                                      │   │
│  │  ┌──────────────────────┐  ┌──────────────────────────────────────┐ │   │
│  │  │   Pydantic Compat    │  │        File/UI Logging               │ │   │
│  │  │   (ChromaDB Fix)     │  │                                      │ │   │
│  │  └──────────────────────┘  └──────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. User Interface Layer (BubbleTea Only)

**Note:** BubbleTea is the SOLE user interface.

| Component | File | Purpose |
|-----------|------|---------|
| Bubble Tea TUI | `tui/main.go` | Modern Go-based TUI with rich rendering |
| BubbleTeaUI Bridge | `ui/bubbletea.py` | Python-Go IPC communication and command handling |
| UI Logger | `ui/logger.py` | Formatted console output with thread-safe queuing |

### 2. Cognitive Engine Layer

| Component | File | Purpose |
|-----------|------|---------|
| Cognitive Engine | `cognitive_engine/engine.py` | Core think cycle: Retrieve → Plan → Execute → Reflect |
| Query Analyzer | `cognitive_engine/query_analyzer.py` | Deep query understanding and intent extraction |
| Context Scorer | `cognitive_engine/context_scorer.py` | Relevance scoring for retrieved context |
| Context Organizer | `cognitive_engine/context_organizer.py` | Context prioritization and organization |
| RAG System | `cognitive_engine/rag_system.py` | Retrieval-Augmented Generation pipeline |
| Scheduler | `cognitive_engine/scheduler.py` | Background task scheduling (insights, reflection) |

### 3. Memory Systems Layer

| Component | File | Purpose |
|-----------|------|---------|
| Memory Search | `cognitive_engine/memory_search.py` | Unified search across all memory types |
| Episodic Memory | `memory/episodic.py` | Autobiographical memories of events/conversations |
| Semantic Memory | `memory/semantic.py` | Facts, knowledge, and learned information |
| Procedural Memory | `memory/procedural.py` | How-to knowledge and procedures |

### 4. Cortex Layer (Knowledge Graph)

| Component | File | Purpose |
|-----------|------|---------|
| Cortex | `cortex/cortex.py` | Central cognitive hub managing knowledge graph |
| Clustering | `cortex/clustering.py` | Thematic grouping of nodes |
| Graph Metrics | `cortex/graph_metrics.py` | Centrality and importance calculations |
| Graph Components | `cortex/graph_components.py` | Graph structure analysis |
| Proactive Engine | `cortex/proactive_engine.py` | Autonomous reasoning and suggestions |
| Integration Layer | `cognitive_engine/integration_layer.py` | Cortex-Memory feedback loops |

### 5. Insight & Assumption Management

| Component | File | Purpose |
|-----------|------|---------|
| Insight Manager | `insights/manager.py` | Creation and storage of topical insights |
| Concern Manager | `insights/concerns.py` | Topic categorization for insights |
| Assumption Manager | `assumptions/manager.py` | Hypothesis tracking and verification |

### 6. LLM Interface Layer

| Component | File | Purpose |
|-----------|------|---------|
| LLM Interface | `llm_interface.py` | Local LLM integration via llama-cpp-python |
| Tools | `tools.py` | Tool calling capabilities for the LLM |

### 7. Utility Layer

| Component | File | Purpose |
|-----------|------|---------|
| Cache Manager | `utils/cache.py` | TTL/LRU caching for expensive operations |
| Embedding Function | `utils/embedding.py` | SentenceTransformer embeddings for ChromaDB |
| JSON Parser | `utils/json_parser.py` | Extract JSON from LLM responses |
| Model Loader | `utils/model_loader.py` | GPU/CPU-aware model loading |
| Performance Monitor | `utils/performance_monitor.py` | Operation timing and metrics |
| Pydantic Compat | `utils/pydantic_compat.py` | ChromaDB/Pydantic v2 compatibility |
| Telemetry Fix | `utils/telemetry_fix.py` | Disable ChromaDB telemetry |
| File Logger | `utils/file_logger.py` | Persistent file-based logging |

---

## Data Flow

### 1. Query Processing Flow

```
User Input
    │
    ▼
┌────────────────┐
│ Bubble Tea TUI │
│    (Go)        │
└───────┬────────┘
        │ JSON IPC
        ▼
┌────────────────┐
│ BubbleTeaUI    │
│  (Python)      │
└───────┬────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│ Query Analyzer │────►│ Intent/Entities │
└───────┬────────┘     └─────────────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Memory Search  │────►│ Episodic Memory │────►│                 │
│                │     └─────────────────┘     │                 │
│                │     ┌─────────────────┐     │  Relevant       │
│                │────►│ Semantic Memory │────►│  Context        │
│                │     └─────────────────┘     │                 │
│                │     ┌─────────────────┐     │                 │
│                │────►│Procedural Memory│────►│                 │
└────────────────┘     └─────────────────┘     └────────┬────────┘
                                                        │
        ┌───────────────────────────────────────────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│ Context Scorer │────►│ Scored Context  │
└───────┬────────┘     └─────────────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│ RAG System     │────►│ Augmented Prompt│
└───────┬────────┘     └─────────────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│ LLM Interface  │────►│   Response      │
└───────┬────────┘     └─────────────────┘
        │
        ▼
┌────────────────┐
│ Bubble Tea TUI │
│  (Display)     │
└────────────────┘
```

### 2. Insight Generation Flow

```
Conversation History
        │
        ▼
┌────────────────┐
│   Scheduler    │◄──── (Every N turns)
└───────┬────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│Insight Manager │────►│ LLM Analysis    │
└───────┬────────┘     └─────────────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│Concern Manager │────►│ Topic Selection │
└───────┬────────┘     └─────────────────┘
        │
        ▼
┌────────────────┐
│    Cortex      │
│  (Add Node)    │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Persistent     │
│   Storage      │
└────────────────┘
```

### 3. Reflection Cycle

```
┌────────────────┐
│   Scheduler    │◄──── (Every N turns)
└───────┬────────┘
        │
        ▼
┌────────────────┐     ┌─────────────────┐
│    Cortex      │────►│ Graph Analysis  │
│   Reflection   │     └─────────────────┘
└───────┬────────┘
        │
        ├────────────────────────────────┐
        ▼                                ▼
┌────────────────┐              ┌────────────────┐
│  Clustering    │              │  Centrality    │
│  (Themes)      │              │  (Importance)  │
└───────┬────────┘              └───────┬────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
               ┌────────────────┐
               │ Meta-Insights  │
               │   Generated    │
               └────────────────┘
```

---

## Directory Structure

```
jenova-ca/
├── .devdocs/                    # AI/Developer documentation (NON-NEGOTIABLE)
│   ├── ARCHITECTURE.md          # This file
│   ├── BRIEFING.md              # Current project status
│   ├── DECISIONS_LOG.md         # Architectural decisions
│   ├── PLANS.md                 # Multi-session plans
│   ├── PROGRESS.md              # Progress tracking
│   ├── SESSION_HANDOFF.md       # Session continuity
│   ├── SUMMARIES.md             # Session summaries
│   ├── TESTS.md                 # Test documentation
│   └── TODOS.md                 # Task lists
│
├── src/jenova/                  # Main source code
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # SOLE application entry point (BubbleTea)
│   ├── llm_interface.py         # LLM integration
│   ├── tools.py                 # Tool calling
│   ├── default_api.py           # API placeholder
│   │
│   ├── cognitive_engine/        # Core cognitive processing
│   │   ├── engine.py            # Main cognitive engine
│   │   ├── query_analyzer.py    # Query understanding
│   │   ├── memory_search.py     # Unified memory search
│   │   ├── context_scorer.py    # Relevance scoring
│   │   ├── context_organizer.py # Context organization
│   │   ├── rag_system.py        # RAG pipeline
│   │   ├── scheduler.py         # Background tasks
│   │   ├── integration_layer.py # Cortex-Memory integration
│   │   └── document_processor.py# (Deprecated)
│   │
│   ├── memory/                  # Memory systems
│   │   ├── episodic.py          # Episodic memory
│   │   ├── semantic.py          # Semantic memory
│   │   └── procedural.py        # Procedural memory
│   │
│   ├── cortex/                  # Knowledge graph
│   │   ├── cortex.py            # Core cortex
│   │   ├── clustering.py        # Thematic clustering
│   │   ├── graph_metrics.py     # Graph analysis
│   │   ├── graph_components.py  # Component analysis
│   │   └── proactive_engine.py  # Autonomous reasoning
│   │
│   ├── insights/                # Insight management
│   │   ├── manager.py           # Insight lifecycle
│   │   └── concerns.py          # Topic categorization
│   │
│   ├── assumptions/             # Assumption tracking
│   │   └── manager.py           # Assumption lifecycle
│   │
│   ├── ui/                      # User interface (BubbleTea only)
│   │   ├── bubbletea.py         # Go TUI bridge with IPC
│   │   └── logger.py            # UI logging
│   │
│   ├── utils/                   # Utilities
│   │   ├── cache.py             # Caching
│   │   ├── embedding.py         # Embedding function
│   │   ├── json_parser.py       # JSON extraction
│   │   ├── model_loader.py      # Model loading
│   │   ├── performance_monitor.py # Metrics
│   │   ├── pydantic_compat.py   # Compatibility
│   │   ├── telemetry_fix.py     # Telemetry disable
│   │   └── file_logger.py       # File logging
│   │
│   └── config/                  # Configuration
│       ├── __init__.py          # Config loader
│       ├── main_config.yaml     # Main configuration
│       └── persona.yaml         # Persona definition
│
├── tui/                         # Go Bubble Tea TUI
│   ├── main.go                  # TUI implementation
│   ├── go.mod                   # Go module
│   └── go.sum                   # Go dependencies
│
├── tests/                       # Test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_basic.py            # Basic tests
│   ├── test_cognitive_engine.py # Engine tests
│   ├── test_cortex.py           # Cortex tests
│   └── test_memory.py           # Memory tests
│
├── finetune/                    # Fine-tuning utilities
│   ├── README.md                # Fine-tuning docs
│   └── train.py                 # Training data generator
│
└── [Root Files]
    ├── pyproject.toml           # Project configuration
    ├── requirements.txt         # Dependencies (prompt-toolkit removed)
    ├── setup.py                 # Setup script
    ├── jenova                   # Executable entry point
    ├── README.md                # Main documentation
    └── ...
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **LLM** | llama-cpp-python | Local LLM inference |
| **Embeddings** | SentenceTransformers | Text embeddings |
| **Vector DB** | ChromaDB | Persistent vector storage |
| **TUI** | Bubble Tea (Go) | Modern terminal interface |
| **IPC Bridge** | Python subprocess | Python-Go communication |
| **Logging** | rich | Formatted console output |
| **Config** | YAML | Configuration files |
| **Testing** | pytest | Test framework |

---

## Key Design Decisions

1. **BubbleTea-Only UI**: Single UI implementation for maintainability (Phase A decision)
2. **Local-First**: All LLM inference runs locally using llama-cpp-python
3. **Multi-Memory**: Three-tier memory system (Episodic, Semantic, Procedural)
4. **Knowledge Graph**: Cortex maintains interconnected knowledge nodes
5. **IPC Architecture**: Go handles UI rendering, Python handles cognitive logic
6. **100% FOSS**: Fully open-source, no proprietary dependencies
7. **ChromaDB**: Vector database for semantic search capabilities
8. **Modular Design**: Clear separation of concerns across layers

---

## Security Considerations

- ✅ No hardcoded credentials
- ✅ User data stored in `~/.jenova-ai/users/<username>/`
- ✅ `.gitignore` excludes sensitive files
- ✅ No unsafe eval/exec patterns
- ✅ ChromaDB telemetry disabled by default

---

## Phase A Changes Summary

The following changes were made during Phase A (UI Consolidation):

### Removed Files
- `src/jenova/main_bubbletea.py` (merged into main.py)
- `src/jenova/ui/terminal.py` (Python UI removed)

### Modified Files
- `src/jenova/main.py` - Now sole entry point with BubbleTeaUI
- `src/jenova/ui/bubbletea.py` - Enhanced with full feature parity
- `src/jenova/ui/__init__.py` - Updated exports
- `jenova` - Simplified executable
- `requirements.txt` - Removed prompt-toolkit dependency

### Dependencies Removed
- `prompt-toolkit` (was only used by terminal.py)
