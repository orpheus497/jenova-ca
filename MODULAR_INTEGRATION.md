# JENOVA Cognitive Architecture - Modular Integration Guide

This guide explains how to use the JENOVA Cognitive Architecture as a pluggable component in your own AI applications.

## Overview

JENOVA can be used in two ways:

1. **As a Complete Application**: Run the full terminal-based AI assistant
2. **As a Modular Cognitive Layer**: Integrate the cognitive architecture into your own applications

This guide focuses on the second use case.

## Quick Start

### Installation

```bash
pip install jenova-ai
```

### Basic Usage

```python
from jenova import CognitiveArchitecture

# Create with default components (uses local LLM + ChromaDB)
arch = CognitiveArchitecture.create_default(
    user_data_path="~/.my_app/cognitive_data"
)

# Process a query through the cognitive cycle
response = arch.think("What did we discuss yesterday?", user="alice")
print(response)

# Store knowledge
arch.remember("User prefers Python over JavaScript", user="alice")

# Retrieve relevant context
context = arch.retrieve("programming preferences", user="alice")
```

## Core Concepts

### The Cognitive Cycle

JENOVA implements a cognitive cycle inspired by human cognition:

1. **RETRIEVE**: Search multi-layered memory for relevant context
2. **PLAN**: Generate an execution plan based on context
3. **EXECUTE**: Generate a response using RAG (Retrieval-Augmented Generation)
4. **REFLECT**: Extract insights and update the knowledge graph

### Memory Layers

JENOVA uses four types of memory:

| Memory Type | Purpose | Example |
|------------|---------|---------|
| **Episodic** | Conversations and events | "Yesterday we discussed the project deadline" |
| **Semantic** | Facts and knowledge | "Python was created by Guido van Rossum" |
| **Procedural** | How-to and procedures | "To deploy: 1. Run tests 2. Build 3. Deploy" |
| **Insight** | Learned insights | "User prefers concise explanations" |

### Knowledge Graph (Cortex)

The Cortex maintains a graph of interconnected cognitive nodes:
- Insights and their relationships
- Assumptions about users
- Document knowledge

It supports deep reflection to:
- Link orphan nodes
- Generate meta-insights
- Prune outdated knowledge

## Custom Integrations

### Using a Custom LLM

```python
from jenova import CognitiveArchitecture, LLMAdapter
from typing import List, Optional

class OpenAIAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens or 512,
        )
        return response.choices[0].message.content
    
    def generate_with_context(
        self,
        prompt: str,
        context: List[str],
        **kwargs
    ) -> str:
        context_str = "\n".join(f"- {c}" for c in context)
        full_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}"
        return self.generate(full_prompt, **kwargs)

# Use the custom adapter
arch = CognitiveArchitecture(
    llm=OpenAIAdapter(api_key="sk-..."),
)
```

### Using a Custom Memory Backend

```python
from jenova import CognitiveArchitecture, MemoryBackend, MemoryEntry, MemoryType, SearchResult
from typing import List, Optional

class PineconeBackend(MemoryBackend):
    def __init__(self, api_key: str, index_name: str):
        import pinecone
        pinecone.init(api_key=api_key)
        self.index = pinecone.Index(index_name)
    
    def store(self, entry: MemoryEntry) -> str:
        # Your implementation
        pass
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[MemoryType] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        # Your implementation
        pass
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        # Your implementation
        pass
    
    def delete(self, entry_id: str) -> bool:
        # Your implementation
        pass
    
    def count(self, **kwargs) -> int:
        # Your implementation
        pass

# Use the custom backend
arch = CognitiveArchitecture(
    memory_backend=PineconeBackend(api_key="...", index_name="memories"),
)
```

### Using Custom Embeddings

```python
from jenova import CognitiveArchitecture, EmbeddingProvider
from typing import List, Union

class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, api_key: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [e.embedding for e in response.data]
    
    @property
    def dimension(self) -> int:
        return 1536  # text-embedding-3-small dimension
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed(query)[0]

# Use with the architecture
from jenova.core.adapters import create_default_memory

embedding = OpenAIEmbeddings(api_key="sk-...")
memory = create_default_memory(embedding_provider=embedding)

arch = CognitiveArchitecture(
    llm=my_llm,
    memory_backend=memory,
    embedding_provider=embedding,
)
```

## API Reference

### CognitiveArchitecture

The main entry point for the cognitive framework.

#### Methods

| Method | Description |
|--------|-------------|
| `think(query, user, context=None)` | Process a query through the cognitive cycle |
| `retrieve(query, user, n_results=10)` | Retrieve relevant context from memory |
| `remember(content, user, memory_type)` | Store information in memory |
| `reflect(user)` | Trigger deep reflection on the knowledge graph |
| `learn_insight(insight, topic, user)` | Manually store an insight |
| `get_history(user)` | Get conversation history for a user |
| `clear_history(user)` | Clear conversation history |
| `health_check()` | Check health of all components |

#### Configuration

```python
from jenova import CognitiveArchitecture, CognitiveConfig

config = CognitiveConfig(
    memory_cache_size=100,        # Items to cache
    insight_interval=5,           # Generate insights every N turns
    assumption_interval=7,        # Generate assumptions every N turns
    reflection_interval=20,       # Reflect on graph every N turns
    max_context_items=10,         # Max context for RAG
    rerank_enabled=True,          # Enable LLM re-ranking
    llm_timeout=120,              # LLM timeout in seconds
)

arch = CognitiveArchitecture(
    llm=my_llm,
    config=config,
)
```

### Interfaces

JENOVA provides these interfaces for custom implementations:

| Interface | Purpose |
|-----------|---------|
| `LLMAdapter` | Wrap any LLM |
| `EmbeddingProvider` | Use any embedding model |
| `MemoryBackend` | Use any vector store |
| `KnowledgeGraph` | Custom knowledge graph |
| `ReasoningEngine` | Custom cognitive cycle |
| `InsightGenerator` | Custom insight generation |
| `Logger` | Integrate with your logging |

## Examples

### Chatbot with Memory

```python
from jenova import CognitiveArchitecture

arch = CognitiveArchitecture.create_default()

def chat(message: str, user: str) -> str:
    """Process a chat message with cognitive context."""
    return arch.think(message, user=user)

# Example conversation
user = "alice"
print(chat("Hi, I'm learning Python!", user))
print(chat("What's a good project to start with?", user))
print(chat("What was I learning again?", user))  # Uses episodic memory
```

### Knowledge Base Agent

```python
from jenova import CognitiveArchitecture, MemoryType

arch = CognitiveArchitecture.create_default()

# Build knowledge base
arch.remember(
    "Python 3.12 introduces improved error messages and f-string syntax.",
    user="system",
    memory_type=MemoryType.SEMANTIC,
)
arch.remember(
    "To install Python packages: pip install <package>",
    user="system",
    memory_type=MemoryType.PROCEDURAL,
)

# Query the knowledge base
response = arch.think("How do I install packages in Python?", user="system")
```

### Multi-User Application

```python
from jenova import CognitiveArchitecture

arch = CognitiveArchitecture.create_default()

# Each user has isolated memory and history
users = ["alice", "bob", "charlie"]

for user in users:
    # User-specific interactions
    arch.think(f"Hi, I'm {user}", user=user)
    
    # User-specific knowledge
    arch.remember(f"{user}'s favorite color is blue", user=user)

# Retrieve is user-scoped
alice_context = arch.retrieve("favorite color", user="alice")
bob_context = arch.retrieve("favorite color", user="bob")
```

## Best Practices

1. **Initialize Once**: Create a single `CognitiveArchitecture` instance and reuse it
2. **Use User IDs**: Always provide a consistent user identifier for personalization
3. **Store Important Facts**: Use `remember()` to store key information
4. **Trigger Reflection**: Periodically call `reflect()` to consolidate knowledge
5. **Monitor Health**: Use `health_check()` in production environments

## Migration from Previous Versions

If you're upgrading from JENOVA 6.x:

```python
# Old way (direct component access)
from jenova.cognitive_engine import CognitiveEngine

# New way (through CognitiveArchitecture)
from jenova import CognitiveArchitecture
arch = CognitiveArchitecture.create_default()

# The full application still works the same way
from jenova.main import main
main()
```

The modular API is additive - all existing functionality continues to work.

## Further Reading

- [Architecture Overview](README.md#2-architecture-overview-core-design-principles)
- [Memory Systems](README.md#37-multi-layered-long-term-memory)
- [Cognitive Graph](README.md#33-the-cortex-a-graph-based-cognitive-core)
- [RAG System](README.md#32-the-rag-system-a-core-component-of-the-psyche)
