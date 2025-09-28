# Jenova AI: Retrieval-Augmented Generation (RAG) Architecture

This document details the RAG implementation within the Jenova Cognitive Architecture. It is loaded into Semantic Memory on startup, making the AI self-aware of its own design.

## Overview

Jenova's RAG is a hybrid system designed to ground the Language Model's (LLM) responses in a rich, multi-faceted context. It achieves this by retrieving information from three distinct memory sources before generating a response. This prevents hallucination, improves factual accuracy, and allows the AI to learn and evolve.

## The Hybrid Retrieval Approach

The `MemorySearch` component performs a parallel query across three specialized memory systems:

1.  **Semantic Memory:** Stores hard, verifiable facts and ingested documents. On application startup, all facts and documents are vectorized and loaded into an in-memory ChromaDB collection, allowing for semantic similarity search.

2.  **Episodic Memory:** Stores summaries of past conversations and events, providing conversational context and a sense of history.

3.  **Procedural Memory:** Stores "how-to" guides and task execution steps, providing knowledge about processes and skills.

## The Re-Ranking Process

After retrieving results from all three databases, the `MemorySearch` component performs a re-ranking step. It uses the distance scores (a measure of semantic similarity) from the vector searches to sort the combined list of documents. This ensures that the most relevant pieces of information, regardless of their source, are prioritized and passed to the reasoning engine.