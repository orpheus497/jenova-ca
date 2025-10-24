# Fine-Tuning Framework - DEPRECATED

**This directory is deprecated as of v3.2.0**

The automatic fine-tuning system has been removed as part of the migration to GGUF models with llama-cpp-python. GGUF models are pre-quantized and optimized for inference, making traditional fine-tuning workflows incompatible.

## Why was it removed?

1. **GGUF Format**: GGUF models are quantized and optimized for inference. They cannot be fine-tuned directly using traditional methods like LoRA/PEFT.

2. **Simplified Architecture**: The new architecture focuses on RAG (Retrieval-Augmented Generation) and semantic memory for personalization instead of model fine-tuning.

3. **Local-First Approach**: Fine-tuning requires significant computational resources and complexity. The RAG approach provides personalization without modifying model weights.

## Alternative Approaches for Personalization

The JENOVA Cognitive Architecture now uses:

1. **Semantic Memory**: Your interactions are stored in a vector database and retrieved contextually during conversations.

2. **RAG System**: Documents and knowledge are retrieved and injected into prompts for context-aware responses.

3. **Cognitive Graph**: Insights, memories, and assumptions are organized in an interconnected graph for deep personalization.

## For Advanced Users

If you still want to fine-tune models:

1. **Use Native Format**: Fine-tune using the model's native format (e.g., safetensors, PyTorch) before converting to GGUF.

2. **External Tools**: Use tools like `llama.cpp`'s training features or Axolotl for GGUF-compatible training pipelines.

3. **Convert After Training**: Train → Quantize → Convert to GGUF for deployment.

---

The files in this directory are retained for reference but are no longer functional with the current system.
