# JENOVA - Fine-Tuning Data Generation

This directory contains tools to create comprehensive fine-tuning datasets from JENOVA's complete cognitive architecture. The system extracts training data from all cognitive sources to create a rich dataset that can be used for model fine-tuning.

## Overview

The `train.py` script generates a comprehensive `.jsonl` training file from your JENOVA cognitive architecture, including:

- **Insights:** Learned conclusions organized by topics
- **Episodic Memory:** Conversation history and interactions
- **Semantic Memory:** Factual knowledge with sources
- **Procedural Memory:** How-to procedures and instructions
- **Assumptions:** Verified knowledge about user preferences
- **Documents:** Knowledge from processed documents in cognitive graph

## How It Works

1. **Data Collection:** As you interact with JENOVA, the cognitive architecture automatically stores data across multiple memory systems and generates insights
2. **Extraction:** The `train.py` script scans your user data directory (`~/.jenova-ai/users/<username>/`) and extracts training examples from all sources
3. **Formatting:** Data is formatted into conversational pairs (user query → assistant response) suitable for instruction fine-tuning
4. **Output:** A `.jsonl` file is generated with comprehensive training data

## Usage

### Generate Training Data

From the project root directory:

```bash
python finetune/train.py
```

This creates `finetune_train.jsonl` with training data from your complete cognitive architecture.

### Options

```bash
python finetune/train.py --user-data-dir ~/.jenova-ai/users/myname --output-file my_training.jsonl
```

Arguments:
- `--user-data-dir`: Path to user data directory (default: `~/.jenova-ai/users/<current_user>`)
- `--output-file`: Output filename (default: `finetune_train.jsonl`)

## Fine-Tuning with External Tools

The generated `.jsonl` file can be used with various fine-tuning approaches:

### Option 1: llama.cpp Training (GGUF Compatible)

llama.cpp provides tools for training and fine-tuning GGUF models directly:

```bash
# See llama.cpp documentation for training utilities
./finetune --model model.gguf --data finetune_train.jsonl
```

### Option 2: HuggingFace Transformers (Convert First)

For models in PyTorch/safetensors format:

1. Convert your GGUF model to native format using llama.cpp conversion tools
2. Fine-tune using Transformers/PEFT:
   ```bash
   # Install dependencies
   pip install transformers peft accelerate datasets
   
   # Use standard HuggingFace fine-tuning pipeline with finetune_train.jsonl
   ```
3. Convert fine-tuned model back to GGUF

### Option 3: Axolotl (Advanced)

Axolotl provides a comprehensive fine-tuning framework:

```bash
# Install Axolotl
pip install axolotl

# Configure and run fine-tuning with finetune_train.jsonl
```

## Data Format

The generated `.jsonl` file contains entries in this format:

```json
{
  "messages": [
    {"role": "user", "content": "User's question or prompt"},
    {"role": "assistant", "content": "JENOVA's response with learned knowledge"}
  ],
  "source": "insights|episodic|semantic|procedural|assumptions|documents",
  "metadata": {...}
}
```

Each line is a complete training example representing JENOVA's learned knowledge.

## Best Practices

1. **Accumulate Data:** Interact with JENOVA extensively before generating training data (aim for 50+ insights minimum)
2. **Review Output:** Check the generated `.jsonl` file to ensure quality
3. **Regular Updates:** Regenerate training data periodically as you continue interacting with JENOVA
4. **Backup:** Keep copies of training data files for different stages of JENOVA's development

## Integration with JENOVA

After fine-tuning your model:

1. Convert the fine-tuned model to GGUF format (if needed)
2. Place the `.gguf` file in `./models/`
3. Update `src/jenova/config/main_config.yaml` with the new model path:
   ```yaml
   model:
     model_path: './models/your-finetuned-model.gguf'
   ```
4. Run JENOVA with your personalized model

## Technical Notes

- **Memory Format:** ChromaDB data is extracted from SQLite databases in memory directories
- **Document Knowledge:** Extracts both full documents and chunks from cognitive graph
- **Source Attribution:** Each training entry includes metadata about its source for analysis
- **Privacy:** All data is local; no external services are used

## Troubleshooting

**No data generated:**
- Interact with JENOVA more to build cognitive data
- Check that `~/.jenova-ai/users/<username>/` exists and contains data

**ChromaDB errors:**
- Ensure memory databases are not corrupted
- Check that ChromaDB sqlite files exist in memory directories

**JSON errors:**
- Corrupted files are automatically skipped with warnings
- Check file logger for details on skipped entries