##Script function and purpose: Fine-tuning training script for JENOVA embedding model
"""
JENOVA Embedding Model Training

Training script for fine-tuning the JENOVA embedding model
on learned insights and interactions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


##Function purpose: Load training data from JSONL file
def load_training_data(path: Path) -> list[dict[str, str]]:
    """
    Load training data from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of training examples
    """
    import json

    examples: list[dict[str, str]] = []

    ##Loop purpose: Read each line as JSON
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            ##Condition purpose: Skip empty lines
            if line:
                examples.append(json.loads(line))

    return examples


##Function purpose: Train embedding model on examples
def train(
    model_name: str = "all-MiniLM-L6-v2",
    training_data_path: Path | None = None,
    output_path: Path | None = None,
    epochs: int = 3,
    batch_size: int = 16,
) -> Path:
    """
    Train the JENOVA embedding model.

    Args:
        model_name: Base model to fine-tune
        training_data_path: Path to training JSONL
        output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Path to saved model
    """
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    ##Step purpose: Set default paths
    if training_data_path is None:
        training_data_path = Path(".jenova-ai/training_data.jsonl")
    if output_path is None:
        output_path = Path(".jenova-ai/models/jenova-embedding")

    ##Step purpose: Load base model
    model = SentenceTransformer(model_name)

    ##Step purpose: Load and convert training data
    raw_examples = load_training_data(training_data_path)

    ##Step purpose: Convert to InputExamples (assuming contrastive format)
    train_examples: list[InputExample] = []
    ##Loop purpose: Convert each raw example
    for example in raw_examples:
        ##Condition purpose: Handle different formats
        if "anchor" in example and "positive" in example:
            train_examples.append(
                InputExample(
                    texts=[example["anchor"], example["positive"]],
                )
            )

    ##Step purpose: Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    ##Step purpose: Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)

    ##Step purpose: Train model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=str(output_path),
    )

    return output_path


##Condition purpose: Allow running as script
if __name__ == "__main__":
    import argparse

    ##Step purpose: Parse arguments
    parser = argparse.ArgumentParser(description="Train JENOVA embedding model")
    parser.add_argument("--data", type=Path, help="Training data JSONL")
    parser.add_argument("--output", type=Path, help="Output model path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--base-model", default="all-MiniLM-L6-v2", help="Base model")

    args = parser.parse_args()

    ##Action purpose: Run training
    output = train(
        model_name=args.base_model,
        training_data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"Model saved to: {output}")
