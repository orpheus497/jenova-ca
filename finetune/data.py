##Script function and purpose: Data utilities for fine-tuning training data preparation
"""
Training Data Utilities

Utilities for preparing and managing training data for
fine-tuning the JENOVA embedding model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


##Class purpose: Training example for contrastive learning
@dataclass
class TrainingExample:
    """A training example for contrastive learning."""

    anchor: str
    """The anchor text (e.g., query)."""

    positive: str
    """A positive match for the anchor."""

    negative: str | None = None
    """Optional hard negative example."""

    ##Method purpose: Convert to dict for JSON serialization
    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        result = {"anchor": self.anchor, "positive": self.positive}
        ##Condition purpose: Include negative if present
        if self.negative:
            result["negative"] = self.negative
        return result


##Class purpose: Manager for training data collection
class TrainingDataCollector:
    """
    Collects training data from JENOVA interactions.

    Gathers positive examples from successful retrievals
    and user feedback to improve embeddings.
    """

    ##Method purpose: Initialize with output path
    def __init__(self, output_path: Path) -> None:
        """
        Initialize collector.

        Args:
            output_path: Path to JSONL output file
        """
        self.output_path = output_path
        self._examples: list[TrainingExample] = []

    ##Method purpose: Add a training example
    def add_example(
        self,
        anchor: str,
        positive: str,
        negative: str | None = None,
    ) -> None:
        """
        Add a training example.

        Args:
            anchor: Anchor text
            positive: Positive match
            negative: Optional hard negative
        """
        self._examples.append(
            TrainingExample(
                anchor=anchor,
                positive=positive,
                negative=negative,
            )
        )

    ##Method purpose: Save collected examples to file
    def save(self) -> int:
        """
        Save examples to JSONL file.

        Returns:
            Number of examples saved
        """
        ##Step purpose: Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        ##Step purpose: Write examples
        with open(self.output_path, "a", encoding="utf-8") as f:
            ##Loop purpose: Write each example as JSON line
            for example in self._examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        saved_count = len(self._examples)
        self._examples.clear()
        return saved_count

    ##Method purpose: Get pending example count
    @property
    def pending_count(self) -> int:
        """Get number of unsaved examples."""
        return len(self._examples)


##Function purpose: Create training examples from successful retrieval
def create_retrieval_examples(
    query: str,
    retrieved_contents: list[str],
    max_positives: int = 3,
) -> list[TrainingExample]:
    """
    Create training examples from a successful retrieval.

    Args:
        query: The user query
        retrieved_contents: Contents that were retrieved and used
        max_positives: Maximum positive examples to create

    Returns:
        List of training examples
    """
    examples: list[TrainingExample] = []

    ##Loop purpose: Create example for each retrieved content
    for content in retrieved_contents[:max_positives]:
        examples.append(
            TrainingExample(
                anchor=query,
                positive=content,
            )
        )

    return examples
