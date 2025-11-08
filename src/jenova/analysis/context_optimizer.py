# The JENOVA Cognitive Architecture - Context Optimizer
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Context window optimization for efficient LLM usage.

Optimizes context window usage through intelligent token counting,
relevance scoring, and semantic chunking.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ContextSegment:
    """Represents a segment of context."""
    content: str
    tokens: int
    relevance_score: float
    source: str  # 'memory', 'conversation', 'document', etc.
    metadata: Dict = None


class ContextOptimizer:
    """
    Optimize context window usage for LLM calls.

    Features:
    - Token counting with configurable models
    - Relevance-based prioritization
    - Sliding window optimization
    - Semantic chunking for long documents
    - Cache-aware optimization
    """

    def __init__(self, max_tokens: int = 4096, reserve_tokens: int = 512):
        """
        Initialize context optimizer.

        Args:
            max_tokens: Maximum context window size
            reserve_tokens: Tokens to reserve for response generation
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple heuristic: ~4 characters per token for English.
        For production, integrate with sentence-transformers tokenizer.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Simple estimation: 4 chars per token (conservative)
        # Add extra for special tokens and formatting
        base_tokens = len(text) // 4

        # Count newlines and add overhead
        newlines = text.count('\n')

        return base_tokens + (newlines * 2) + 10

    def chunk_text(self, text: str, max_chunk_tokens: int = 512) -> List[str]:
        """
        Chunk text into semantic segments.

        Args:
            text: Input text
            max_chunk_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Split on double newlines (paragraph breaks) first
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.estimate_tokens(para)

            # If single paragraph exceeds limit, split on sentences
            if para_tokens > max_chunk_tokens:
                # Split on sentence boundaries
                sentences = re.split(r'([.!?]\s+)', para)

                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]  # Include punctuation

                    sent_tokens = self.estimate_tokens(sentence)

                    if current_tokens + sent_tokens > max_chunk_tokens:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens
            else:
                # Add paragraph to current chunk
                if current_tokens + para_tokens > max_chunk_tokens:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def score_relevance(self, segment: str, query: str) -> float:
        """
        Score relevance of segment to query.

        Uses simple keyword overlap and TF-IDF-like scoring.

        Args:
            segment: Text segment to score
            query: Query text

        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not segment or not query:
            return 0.0

        # Normalize text
        segment_lower = segment.lower()
        query_lower = query.lower()

        # Extract keywords (simple: split on whitespace and punctuation)
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        segment_words = re.findall(r'\b\w+\b', segment_lower)

        if not query_words or not segment_words:
            return 0.0

        # Calculate keyword overlap
        matches = sum(1 for word in segment_words if word in query_words)

        # Base score from overlap ratio
        overlap_score = matches / len(segment_words)

        # Boost score if query words appear in order
        query_text = ' '.join(query_words)
        if query_text in segment_lower:
            overlap_score *= 1.5

        # Boost for exact phrase matches
        query_phrases = re.findall(r'"([^"]+)"', query)
        for phrase in query_phrases:
            if phrase.lower() in segment_lower:
                overlap_score *= 1.3

        # Cap at 1.0
        return min(overlap_score, 1.0)

    def optimize_context(
        self,
        segments: List[ContextSegment],
        query: str,
        preserve_order: bool = False
    ) -> List[ContextSegment]:
        """
        Optimize context by selecting most relevant segments within token budget.

        Args:
            segments: List of context segments
            query: Current query
            preserve_order: Whether to preserve original order

        Returns:
            Optimized list of segments
        """
        if not segments:
            return []

        # Score each segment for relevance if not already scored
        for segment in segments:
            if segment.relevance_score == 0.0:
                segment.relevance_score = self.score_relevance(
                    segment.content, query
                )

        # Sort by relevance (descending) if not preserving order
        if not preserve_order:
            sorted_segments = sorted(
                segments,
                key=lambda s: s.relevance_score,
                reverse=True
            )
        else:
            sorted_segments = segments

        # Select segments within token budget
        selected = []
        total_tokens = 0

        for segment in sorted_segments:
            if total_tokens + segment.tokens <= self.available_tokens:
                selected.append(segment)
                total_tokens += segment.tokens
            else:
                # Check if we can fit a truncated version
                remaining_tokens = self.available_tokens - total_tokens
                if remaining_tokens > 50:  # Only if meaningful space left
                    # Truncate segment to fit
                    truncated_content = self.truncate_to_tokens(
                        segment.content,
                        remaining_tokens
                    )
                    if truncated_content:
                        truncated_segment = ContextSegment(
                            content=truncated_content,
                            tokens=self.estimate_tokens(truncated_content),
                            relevance_score=segment.relevance_score,
                            source=segment.source,
                            metadata=segment.metadata
                        )
                        selected.append(truncated_segment)
                break

        # Restore original order if requested
        if preserve_order and selected:
            # Create index map
            index_map = {id(seg): i for i, seg in enumerate(segments)}
            selected.sort(key=lambda s: index_map.get(id(s), float('inf')))

        return selected

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token budget.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        if not text:
            return ""

        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Estimate characters to keep
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.9)  # 90% to be safe

        # Try to truncate at sentence boundary
        truncated = text[:target_chars]

        # Find last sentence boundary
        last_period = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? ')
        )

        if last_period > target_chars * 0.5:  # If found in latter half
            truncated = truncated[:last_period + 1]

        # Add ellipsis if truncated
        if len(truncated) < len(text):
            truncated += "..."

        return truncated

    def create_segments(
        self,
        content_dict: Dict[str, str],
        query: str = ""
    ) -> List[ContextSegment]:
        """
        Create context segments from content dictionary.

        Args:
            content_dict: Dict with source names as keys and content as values
            query: Optional query for relevance scoring

        Returns:
            List of context segments
        """
        segments = []

        for source, content in content_dict.items():
            if not content:
                continue

            # Chunk large content
            if self.estimate_tokens(content) > 512:
                chunks = self.chunk_text(content, max_chunk_tokens=512)
                for i, chunk in enumerate(chunks):
                    tokens = self.estimate_tokens(chunk)
                    relevance = self.score_relevance(chunk, query) if query else 0.0

                    segment = ContextSegment(
                        content=chunk,
                        tokens=tokens,
                        relevance_score=relevance,
                        source=f"{source}_chunk_{i}",
                        metadata={"chunk_index": i, "total_chunks": len(chunks)}
                    )
                    segments.append(segment)
            else:
                # Single segment
                tokens = self.estimate_tokens(content)
                relevance = self.score_relevance(content, query) if query else 0.0

                segment = ContextSegment(
                    content=content,
                    tokens=tokens,
                    relevance_score=relevance,
                    source=source,
                    metadata={}
                )
                segments.append(segment)

        return segments

    def get_optimization_stats(self, segments: List[ContextSegment]) -> Dict:
        """
        Get statistics about context optimization.

        Args:
            segments: List of segments

        Returns:
            Statistics dictionary
        """
        if not segments:
            return {
                "total_segments": 0,
                "total_tokens": 0,
                "avg_relevance": 0.0,
                "utilization": 0.0
            }

        total_tokens = sum(s.tokens for s in segments)
        avg_relevance = sum(s.relevance_score for s in segments) / len(segments)
        utilization = total_tokens / self.available_tokens

        return {
            "total_segments": len(segments),
            "total_tokens": total_tokens,
            "avg_relevance": round(avg_relevance, 3),
            "utilization": round(utilization, 3),
            "sources": list(set(s.source for s in segments))
        }

    def optimize(self, context: str, max_tokens: int) -> str:
        """
        Legacy method for backward compatibility.

        Args:
            context: Context string
            max_tokens: Maximum tokens

        Returns:
            Optimized context string
        """
        segments = self.create_segments({"context": context})
        self.available_tokens = max_tokens
        optimized_segments = self.optimize_context(segments, "", preserve_order=True)

        return ' '.join(s.content for s in optimized_segments)
