# The JENOVA Cognitive Architecture - Context Compression
# Copyright (c) 2024, orpheus497. All rights reserved.
#
# The JENOVA Cognitive Architecture is licensed under the MIT License.
# A copy of the license can be found in the LICENSE file in the root directory of this source tree.

"""
Phase 24: Context Compression.

Compresses low-relevance context using multiple strategies:
- Extract ive: Select key sentences using TF-IDF
- Abstractive: LLM-generated summaries
- Hybrid: Combine both approaches for balanced compression
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import math


class ContextCompressor:
    """
    Compresses context using extractive, abstractive, or hybrid strategies.

    Strategies:
        - Extractive: Select most important sentences using TF-IDF scoring
        - Abstractive: Generate LLM summary of content
        - Hybrid: Combine extractive (50%) and abstractive (50%)

    Example:
        >>> compressor = ContextCompressor(llm_interface)
        >>> compressed = compressor.compress_context(
        ...     long_text,
        ...     target_ratio=0.3,
        ...     strategy="hybrid"
        ... )
    """

    def __init__(self, llm_interface=None):
        """
        Initialize context compressor.

        Args:
            llm_interface: Optional LLM interface for abstractive compression
        """
        self.llm_interface = llm_interface

        # Stop words for TF-IDF (common words to ignore)
        self.stop_words = set(
            [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "under",
                "again",
                "further",
                "then",
                "once",
                "here",
                "there",
                "when",
                "where",
                "why",
                "how",
                "all",
                "both",
                "each",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
                "no",
                "nor",
                "not",
                "only",
                "own",
                "same",
                "so",
                "than",
                "too",
                "very",
                "can",
                "will",
                "just",
                "should",
                "now",
            ]
        )

    def compress_context(
        self, content: str, target_ratio: float = 0.3, strategy: str = "hybrid"
    ) -> str:
        """
        Compress context to target compression ratio.

        Args:
            content: Text content to compress
            target_ratio: Target compression ratio (0.0-1.0)
            strategy: Compression strategy (extractive, abstractive, hybrid)

        Returns:
            Compressed text

        Example:
            >>> compressed = compressor.compress_context(
            ...     "Long text here...",
            ...     target_ratio=0.3,
            ...     strategy="extractive"
            ... )
        """
        if not content or not content.strip():
            return ""

        if target_ratio >= 1.0:
            return content  # No compression needed

        if strategy == "extractive":
            return self._extractive_compression(content, target_ratio)
        elif strategy == "abstractive":
            return self._abstractive_compression(content, target_ratio)
        elif strategy == "hybrid":
            return self._hybrid_compression(content, target_ratio)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be 'extractive', 'abstractive', or 'hybrid'"
            )

    def _extractive_compression(self, content: str, ratio: float) -> str:
        """
        Extract most important sentences using TF-IDF.

        Args:
            content: Text to compress
            ratio: Target ratio (0.0-1.0)

        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = self._split_sentences(content)

        if not sentences:
            return ""

        # Calculate number of sentences to keep
        num_sentences = max(1, int(len(sentences) * ratio))

        # Calculate sentence importance scores
        sentence_scores = self._calculate_sentence_importance(sentences)

        # Select top sentences
        top_indices = sorted(
            range(len(sentence_scores)),
            key=lambda i: sentence_scores[i],
            reverse=True,
        )[:num_sentences]

        # Maintain original order
        top_indices.sort()

        # Assemble compressed text
        return " ".join(sentences[i] for i in top_indices)

    def _abstractive_compression(self, content: str, ratio: float) -> str:
        """
        Generate LLM summary of content.

        Args:
            content: Text to summarize
            ratio: Target ratio (determines max_tokens)

        Returns:
            Abstractive summary
        """
        if self.llm_interface is None:
            # Fallback to extractive if no LLM available
            return self._extractive_compression(content, ratio)

        # Calculate target token count
        original_tokens = len(content) // 4  # Approximation
        max_tokens = int(original_tokens * ratio)

        # Generate summary prompt
        prompt = f"""Summarize the following text concisely in approximately {max_tokens} tokens:

{content}

Summary:"""

        try:
            # Generate summary using LLM
            summary = self.llm_interface.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more focused summary
            )

            return summary.strip()

        except Exception as e:
            # Fallback to extractive on error
            return self._extractive_compression(content, ratio)

    def _hybrid_compression(self, content: str, ratio: float) -> str:
        """
        Combine extractive and abstractive compression.

        Strategy: Extractive (50% of target) + Abstractive (50% of target)

        Args:
            content: Text to compress
            ratio: Target ratio

        Returns:
            Hybrid compressed text
        """
        # Split ratio between extractive and abstractive
        extractive_ratio = ratio * 1.5  # Get more from extractive
        abstractive_ratio = ratio * 0.5  # Summarize more aggressively

        # Get extractive compression first
        extractive_result = self._extractive_compression(content, extractive_ratio)

        # Then apply abstractive compression
        if self.llm_interface is not None:
            return self._abstractive_compression(extractive_result, abstractive_ratio)
        else:
            # If no LLM, just return extractive
            return extractive_result

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (handles . ! ?)
        # More sophisticated: use nltk.sent_tokenize
        sentence_pattern = r"[.!?]+\s+"
        sentences = re.split(sentence_pattern, text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _calculate_sentence_importance(self, sentences: List[str]) -> List[float]:
        """
        Calculate importance score for each sentence using TF-IDF.

        Args:
            sentences: List of sentences

        Returns:
            List of importance scores (same length as sentences)
        """
        if not sentences:
            return []

        # Calculate TF-IDF scores
        # TF: term frequency within sentence
        # IDF: inverse document frequency across all sentences

        # 1. Calculate document frequency (DF)
        df: Dict[str, int] = Counter()
        for sentence in sentences:
            words = set(self._tokenize(sentence))
            for word in words:
                df[word] += 1

        # 2. Calculate IDF
        num_sentences = len(sentences)
        idf: Dict[str, float] = {}
        for word, freq in df.items():
            idf[word] = math.log(num_sentences / freq)

        # 3. Calculate TF-IDF score for each sentence
        scores: List[float] = []
        for sentence in sentences:
            words = self._tokenize(sentence)
            if not words:
                scores.append(0.0)
                continue

            # TF for this sentence
            tf: Dict[str, float] = Counter(words)
            total_words = len(words)
            for word in tf:
                tf[word] /= total_words

            # TF-IDF score = sum of (TF * IDF) for all words
            tfidf_score = sum(tf[word] * idf.get(word, 0.0) for word in tf)

            scores.append(tfidf_score)

        return scores

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words, removing stop words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (lowercase, no stop words)
        """
        # Convert to lowercase and split on non-alphanumeric
        words = re.findall(r"\b[a-z0-9]+\b", text.lower())

        # Remove stop words
        words = [w for w in words if w not in self.stop_words and len(w) > 2]

        return words

    def get_compression_stats(
        self, original: str, compressed: str
    ) -> Dict[str, Any]:
        """
        Calculate compression statistics.

        Args:
            original: Original text
            compressed: Compressed text

        Returns:
            Dict with stats (original_length, compressed_length, ratio, etc.)
        """
        original_len = len(original)
        compressed_len = len(compressed)

        original_tokens = len(original) // 4
        compressed_tokens = len(compressed) // 4

        return {
            "original_chars": original_len,
            "compressed_chars": compressed_len,
            "char_ratio": compressed_len / original_len if original_len > 0 else 0.0,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_ratio": compressed_tokens / original_tokens
            if original_tokens > 0
            else 0.0,
            "reduction_percent": (
                ((original_len - compressed_len) / original_len * 100)
                if original_len > 0
                else 0.0
            ),
        }
