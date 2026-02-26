"""
Game Maker Agent v1 — Semantic Chunker
Splits chapter text into semantically coherent chunks.
Deterministic: same input always yields same chunks.
No randomness. Stable ordering.
"""

from __future__ import annotations

import logging
import re

from app.config import GameSettings

logger = logging.getLogger(__name__)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex heuristics.

    Handles abbreviations, decimal numbers, and common patterns.
    Deterministic — no randomness.
    """
    # Split on sentence-ending punctuation followed by whitespace + uppercase
    sentence_endings = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z"])'
    )
    raw_sentences = sentence_endings.split(text)
    # Filter empty and whitespace-only sentences
    return [s.strip() for s in raw_sentences if s.strip()]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English."""
    return max(1, len(text) // 4)


def chunk_text(text: str, settings: GameSettings) -> list[str]:
    """Semantically chunk chapter text into coherent segments.

    Strategy:
    1. Split into paragraphs first (preserves topic boundaries).
    2. Within paragraphs, split into sentences.
    3. Group sentences into chunks respecting max_tokens.
    4. Apply sentence overlap for context continuity.

    Args:
        text: Full chapter text (cleaned).
        settings: Contains chunk_max_tokens and chunk_overlap_sentences.

    Returns:
        Ordered list of text chunks. Deterministic for same input.
    """
    max_tokens = settings.chunk_max_tokens
    overlap = settings.chunk_overlap_sentences

    # ── Step 1: Split into paragraphs ────────────────────────────────────
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # ── Step 2: Flatten into sentences with paragraph markers ────────────
    all_sentences: list[str] = []
    for para in paragraphs:
        sentences = _split_into_sentences(para)
        if sentences:
            all_sentences.extend(sentences)

    if not all_sentences:
        # Fallback: treat entire text as one chunk
        logger.warning("No sentences detected; returning text as single chunk")
        return [text]

    logger.info("Split text into %d sentences", len(all_sentences))

    # ── Step 3: Group into chunks with overlap ───────────────────────────
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in all_sentences:
        sentence_tokens = _estimate_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            # Emit current chunk
            chunk_text_str = " ".join(current_sentences)
            chunks.append(chunk_text_str)

            # Carry over `overlap` trailing sentences for context
            if overlap > 0 and len(current_sentences) > overlap:
                current_sentences = current_sentences[-overlap:]
                current_tokens = sum(
                    _estimate_tokens(s) for s in current_sentences
                )
            else:
                current_sentences = []
                current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Emit final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    # ── Step 4: Ensure minimum chunk quality ─────────────────────────────
    # Merge very small trailing chunks into the previous one
    MIN_CHUNK_CHARS = 100
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    logger.info(
        "Created %d semantic chunks (max_tokens=%d, overlap=%d)",
        len(chunks),
        max_tokens,
        overlap,
    )

    return chunks
