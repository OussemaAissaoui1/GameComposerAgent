"""
Game Maker Agent v1 — Formatter Service
Separates public puzzles from private answers.
Generates blockchain-compatible anchor strings.
Multi-chapter: merges 4 chapters × 5 questions = 20 total.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import AppSettings
from app.models.schemas import (
    GameMeta,
    GamePayload,
    LLMGenerationOutput,
    LLMQuestion,
    Option,
    PrivateAnswerKey,
    PublicPuzzle,
)

logger = logging.getLogger(__name__)


@dataclass
class ChapterResult:
    """Container for one chapter's validated LLM output."""

    chapter_id: str
    chapter_title: str
    output: LLMGenerationOutput


def _make_puzzle_id(chapter_id: str, question_number: int) -> str:
    """Deterministic puzzle ID: ch{chapter_id}_q{nn}."""
    return f"ch{chapter_id}_q{question_number:02d}"


def _make_anchor_string(puzzle_id: str, correct_option_id: str) -> str:
    """Blockchain-compatible anchor: '<puzzle_id>|<correct_option_id>'.

    No hashing is performed here — only the anchor string is generated.
    Downstream systems (on-chain contracts) handle hashing.
    """
    return f"{puzzle_id}|{correct_option_id}"


def _get_correct_option_id(question: LLMQuestion) -> str:
    """Extract the correct option ID from a question."""
    for option in question.options:
        if option.is_correct:
            return option.option_id
    raise ValueError(
        f"Q{question.question_number}: No correct option found "
        "(should have been caught by validator)"
    )


def format_game_output(
    chapter_results: list[ChapterResult],
    difficulty_target: int,
    settings: AppSettings,
) -> GamePayload:
    """Transform validated multi-chapter LLM outputs into a single game payload.

    Separation guarantee:
    - public_puzzles: NO correct answer indication
    - private_answer_key: Contains correct_option_id + anchor

    Args:
        chapter_results: List of ChapterResult (one per chapter, 4 total).
        difficulty_target: Original difficulty target from request.
        settings: Application settings.

    Returns:
        Complete GamePayload with 20 public puzzles and 20 private keys.
    """
    public_puzzles: list[PublicPuzzle] = []
    private_keys: list[PrivateAnswerKey] = []
    chapter_titles: dict[str, str] = {}

    for cr in chapter_results:
        chapter_titles[cr.chapter_id] = cr.chapter_title

        sorted_questions = sorted(
            cr.output.questions,
            key=lambda q: q.question_number,
        )

        for question in sorted_questions:
            puzzle_id = _make_puzzle_id(cr.chapter_id, question.question_number)
            correct_id = _get_correct_option_id(question)
            anchor = _make_anchor_string(puzzle_id, correct_id)

            # ── Public puzzle: strip correctness info ────────────────────
            public_options = [
                Option(option_id=opt.option_id, text=opt.text)
                for opt in sorted(question.options, key=lambda o: o.option_id)
            ]

            public_puzzles.append(
                PublicPuzzle(
                    puzzle_id=puzzle_id,
                    chapter_id=cr.chapter_id,
                    chapter_title=cr.chapter_title,
                    question=question.question,
                    options=public_options,
                    difficulty=question.difficulty,
                    difficulty_rating=question.difficulty_rating,
                    min_solve_time_seconds=question.min_solve_time_seconds,
                    source_chunk_index=question.source_chunk_index,
                )
            )

            # ── Private answer key ───────────────────────────────────────
            private_keys.append(
                PrivateAnswerKey(
                    puzzle_id=puzzle_id,
                    chapter_id=cr.chapter_id,
                    correct_option_id=correct_id,
                    anchor_string=anchor,
                    explanation=question.explanation,
                )
            )

    # ── Meta ─────────────────────────────────────────────────────────────
    meta = GameMeta(
        chapters=list(chapter_titles.keys()),
        chapter_titles=chapter_titles,
        questions_per_chapter=settings.game.questions_per_chapter,
        total_questions=len(public_puzzles),
        difficulty_target=difficulty_target,
        difficulty_distribution_per_chapter=settings.game.difficulty_distribution,
        model_used=settings.llm.model_name,
        temperature=settings.llm.temperature,
        version=settings.version,
    )

    payload = GamePayload(
        meta=meta,
        public_puzzles=public_puzzles,
        private_answer_key=private_keys,
    )

    logger.info(
        "Formatted game output: %d chapters, %d public puzzles, "
        "%d private keys, %d anchor strings",
        len(chapter_results),
        len(public_puzzles),
        len(private_keys),
        len(private_keys),
    )

    for pk in private_keys:
        logger.debug("Anchor: %s", pk.anchor_string)

    return payload
