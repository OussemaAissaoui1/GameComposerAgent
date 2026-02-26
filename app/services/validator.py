"""
Game Maker Agent v1 — Validation Layer
Enforces all game integrity rules before output.
Validates per-chapter output (5 questions each).
Stateless. Raises on any violation.
"""

from __future__ import annotations

import logging
from collections import Counter

from app.config import GameSettings
from app.models.schemas import LLMGenerationOutput, LLMQuestion, Difficulty

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when generated game data fails validation."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            f"Validation failed with {len(violations)} violation(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def validate_chapter_generation(
    output: LLMGenerationOutput,
    settings: GameSettings,
    num_chunks: int,
    chapter_id: str,
) -> LLMGenerationOutput:
    """Validate a single chapter's LLM generation output (5 questions).

    Checks:
    1. Exactly 5 questions.
    2. Exactly 4 options per question.
    3. Exactly 1 correct option per question.
    4. Difficulty distribution: 2 medium + 3 hard.
    5. No duplicate question semantics.
    6. No duplicate options within a question.
    7. All source_chunk_index values are valid.
    8. Question numbering is sequential 1–5.
    9. Option IDs must be A, B, C, D.
    10. Difficulty ratings in correct range.

    Args:
        output: Parsed LLM output for one chapter.
        settings: Game configuration.
        num_chunks: Number of source chunks available.
        chapter_id: Chapter identifier (for error messages).

    Returns:
        The validated output (passthrough if valid).

    Raises:
        ValidationError: If any rule is violated.
    """
    violations: list[str] = []
    questions = output.questions
    prefix = f"Ch{chapter_id}"

    # ── Rule 1: Exactly 5 questions ──────────────────────────────────────
    if len(questions) != settings.questions_per_chapter:
        violations.append(
            f"{prefix}: Expected {settings.questions_per_chapter} questions, "
            f"got {len(questions)}"
        )

    # ── Rule 2 & 3: Options per question ─────────────────────────────────
    for q in questions:
        if len(q.options) != settings.options_per_question:
            violations.append(
                f"{prefix}/Q{q.question_number}: Expected "
                f"{settings.options_per_question} options, got {len(q.options)}"
            )

        correct_count = sum(1 for o in q.options if o.is_correct)
        if correct_count != 1:
            violations.append(
                f"{prefix}/Q{q.question_number}: Expected 1 correct option, "
                f"got {correct_count}"
            )

    # ── Rule 4: Difficulty distribution ──────────────────────────────────
    difficulty_counts = Counter(q.difficulty for q in questions)
    expected = settings.difficulty_distribution
    for diff_name, expected_count in expected.items():
        actual = difficulty_counts.get(Difficulty(diff_name), 0)
        if actual != expected_count:
            violations.append(
                f"{prefix}: Difficulty '{diff_name}': "
                f"expected {expected_count}, got {actual}"
            )

    # ── Rule 5: No duplicate questions ───────────────────────────────────
    normalized_questions = [_normalize_text(q.question) for q in questions]
    seen_questions: set[str] = set()
    for i, nq in enumerate(normalized_questions):
        if nq in seen_questions:
            violations.append(
                f"{prefix}/Q{questions[i].question_number}: Duplicate question"
            )
        seen_questions.add(nq)

    # ── Rule 6: No duplicate options within a question ───────────────────
    for q in questions:
        option_texts = [_normalize_text(o.text) for o in q.options]
        if len(option_texts) != len(set(option_texts)):
            violations.append(
                f"{prefix}/Q{q.question_number}: Duplicate options detected"
            )

    # ── Rule 7: Valid source_chunk_index ──────────────────────────────────
    for q in questions:
        if q.source_chunk_index < 0 or q.source_chunk_index >= num_chunks:
            violations.append(
                f"{prefix}/Q{q.question_number}: source_chunk_index "
                f"{q.source_chunk_index} out of range [0, {num_chunks - 1}]"
            )

    # ── Rule 8: Sequential numbering 1–5 ────────────────────────────────
    expected_numbers = list(range(1, settings.questions_per_chapter + 1))
    actual_numbers = sorted(q.question_number for q in questions)
    if actual_numbers != expected_numbers:
        violations.append(
            f"{prefix}: Question numbering mismatch: "
            f"expected {expected_numbers}, got {actual_numbers}"
        )

    # ── Rule 9: Option IDs must be A, B, C, D ───────────────────────────
    for q in questions:
        option_ids = sorted(o.option_id for o in q.options)
        if option_ids != ["A", "B", "C", "D"]:
            violations.append(
                f"{prefix}/Q{q.question_number}: Option IDs must be A,B,C,D — "
                f"got {option_ids}"
            )

    # ── Rule 10: Difficulty ratings in range ─────────────────────────────
    for q in questions:
        _validate_difficulty_rating(q, violations, prefix)

    # ── Report ───────────────────────────────────────────────────────────
    if violations:
        logger.error("Validation failed for chapter %s: %s", chapter_id, violations)
        raise ValidationError(violations)

    logger.info("Validation passed for chapter %s: all rules satisfied", chapter_id)
    return output


def _normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace for comparison."""
    return " ".join(text.lower().split())


def _validate_difficulty_rating(
    q: LLMQuestion,
    violations: list[str],
    prefix: str = "",
) -> None:
    """Check difficulty_rating aligns with difficulty tier."""
    ranges = {
        Difficulty.MEDIUM: (450, 650),
        Difficulty.HARD: (651, 900),
    }
    low, high = ranges.get(q.difficulty, (400, 1000))
    if not (low <= q.difficulty_rating <= high):
        violations.append(
            f"{prefix}/Q{q.question_number}: difficulty_rating "
            f"{q.difficulty_rating} out of range [{low}, {high}] "
            f"for '{q.difficulty.value}'"
        )
