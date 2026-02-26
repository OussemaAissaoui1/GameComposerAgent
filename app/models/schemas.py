"""
Game Maker Agent v1 — Pydantic Schemas
All data contracts: request, response, internal models.
4 chapters × 5 questions = 20 MCQs per game.
Each call generates fresh questions (non-deterministic).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class Difficulty(str, Enum):
    MEDIUM = "medium"
    HARD = "hard"


# ── Request ──────────────────────────────────────────────────────────────────

class GameRequest(BaseModel):
    """POST /generate-game request body.

    No pdf_path needed — chapters are mapped to PDFs in config.
    """

    difficulty_target: int = Field(
        default=700,
        ge=400,
        le=1000,
        description="Target difficulty score (higher = harder)",
    )


# ── Option ───────────────────────────────────────────────────────────────────

class Option(BaseModel):
    """A single MCQ option (public-safe: no correctness flag)."""

    option_id: str = Field(..., pattern=r"^[A-D]$")
    text: str = Field(..., min_length=1)


# ── Public Puzzle ────────────────────────────────────────────────────────────

class PublicPuzzle(BaseModel):
    """
    Publicly visible puzzle data.
    MUST NOT contain the correct answer.
    """

    puzzle_id: str = Field(..., description="e.g. 'ch1_q01'")
    chapter_id: str = Field(..., description="Which chapter this belongs to")
    chapter_title: str = Field(..., description="Human-readable chapter name")
    question: str = Field(..., min_length=10)
    options: list[Option] = Field(..., min_length=4, max_length=4)
    difficulty: Difficulty
    difficulty_rating: int = Field(..., ge=400, le=1000)
    min_solve_time_seconds: int = Field(..., ge=10, le=300)
    source_chunk_index: int = Field(
        ...,
        ge=0,
        description="Index of the chunk from which this question was derived",
    )


# ── Private Answer Key ──────────────────────────────────────────────────────

class PrivateAnswerKey(BaseModel):
    """Private answer metadata — never exposed publicly."""

    puzzle_id: str
    chapter_id: str
    correct_option_id: Literal["A", "B", "C", "D"]
    anchor_string: str = Field(
        ...,
        description="Blockchain anchor: '<puzzle_id>|<correct_option_id>'",
    )
    explanation: str = Field(
        ...,
        min_length=10,
        description="Why this answer is correct, grounded in source text",
    )


# ── Game Meta ────────────────────────────────────────────────────────────────

class GameMeta(BaseModel):
    """Metadata about the generated game."""

    total_questions: int = 20
    questions_per_chapter: int = 5
    chapters: list[str] = Field(default=["1", "2", "3", "4"])
    chapter_titles: dict[str, str] = Field(default_factory=dict)
    difficulty_target: int
    difficulty_distribution_per_chapter: dict[str, int] = Field(
        default={"medium": 2, "hard": 3},
    )
    model_used: str
    temperature: float
    version: str = "1.0.0"


# ── Full Game Response ───────────────────────────────────────────────────────

class GamePayload(BaseModel):
    """Complete game output — meta + public + private."""

    meta: GameMeta
    public_puzzles: list[PublicPuzzle] = Field(..., min_length=20, max_length=20)
    private_answer_key: list[PrivateAnswerKey] = Field(
        ..., min_length=20, max_length=20,
    )


class GameResponse(BaseModel):
    """API response wrapper."""

    status: Literal["success", "error"]
    game: GamePayload | None = None
    error: str | None = None


# ── LLM Raw Output Schema (for parsing LLM JSON response) ───────────────────

class LLMOption(BaseModel):
    """Option as returned by the LLM (includes is_correct for internal use)."""

    option_id: str = Field(..., pattern=r"^[A-D]$")
    text: str = Field(..., min_length=1)
    is_correct: bool


class LLMQuestion(BaseModel):
    """Single question as returned by the LLM."""

    question_number: int = Field(..., ge=1, le=5)
    question: str = Field(..., min_length=10)
    options: list[LLMOption] = Field(..., min_length=4, max_length=4)
    difficulty: Difficulty
    difficulty_rating: int = Field(..., ge=400, le=1000)
    min_solve_time_seconds: int = Field(..., ge=10, le=300)
    explanation: str = Field(..., min_length=10)
    source_chunk_index: int = Field(..., ge=0)

    @field_validator("options")
    @classmethod
    def exactly_one_correct(cls, v: list[LLMOption]) -> list[LLMOption]:
        correct_count = sum(1 for o in v if o.is_correct)
        if correct_count != 1:
            raise ValueError(
                f"Exactly 1 correct option required, got {correct_count}"
            )
        return v


class LLMGenerationOutput(BaseModel):
    """Top-level schema the LLM must produce (5 questions per chapter)."""

    questions: list[LLMQuestion] = Field(..., min_length=5, max_length=5)
