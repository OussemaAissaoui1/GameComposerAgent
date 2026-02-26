"""
Game Maker Agent v1 — Unit Tests
Tests for each pipeline stage + multi-chapter integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppSettings, GameSettings, LLMSettings, PDFSettings
from app.models.schemas import (
    Difficulty,
    LLMGenerationOutput,
    LLMOption,
    LLMQuestion,
)
from app.services.chunker import chunk_text
from app.services.formatter import format_game_output, ChapterResult
from app.services.pdf_loader import PDFSecurityError, _validate_pdf_path
from app.services.validator import ValidationError, validate_chapter_generation


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def game_settings() -> GameSettings:
    return GameSettings()


@pytest.fixture
def pdf_settings() -> PDFSettings:
    return PDFSettings(allowed_directory=Path("/home/oussema/3"))


@pytest.fixture
def llm_settings() -> LLMSettings:
    return LLMSettings(api_key="test-key")


@pytest.fixture
def app_settings(llm_settings, pdf_settings, game_settings) -> AppSettings:
    return AppSettings(llm=llm_settings, pdf=pdf_settings, game=game_settings)


def _make_question(
    number: int,
    difficulty: str,
    rating: int,
    correct: str = "B",
    chunk_index: int = 0,
) -> LLMQuestion:
    """Factory for a valid LLMQuestion."""
    options = []
    for opt_id in ["A", "B", "C", "D"]:
        options.append(
            LLMOption(
                option_id=opt_id,
                text=f"Option {opt_id} for question {number}",
                is_correct=(opt_id == correct),
            )
        )
    return LLMQuestion(
        question_number=number,
        question=f"This is test question number {number} about a concept?",
        options=options,
        difficulty=Difficulty(difficulty),
        difficulty_rating=rating,
        min_solve_time_seconds=30,
        explanation=f"Explanation for question {number} based on source text.",
        source_chunk_index=chunk_index,
    )


def _make_valid_chapter_output(num_chunks: int = 5) -> LLMGenerationOutput:
    """Build a valid 5-question LLM output (per chapter).

    Distribution: 2 medium + 3 hard.
    """
    questions = [
        # 2 medium (450–650)
        _make_question(1, "medium", 500, chunk_index=0),
        _make_question(2, "medium", 600, chunk_index=1),
        # 3 hard (651–900)
        _make_question(3, "hard", 700, chunk_index=2 % num_chunks),
        _make_question(4, "hard", 800, chunk_index=3 % num_chunks),
        _make_question(5, "hard", 850, chunk_index=4 % num_chunks),
    ]
    return LLMGenerationOutput(questions=questions)


def _make_chapter_results(
    num_chapters: int = 4,
    num_chunks: int = 5,
) -> list[ChapterResult]:
    """Build ChapterResult list for 4 chapters."""
    titles = {
        "1": "AI and ML",
        "2": "NLP and LLMs",
        "3": "Agentic Cybersecurity",
        "4": "Bitcoin Vocabulary",
    }
    results = []
    for i in range(1, num_chapters + 1):
        cid = str(i)
        output = _make_valid_chapter_output(num_chunks)
        # Make questions unique across chapters
        for q in output.questions:
            q.question = f"Ch{cid}: {q.question}"
        results.append(
            ChapterResult(
                chapter_id=cid,
                chapter_title=titles[cid],
                output=output,
            )
        )
    return results


# ── PDF Loader Tests ─────────────────────────────────────────────────────────

class TestPDFLoader:
    """Tests for PDF security and loading."""

    def test_path_traversal_blocked(self, pdf_settings: PDFSettings) -> None:
        """Directory traversal must be rejected."""
        with pytest.raises(PDFSecurityError, match="outside allowed directory"):
            _validate_pdf_path("/etc/passwd", pdf_settings)

    def test_non_pdf_rejected(self, pdf_settings: PDFSettings) -> None:
        """Non-PDF files must be rejected."""
        with pytest.raises(PDFSecurityError):
            _validate_pdf_path("/home/oussema/3/file.txt", pdf_settings)

    def test_nonexistent_pdf_rejected(self, pdf_settings: PDFSettings) -> None:
        """Non-existent files must be rejected."""
        with pytest.raises(PDFSecurityError, match="not found"):
            _validate_pdf_path(
                "/home/oussema/3/nonexistent.pdf", pdf_settings
            )


# ── Chunker Tests ────────────────────────────────────────────────────────────

class TestChunker:
    """Tests for semantic chunking."""

    def test_basic_chunking(self, game_settings: GameSettings) -> None:
        text = "First sentence. " * 100 + "\n\n" + "Second paragraph. " * 100
        chunks = chunk_text(text, game_settings)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)

    def test_determinism(self, game_settings: GameSettings) -> None:
        text = "Hello world. " * 200 + "\n\n" + "Another section. " * 200
        run1 = chunk_text(text, game_settings)
        run2 = chunk_text(text, game_settings)
        assert run1 == run2

    def test_single_sentence(self, game_settings: GameSettings) -> None:
        text = "This is a single sentence about artificial intelligence."
        chunks = chunk_text(text, game_settings)
        assert len(chunks) == 1

    def test_empty_text_fallback(self, game_settings: GameSettings) -> None:
        chunks = chunk_text("   \n\n   ", game_settings)
        assert len(chunks) >= 0


# ── Validator Tests ──────────────────────────────────────────────────────────

class TestValidator:
    """Tests for per-chapter game output validation (5 questions)."""

    def test_valid_chapter_passes(self, game_settings: GameSettings) -> None:
        """A correctly formed 5-question output must pass."""
        output = _make_valid_chapter_output(num_chunks=5)
        result = validate_chapter_generation(
            output, game_settings, num_chunks=5, chapter_id="1"
        )
        assert len(result.questions) == 5

    def test_wrong_question_count_fails(
        self, game_settings: GameSettings
    ) -> None:
        """Non-5 question count must fail."""
        output = _make_valid_chapter_output()
        output.questions = output.questions[:3]
        with pytest.raises(ValidationError, match="Expected 5"):
            validate_chapter_generation(
                output, game_settings, num_chunks=5, chapter_id="1"
            )

    def test_wrong_difficulty_distribution_fails(
        self, game_settings: GameSettings
    ) -> None:
        """Wrong difficulty distribution (e.g. 3 medium + 2 hard) must fail."""
        output = _make_valid_chapter_output()
        # Change Q3 from hard to medium → 3 medium + 2 hard (should be 2+3)
        output.questions[2] = _make_question(3, "medium", 550, chunk_index=0)
        with pytest.raises(ValidationError, match="Difficulty"):
            validate_chapter_generation(
                output, game_settings, num_chunks=5, chapter_id="1"
            )

    def test_duplicate_questions_fail(
        self, game_settings: GameSettings
    ) -> None:
        """Duplicate question text must fail."""
        output = _make_valid_chapter_output()
        output.questions[1].question = output.questions[0].question
        with pytest.raises(ValidationError, match="Duplicate question"):
            validate_chapter_generation(
                output, game_settings, num_chunks=5, chapter_id="1"
            )

    def test_invalid_chunk_index_fails(
        self, game_settings: GameSettings
    ) -> None:
        """Out-of-range chunk index must fail."""
        output = _make_valid_chapter_output(num_chunks=5)
        output.questions[0].source_chunk_index = 99
        with pytest.raises(ValidationError, match="source_chunk_index"):
            validate_chapter_generation(
                output, game_settings, num_chunks=5, chapter_id="1"
            )

    def test_medium_rating_out_of_range_fails(
        self, game_settings: GameSettings
    ) -> None:
        """Medium question with rating outside 450-650 must fail."""
        output = _make_valid_chapter_output()
        output.questions[0].difficulty_rating = 200  # too low for medium
        with pytest.raises(ValidationError, match="difficulty_rating"):
            validate_chapter_generation(
                output, game_settings, num_chunks=5, chapter_id="1"
            )

    def test_hard_rating_out_of_range_fails(
        self, game_settings: GameSettings
    ) -> None:
        """Hard question with rating outside 651-900 must fail."""
        output = _make_valid_chapter_output()
        output.questions[2].difficulty_rating = 950  # too high for hard
        with pytest.raises(ValidationError, match="difficulty_rating"):
            validate_chapter_generation(
                output, game_settings, num_chunks=5, chapter_id="1"
            )


# ── Formatter Tests ──────────────────────────────────────────────────────────

class TestFormatter:
    """Tests for multi-chapter output formatting and anchor generation."""

    def test_20_puzzles_from_4_chapters(
        self, app_settings: AppSettings
    ) -> None:
        """4 chapters × 5 questions = 20 puzzles and 20 keys."""
        results = _make_chapter_results()
        payload = format_game_output(results, 700, app_settings)
        assert len(payload.public_puzzles) == 20
        assert len(payload.private_answer_key) == 20

    def test_public_puzzles_have_chapter_info(
        self, app_settings: AppSettings
    ) -> None:
        """Each public puzzle must have chapter_id and chapter_title."""
        results = _make_chapter_results()
        payload = format_game_output(results, 700, app_settings)
        for puzzle in payload.public_puzzles:
            assert puzzle.chapter_id in ("1", "2", "3", "4")
            assert len(puzzle.chapter_title) > 0

    def test_public_puzzles_have_no_correct_flag(
        self, app_settings: AppSettings
    ) -> None:
        """Public puzzles must NOT contain correctness information."""
        results = _make_chapter_results()
        payload = format_game_output(results, 700, app_settings)
        for puzzle in payload.public_puzzles:
            for option in puzzle.options:
                assert not hasattr(option, "is_correct") or \
                    "is_correct" not in option.model_fields

    def test_private_keys_have_anchors(
        self, app_settings: AppSettings
    ) -> None:
        """Each private key must have a properly formatted anchor string."""
        results = _make_chapter_results()
        payload = format_game_output(results, 700, app_settings)
        for key in payload.private_answer_key:
            assert "|" in key.anchor_string
            puzzle_id, option_id = key.anchor_string.split("|")
            assert puzzle_id == key.puzzle_id
            assert option_id == key.correct_option_id
            assert option_id in ("A", "B", "C", "D")

    def test_puzzle_ids_per_chapter(
        self, app_settings: AppSettings
    ) -> None:
        """Puzzle IDs must follow ch{N}_q{NN} format per chapter."""
        results = _make_chapter_results()
        payload = format_game_output(results, 700, app_settings)
        expected_ids = []
        for ch in range(1, 5):
            for q in range(1, 6):
                expected_ids.append(f"ch{ch}_q{q:02d}")
        actual_ids = [p.puzzle_id for p in payload.public_puzzles]
        assert actual_ids == expected_ids

    def test_meta_fields(self, app_settings: AppSettings) -> None:
        """Meta must reflect multi-chapter settings."""
        results = _make_chapter_results()
        payload = format_game_output(results, 700, app_settings)
        assert payload.meta.chapters == ["1", "2", "3", "4"]
        assert payload.meta.questions_per_chapter == 5
        assert payload.meta.total_questions == 20
        assert payload.meta.difficulty_target == 700
        assert payload.meta.temperature == 0.7

    def test_private_keys_have_chapter_id(
        self, app_settings: AppSettings
    ) -> None:
        """Private keys must include chapter_id."""
        results = _make_chapter_results()
        payload = format_game_output(results, 700, app_settings)
        for key in payload.private_answer_key:
            assert key.chapter_id in ("1", "2", "3", "4")


# ── Anchor Determinism Tests ─────────────────────────────────────────────────

class TestAnchorDeterminism:
    """Anchor strings must be deterministic for same input."""

    def test_anchor_string_format(self) -> None:
        from app.services.formatter import _make_anchor_string
        a = _make_anchor_string("ch1_q01", "B")
        assert a == "ch1_q01|B"

    def test_formatter_determinism(self, app_settings: AppSettings) -> None:
        """Same input to formatter must produce identical output."""
        results = _make_chapter_results()
        run1 = format_game_output(results, 700, app_settings)
        run2 = format_game_output(results, 700, app_settings)
        assert run1.model_dump() == run2.model_dump()


# ── Integration Test (Mocked LLM) ───────────────────────────────────────────

class TestIntegration:
    """End-to-end pipeline test with mocked LLM."""

    def test_full_pipeline_validate_and_format(
        self, app_settings: AppSettings
    ) -> None:
        """Full pipeline: validate per-chapter → merge → format."""
        all_results: list[ChapterResult] = []
        titles = {
            "1": "AI and ML",
            "2": "NLP and LLMs",
            "3": "Agentic Cybersecurity",
            "4": "Bitcoin Vocabulary",
        }

        for ch_id in ("1", "2", "3", "4"):
            output = _make_valid_chapter_output(num_chunks=5)
            # Unique questions per chapter
            for q in output.questions:
                q.question = f"Ch{ch_id}: {q.question}"

            validated = validate_chapter_generation(
                output, app_settings.game, num_chunks=5, chapter_id=ch_id
            )
            all_results.append(
                ChapterResult(
                    chapter_id=ch_id,
                    chapter_title=titles[ch_id],
                    output=validated,
                )
            )

        payload = format_game_output(all_results, 700, app_settings)

        assert payload.meta.total_questions == 20
        assert len(payload.public_puzzles) == 20
        assert len(payload.private_answer_key) == 20

        # Verify no answer leakage in public
        for puzzle in payload.public_puzzles:
            puzzle_dict = puzzle.model_dump()
            for option in puzzle_dict["options"]:
                assert "is_correct" not in option

        # Verify all 4 chapters represented
        chapter_ids_seen = set(p.chapter_id for p in payload.public_puzzles)
        assert chapter_ids_seen == {"1", "2", "3", "4"}
