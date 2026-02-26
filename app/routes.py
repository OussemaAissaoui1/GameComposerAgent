"""
Game Maker Agent v1 — API Routes
Single endpoint: POST /generate-game
Multi-chapter pipeline: 4 PDFs × 5 questions = 20 MCQs.
Non-deterministic: fresh questions every call.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException

from app.config import get_settings, CHAPTER_PDF_MAP, CHAPTER_TITLES
from app.models.schemas import GameRequest, GameResponse, GamePayload
from app.services.pdf_loader import (
    load_pdf_text,
    PDFSecurityError,
    PDFExtractionError,
)
from app.services.chunker import chunk_text
from app.services.llm_generator import generate_questions
from app.services.validator import validate_chapter_generation, ValidationError
from app.services.formatter import format_game_output, ChapterResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate-game", response_model=GameResponse)
async def generate_game(request: GameRequest) -> GameResponse:
    """Generate a 20-question MCQ game from 4 PDF chapters.

    Pipeline (per chapter):
    1. Load PDF → extract full text
    2. Chunk text semantically
    3. Generate 5 MCQs via LLM (non-deterministic)
    4. Validate chapter output
    Then merge all chapters → format public/private separation.

    Non-deterministic: fresh questions every call.
    """
    settings = get_settings()
    start_time = time.monotonic()

    logger.info(
        "EVENT:GAME_GENERATION_STARTED chapters=%d difficulty_target=%d",
        len(CHAPTER_PDF_MAP),
        request.difficulty_target,
    )

    try:
        chapter_results: list[ChapterResult] = []

        for chapter_id, pdf_path in CHAPTER_PDF_MAP.items():
            chapter_title = CHAPTER_TITLES[chapter_id]
            logger.info(
                "STEP:CHAPTER_%s_START title=%s pdf=%s",
                chapter_id,
                chapter_title,
                pdf_path,
            )

            # ── Step 1: PDF Ingestion ────────────────────────────────────
            full_text = load_pdf_text(
                pdf_path=pdf_path,
                settings=settings.pdf,
            )
            logger.info(
                "STEP:CHAPTER_%s_PDF_LOADED chars=%d",
                chapter_id,
                len(full_text),
            )

            # ── Step 2: Semantic Chunking ────────────────────────────────
            chunks = chunk_text(full_text, settings.game)
            logger.info(
                "STEP:CHAPTER_%s_CHUNKED chunks=%d",
                chapter_id,
                len(chunks),
            )

            # ── Step 3: LLM Generation (5 Qs) ───────────────────────────
            llm_output = await generate_questions(
                chunks=chunks,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                difficulty_target=request.difficulty_target,
                llm_settings=settings.llm,
                game_settings=settings.game,
            )
            logger.info(
                "STEP:CHAPTER_%s_GENERATED questions=%d",
                chapter_id,
                len(llm_output.questions),
            )

            # ── Step 4: Validation ───────────────────────────────────────
            validated = validate_chapter_generation(
                output=llm_output,
                settings=settings.game,
                num_chunks=len(chunks),
                chapter_id=chapter_id,
            )
            logger.info("STEP:CHAPTER_%s_VALIDATED", chapter_id)

            chapter_results.append(
                ChapterResult(
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    output=validated,
                )
            )

            # Small delay between Groq calls to respect rate limits
            if chapter_id != list(CHAPTER_PDF_MAP.keys())[-1]:
                await asyncio.sleep(2)

        # ── Step 5: Merge & Format ───────────────────────────────────────
        logger.info("STEP:FORMATTING_START chapters=%d", len(chapter_results))
        game_payload: GamePayload = format_game_output(
            chapter_results=chapter_results,
            difficulty_target=request.difficulty_target,
            settings=settings,
        )
        logger.info("STEP:FORMATTING_COMPLETE")

        elapsed = time.monotonic() - start_time
        logger.info(
            "EVENT:GAME_GENERATION_SUCCESS total_questions=%d elapsed=%.2fs",
            len(game_payload.public_puzzles),
            elapsed,
        )

        return GameResponse(status="success", game=game_payload)

    except PDFSecurityError as e:
        logger.error("EVENT:PDF_SECURITY_ERROR %s", e)
        raise HTTPException(status_code=403, detail=str(e)) from e

    except PDFExtractionError as e:
        logger.error("EVENT:PDF_EXTRACTION_ERROR %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    except ValidationError as e:
        logger.error("EVENT:VALIDATION_ERROR violations=%s", e.violations)
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Game generation failed validation",
                "violations": e.violations,
            },
        ) from e

    except ValueError as e:
        logger.error("EVENT:LLM_ERROR %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            logger.error("EVENT:AUTH_ERROR Invalid API key")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing GROQ_API_KEY. Set the environment variable.",
            ) from e
        logger.exception("EVENT:UNEXPECTED_ERROR %s", e)
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from e
