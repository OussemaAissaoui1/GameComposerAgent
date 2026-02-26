"""
Game Maker Agent v1 — LLM Orchestration Service
Uses Groq API (OpenAI-compatible) with llama-3.3-70b-versatile.
Generates 5 hard questions per chapter. Temperature > 0 for variation.
Post-shuffles options to fix LLM's bias toward always making B correct.
"""

from __future__ import annotations

import json
import logging
import random
import re

from openai import OpenAI

from app.config import LLMSettings, GameSettings
from app.models.schemas import LLMGenerationOutput

logger = logging.getLogger(__name__)

# ── Prompt Template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert quiz generation engine for a blockchain-based educational game.

ABSOLUTE CONSTRAINTS:
1. Generate EXACTLY 5 multiple-choice questions.
2. Each question has EXACTLY 4 options labeled A, B, C, D.
3. EXACTLY 1 option per question is correct.
4. ALL questions and answers MUST come ONLY from the provided source text.
5. You MUST NOT use any external knowledge or assumptions.
6. No two questions may test the same fact or concept.
7. No two options within a question may be semantically identical.
8. The correct answer MUST be unambiguously supported by the source text.

QUESTION STYLE — HARD BUT NOT COMPLEX:
- Questions should test UNDERSTANDING, not just memorization.
- Ask "why" and "how" questions, not just "what is".
- Use scenario-based questions: "If X happens, what would Y be?"
- Require the reader to CONNECT concepts from the text.
- Wrong options should be plausible and tricky — not obviously wrong.
- Avoid trivial recall questions like "What does X stand for?"
- Do NOT make questions convoluted or multi-layered — keep them clear but challenging.

DIFFICULTY DISTRIBUTION (MANDATORY):
- Questions 1–2: difficulty "medium" (difficulty_rating 450–650)
- Questions 3–5: difficulty "hard" (difficulty_rating 651–900)

DIFFICULTY GUIDELINES:
- medium: Requires understanding relationships or comparing concepts. min_solve_time: 30–60s.
- hard: Requires synthesis, inference within the text, or applying concepts to scenarios. min_solve_time: 45–90s.

OUTPUT FORMAT — JSON ONLY:
Return ONLY valid JSON matching this exact schema. No markdown, no explanation, no commentary.

{
  "questions": [
    {
      "question_number": 1,
      "question": "...",
      "options": [
        {"option_id": "A", "text": "...", "is_correct": false},
        {"option_id": "B", "text": "...", "is_correct": true},
        {"option_id": "C", "text": "...", "is_correct": false},
        {"option_id": "D", "text": "...", "is_correct": false}
      ],
      "difficulty": "medium",
      "difficulty_rating": 500,
      "min_solve_time_seconds": 40,
      "explanation": "According to the text: '...' — this means that...",
      "source_chunk_index": 0
    }
  ]
}

SELF-CHECK BEFORE RESPONDING:
□ Exactly 5 questions?
□ Each has exactly 4 options (A, B, C, D)?
□ Each has exactly 1 correct option?
□ Questions 1-2 are "medium", 3-5 are "hard"?
□ All content grounded in source text only?
□ Questions test understanding, not just recall?
□ Wrong options are plausible?
□ No duplicate concepts?
□ Correct answers spread across A, B, C, D — NOT always B?
□ Valid JSON output with no extra text?
"""


def _build_user_prompt(
    chunks: list[str],
    chapter_id: str,
    chapter_title: str,
    difficulty_target: int,
) -> str:
    """Build the user prompt with source chunks embedded."""
    chunk_sections: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_sections.append(
            f"--- SOURCE CHUNK {i} ---\n{chunk}\n--- END CHUNK {i} ---"
        )

    joined_chunks = "\n\n".join(chunk_sections)

    return f"""\
TASK: Generate exactly 5 challenging MCQs from Chapter {chapter_id}: "{chapter_title}".
TARGET DIFFICULTY SCORE: {difficulty_target}
NUMBER OF SOURCE CHUNKS: {len(chunks)}

SOURCE TEXT (use ONLY this material):

{joined_chunks}

Generate 5 questions now as a single JSON object. Remember:
- Questions 1-2: medium difficulty (understanding/comparison)
- Questions 3-5: hard difficulty (synthesis/application/inference)
- Make wrong options PLAUSIBLE — not obviously incorrect
- Test UNDERSTANDING, not memorization
- Reference source_chunk_index for each question
- Every answer must be directly supported by the source text above.
- Generate DIFFERENT questions each time — explore different aspects of the text.
"""


def _extract_json_from_response(raw: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_block:
        return json_block.group(1)

    json_obj = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_obj:
        return json_obj.group(0)

    return raw


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def _select_chunks_within_budget(
    chunks: list[str],
    max_input_tokens: int = 6000,
) -> list[str]:
    """Select a subset of chunks that fits the token budget.

    Strategy: evenly sample from across the text to maintain coverage.
    """
    system_overhead = _estimate_tokens(SYSTEM_PROMPT) + 500
    available = max_input_tokens - system_overhead

    total_tokens = sum(_estimate_tokens(c) for c in chunks)

    if total_tokens <= available:
        return chunks

    if len(chunks) <= 2:
        return chunks

    selected: list[str] = []
    running_tokens = 0

    step = max(1, len(chunks) // 10)
    indices = list(range(0, len(chunks), step))
    if indices[-1] != len(chunks) - 1:
        indices.append(len(chunks) - 1)

    for idx in indices:
        chunk_tokens = _estimate_tokens(chunks[idx])
        if running_tokens + chunk_tokens > available:
            break
        selected.append(chunks[idx])
        running_tokens += chunk_tokens

    logger.info(
        "Chunk budget: %d/%d chunks selected (%d/%d est. tokens)",
        len(selected),
        len(chunks),
        running_tokens,
        total_tokens,
    )

    return selected if selected else [chunks[0]]


def _shuffle_options(output: LLMGenerationOutput) -> LLMGenerationOutput:
    """Randomly shuffle option positions for each question.

    LLMs have a strong bias toward placing the correct answer at B.
    This post-processing step randomly reassigns option_ids (A,B,C,D)
    so correct answers are uniformly distributed across all positions.

    Also enforces that across 5 questions, no single letter is the
    correct answer more than twice — ensuring visible variety.
    """
    from app.models.schemas import LLMOption

    LABELS = ["A", "B", "C", "D"]
    used_correct_positions: list[str] = []

    for q in output.questions:
        # Collect option texts and which one is correct
        options_data = [(opt.text, opt.is_correct) for opt in q.options]

        # Try up to 10 shuffles to avoid repeating the same correct position
        for _ in range(10):
            random.shuffle(options_data)
            correct_label = LABELS[
                next(i for i, (_, c) in enumerate(options_data) if c)
            ]
            # Accept if this position isn't overused (max 2 per letter)
            if used_correct_positions.count(correct_label) < 2:
                break

        used_correct_positions.append(correct_label)

        # Rebuild options with new labels
        q.options = [
            LLMOption(
                option_id=LABELS[i],
                text=text,
                is_correct=is_correct,
            )
            for i, (text, is_correct) in enumerate(options_data)
        ]

    # Log distribution for monitoring
    correct_ids = []
    for q in output.questions:
        for opt in q.options:
            if opt.is_correct:
                correct_ids.append(opt.option_id)
    logger.info(
        "Post-shuffle correct answer distribution: %s",
        {lbl: correct_ids.count(lbl) for lbl in LABELS},
    )

    return output


async def generate_questions(
    chunks: list[str],
    chapter_id: str,
    chapter_title: str,
    difficulty_target: int,
    llm_settings: LLMSettings,
    game_settings: GameSettings,
) -> LLMGenerationOutput:
    """Generate 5 MCQs via Groq LLM for one chapter.

    Temperature > 0 means each call produces DIFFERENT questions.
    No fixed seed — every generation is fresh.

    Args:
        chunks: Semantic text chunks from the chapter PDF.
        chapter_id: Chapter identifier.
        chapter_title: Human-readable chapter name.
        difficulty_target: Target difficulty score.
        llm_settings: LLM configuration.
        game_settings: Game rules.

    Returns:
        Parsed and validated LLMGenerationOutput (5 questions).

    Raises:
        ValueError: If LLM output cannot be parsed or validated.
    """
    client = OpenAI(
        api_key=llm_settings.api_key,
        base_url=llm_settings.base_url,
    )

    selected_chunks = _select_chunks_within_budget(chunks)
    user_prompt = _build_user_prompt(
        selected_chunks, chapter_id, chapter_title, difficulty_target
    )

    logger.info(
        "Calling LLM: model=%s, temperature=%s, chapter=%s (%s), chunks=%d/%d",
        llm_settings.model_name,
        llm_settings.temperature,
        chapter_id,
        chapter_title,
        len(selected_chunks),
        len(chunks),
    )

    response = client.chat.completions.create(
        model=llm_settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=llm_settings.temperature,
        max_tokens=llm_settings.max_tokens,
        top_p=llm_settings.top_p,
    )

    raw_content = response.choices[0].message.content
    if not raw_content:
        raise ValueError("LLM returned empty response")

    logger.info("LLM response received: %d characters", len(raw_content))
    logger.debug("Raw LLM output: %s", raw_content[:500])

    # ── Parse JSON ───────────────────────────────────────────────────────
    json_str = _extract_json_from_response(raw_content)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s", e)
        logger.error("Attempted to parse: %s", json_str[:1000])
        raise ValueError(f"LLM output is not valid JSON: {e}") from e

    # ── Validate with Pydantic ───────────────────────────────────────────
    try:
        output = LLMGenerationOutput.model_validate(parsed)
    except Exception as e:
        logger.error("Schema validation failed: %s", e)
        raise ValueError(f"LLM output failed schema validation: {e}") from e

    # ── Shuffle options to fix correct-answer bias ───────────────────────
    output = _shuffle_options(output)

    logger.info(
        "Successfully generated %d questions for chapter %s",
        len(output.questions),
        chapter_id,
    )

    return output
