"""
Game Maker Agent v1 — Configuration Module
Centralizes all configuration. No global mutable state.
All settings are frozen Pydantic models.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Chapter-to-PDF Mapping ───────────────────────────────────────────────────
# Each chapter maps to a specific PDF. The game generates 5 questions per chapter.

CHAPTER_PDF_MAP: dict[str, str] = {
    "1": "/home/oussema/3/Artificial Intelligence, Machine Learning, and Deep Learning.pdf",
    "2": "/home/oussema/3/room2_nlp_llms.pdf",
    "3": "/home/oussema/3/room3_agentic_cybersec (1).pdf",
    "4": "/home/oussema/3/bitcoin_vocabulary.pdf",
}

CHAPTER_TITLES: dict[str, str] = {
    "1": "Artificial Intelligence, Machine Learning, and Deep Learning",
    "2": "Natural Language Processing and Large Language Models",
    "3": "Agentic Cybersecurity",
    "4": "Bitcoin Vocabulary",
}


class LLMSettings(BaseSettings):
    """Groq / OpenAI-compatible LLM settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_", frozen=True)

    api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", ""),
        description="Groq API key",
    )
    base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq-compatible OpenAI base URL",
    )
    model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model identifier",
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature > 0 for varied output on each call",
    )
    max_tokens: int = Field(default=4096)
    top_p: float = Field(default=0.9)


class PDFSettings(BaseSettings):
    """Security constraints for PDF ingestion."""

    model_config = SettingsConfigDict(env_prefix="PDF_", frozen=True)

    allowed_directory: Path = Field(
        default=Path("/home/oussema/3"),
        description="Only PDFs under this directory are allowed",
    )
    max_size_bytes: int = Field(
        default=50 * 1024 * 1024,  # 50 MB
        description="Maximum allowed PDF file size",
    )


class GameSettings(BaseSettings):
    """Game generation constants."""

    model_config = SettingsConfigDict(frozen=True)

    questions_per_chapter: int = 5
    total_chapters: int = 4
    total_questions: int = 20  # 5 * 4
    options_per_question: int = 4
    difficulty_distribution: dict[str, int] = Field(
        default={"medium": 2, "hard": 3},
        description="Per-chapter: 2 medium + 3 hard = 5 questions",
    )
    chunk_max_tokens: int = Field(
        default=1500,
        description="Target max tokens per semantic chunk",
    )
    chunk_overlap_sentences: int = Field(
        default=2,
        description="Sentence overlap between chunks for context continuity",
    )


class AppSettings(BaseSettings):
    """Top-level application settings — composition root."""

    model_config = SettingsConfigDict(frozen=True)

    app_name: str = "Game Maker Agent v1"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    pdf: PDFSettings = Field(default_factory=PDFSettings)
    game: GameSettings = Field(default_factory=GameSettings)


def get_settings() -> AppSettings:
    """Factory — returns a fresh frozen settings instance."""
    return AppSettings()
    debug: bool = False
    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    pdf: PDFSettings = Field(default_factory=PDFSettings)
    game: GameSettings = Field(default_factory=GameSettings)


def get_settings() -> AppSettings:
    """Factory — returns a fresh frozen settings instance."""
    return AppSettings()
